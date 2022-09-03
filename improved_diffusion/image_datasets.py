import string, os, random, json
from PIL import Image
import blobfile as bf
# from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset, BatchSampler, RandomSampler, SequentialSampler
import torch as th
import torch.nn.functional as F
import torchvision.transforms as T
from .crop import RandomResizedProtectedCropLazy
from .dist_util import FakeMPI
MPI = FakeMPI()

import tokenizers
from tqdm.auto import trange
from tqdm.contrib.concurrent import thread_map

import imagesize

import clip

def make_char_level_tokenizer(legacy_padding_behavior=True):
    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token="<unk>"))
    if legacy_padding_behavior:
        trainer = tokenizers.trainers.BpeTrainer(special_tokens=["<s>", "</s>", "<unk>", "<pad>", "<mask>"])
    else:
        trainer = tokenizers.trainers.BpeTrainer(special_tokens=["<pad>", "</s>", "<s>", "<unk>", "<mask>"])
    tokenizer.train_from_iterator([[c] for c in string.printable], trainer)
    tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
        "<s> $0 </s>", special_tokens=[("<s>", tokenizer.token_to_id('<s>')), ("</s>", tokenizer.token_to_id('</s>'))]
    )
    return tokenizer


def load_tokenizer(tokenizer_path  = "tokenizer_file", max_seq_len=64, char_level=False, legacy_padding_behavior=True):
    if char_level:
        tokenizer = make_char_level_tokenizer(legacy_padding_behavior=legacy_padding_behavior)
    else:
        tokenizer = tokenizers.Tokenizer.from_file(tokenizer_path)
    tokenizer.enable_truncation(max_seq_len)

    pad_id = 0 if legacy_padding_behavior else tokenizer.token_to_id('<pad>')
    tokenizer.enable_padding(pad_id=pad_id)
    return tokenizer


def tokenize(tokenizer, txt):
    return [t.ids for t in tokenizer.encode_batch(txt)]


def clip_pkeep(probs, middle_pkeep=0.5):
    return probs[2] + middle_pkeep * probs[1]


def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False,
    txt=False, monochrome=False, offset=0, min_filesize=0,
    txt_pdrop=0., txt_drop_string='<mask><mask><mask><mask>',
    crop_prob=0., crop_min_scale=0.75, crop_max_scale=1.,
    use_special_crop_for_empty_string=False,
    crop_prob_es=0., crop_min_scale_es=0.25, crop_max_scale_es=1.,
    crop_without_resize=False,
    safebox_path="",
    use_random_safebox_for_empty_string=False,
    flip_lr_prob_es=0.,
    px_scales_path="",
    return_dataset=False,
    pin_memory=False,
    prefetch_factor=2,
    num_workers=1,
    min_imagesize=0,
    capt_path="",
    capt_pdrop=0.1,
    require_capts=False,
    all_pdrop=0.1,
    class_map_path=None,
    class_ix_unk=0,
    class_ix_drop=999,
    class_pdrop=0.1,
    clip_prob_path=None,
    clip_prob_middle_pkeep=0.5,
    exclusions_data_path=None,
    tokenizer=None,
    debug=False,
    max_workers_dir_scan=32,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    safeboxes = None
    if safebox_path and os.path.exists(safebox_path):
        print('using safebox_path')
        with open(safebox_path, 'r') as f:
            safeboxes = json.load(f)

    px_scales = None
    if px_scales_path and os.path.exists(px_scales_path):
        print('using px_scales_path')
        with open(px_scales_path, 'r') as f:
            px_scales = json.load(f)

    capts = None
    if capt_path and os.path.exists(capt_path):
        print('using capt_path')
        with open(capt_path, 'r') as f:
            capts = json.load(f)

    class_map = None
    if class_map_path and os.path.exists(class_map_path):
        print('using class_map_path')
        with open(class_map_path, 'r') as f:
            class_map = json.load(f)

        all_class_values = set(class_map.values())
        if class_ix_unk in all_class_values:
            raise ValueError(f"passed {class_ix_unk} as class_ix_unk, but it's used in class map")
        if (class_pdrop > 0) and (class_ix_drop in all_class_values):
            raise ValueError(f"passed {class_ix_drop} as class_ix_drop, but it's used in class map")

    clip_probs = None
    if clip_prob_path and os.path.exists(clip_prob_path):
        print('using clip_prob_path')
        with open(clip_prob_path, 'r') as f:
            clip_probs = json.load(f)

    excluded_paths = None
    if exclusions_data_path and os.path.exists(exclusions_data_path):
        print('using exclusions_data_path')
        with open(exclusions_data_path, 'r') as f:
            exclusions_data = json.load(f)
        excluded_paths = set(exclusions_data['excluded'])

    all_files, image_file_to_text_file, file_sizes, image_file_to_safebox, image_file_to_px_scales, image_file_to_capt = _list_image_files_recursively(data_dir, txt=txt, min_filesize=min_filesize, min_imagesize=min_imagesize, safeboxes=safeboxes, px_scales=px_scales, capts=capts, require_capts=require_capts, excluded_paths=excluded_paths, max_workers=max_workers)
    print(f"found {len(all_files)} images, {len(image_file_to_text_file)} texts, {len(image_file_to_capt)} capts")
    all_files = all_files[offset:]

    n_texts = sum(1 for k in file_sizes.keys() if k.endswith('.txt'))  # sanity check
    nonempty_text_files = {k for k in file_sizes.keys() if k.endswith('.txt') and file_sizes[k] > 0}
    n_nonempty_texts = len(nonempty_text_files)
    # n_nonempty_texts = sum(file_sizes[k] > 0 for k in file_sizes.keys() if k.endswith('.txt'))
    n_empty_texts = n_texts - n_nonempty_texts

    if n_texts > 0:
        text_file_to_image_file = {v: k for k, v in image_file_to_text_file.items()}  # computed for logging
        n_with_safebox = sum(text_file_to_image_file[k] in image_file_to_safebox for k in nonempty_text_files)

        frac_empty = n_empty_texts/n_texts
        frac_nonempty = n_nonempty_texts/n_texts

        print(f"of {n_texts} texts, {n_empty_texts} ({frac_empty:.1%}) are empty, {n_nonempty_texts} ({frac_nonempty:.1%}) are nonempty")
        print(f"of {n_nonempty_texts} nonempty texts, {n_with_safebox} have safeboxes (all safeboxes: {len(image_file_to_safebox)})")

    if px_scales is not None:
        n_with_px_scale = len(set(text_file_to_image_file.values()).intersection(image_file_to_px_scales.keys()))
        print(f"of {n_texts} texts, {n_with_px_scale} have px scales (all px scales: {len(image_file_to_px_scales)})")

    n_images_with_capts = len(set(all_files).intersection(image_file_to_capt.keys()))
    print(f"of {len(all_files)} images, {n_images_with_capts} have capts (all capts: {len(image_file_to_capt)})")

    if clip_probs is not None:
        n_images_with_clip_probs = len(set(all_files).intersection(clip_probs.keys()))
        print(f"of {len(all_files)} images, {n_images_with_clip_probs} have clip_probs (all clip_probs: {len(clip_probs)})")

    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        if class_map is not None:
            classes = [class_map.get(x, class_ix_unk) for x in class_names]
        else:
            sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
            classes = [sorted_classes[x] for x in class_names]

    pre_resize_transform = None
    pre_resize_transform_for_empty_string = []

    if crop_prob > 0:
        print("using crop")
        if safeboxes is not None and (not crop_without_resize):
            print('using safebox crop')
            imode, tsize = (T.functional.InterpolationMode.BICUBIC, (image_size,))
            def safebox_crop(img, safebox, pre_applied_rescale_factor):
                tform = RandomResizedProtectedCropLazy(size=tsize, min_area=crop_min_scale, max_area=crop_max_scale, interpolation=imode, debug=debug)
                if random.random() < crop_prob:
                    return tform(img, safebox, pre_applied_rescale_factor=pre_applied_rescale_factor)
                return img
            pre_resize_transform = safebox_crop
            if (not use_special_crop_for_empty_string) or (crop_prob_es <= 0):
                use_special_crop_for_empty_string = True
                crop_prob_es = crop_prob
                crop_min_scale_es = crop_min_scale
                crop_max_scale_es = crop_max_scale
        else:
            imode, tsize = (T.functional.InterpolationMode.BICUBIC, (image_size,))
            if crop_without_resize:
                cropper = T.RandomCrop(size=tsize)
            else:
                cropper = T.RandomResizedCrop(
                    size=tsize, ratio=(1, 1), scale=(crop_min_scale, crop_max_scale), interpolation=imode
                )
            pre_resize_transform = T.RandomApply(
                transforms=[
                    cropper,
                ],
                p=crop_prob
            )

    use_es_crop = use_special_crop_for_empty_string and (crop_prob_es > 0)
    use_es_regular_crop = use_es_crop and (not use_random_safebox_for_empty_string)

    if use_es_crop:
        print('using es crop')

    if use_es_regular_crop:
        print("using es regular crop")
        imode, tsize = (T.functional.InterpolationMode.BICUBIC, (image_size,))
        if crop_without_resize:
            cropper = T.RandomCrop(size=tsize)
        else:
            cropper = T.RandomResizedCrop(
                size=tsize, ratio=(1, 1), scale=(crop_min_scale_es, crop_max_scale_es), interpolation=imode
            )
        pre_resize_transform_for_empty_string.append(
            T.RandomApply(
                transforms=[
                    cropper,
                ],
                p=crop_prob_es
            )
        )

    if flip_lr_prob_es > 0:
        print("using flip")
        pre_resize_transform_for_empty_string.append(T.RandomHorizontalFlip(p=flip_lr_prob_es))

    if len(pre_resize_transform_for_empty_string) > 0:
        pre_resize_transform_for_empty_string = T.Compose(pre_resize_transform_for_empty_string)
    else:
        pre_resize_transform_for_empty_string = None

    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        image_file_to_text_file=image_file_to_text_file,
        txt=txt,
        monochrome=monochrome,
        file_sizes=file_sizes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        txt_pdrop=txt_pdrop,
        txt_drop_string=txt_drop_string,
        pre_resize_transform=pre_resize_transform,
        pre_resize_transform_for_empty_string=pre_resize_transform_for_empty_string,
        image_file_to_safebox=image_file_to_safebox,
        use_random_safebox_for_empty_string=use_random_safebox_for_empty_string,
        image_file_to_px_scales=image_file_to_px_scales,
        image_file_to_capt=image_file_to_capt,
        capt_pdrop=capt_pdrop,
        all_pdrop=all_pdrop,
        class_ix_drop=class_ix_drop,
        class_pdrop=class_pdrop,
        tokenizer=tokenizer,
    )
    if return_dataset:
        return dataset
    clip_probs_by_idxs = None
    if clip_probs is not None:
        clip_probs_by_idxs = {
            i: clip_probs.get(p)
            for i, p in enumerate(dataset.local_images)
            if p in clip_probs
        }
        print(f"len(clip_probs_by_idxs): {len(clip_probs_by_idxs)}")
        avg_pkeep = np.mean([clip_pkeep(p, middle_pkeep=clip_prob_middle_pkeep) for p in clip_probs_by_idxs.values()])
        eff_len = avg_pkeep * len(dataset)
        eff_steps_per = eff_len / batch_size
        print(f"avg_pkeep {avg_pkeep:.3f} | effective data size {eff_len:.1f} | effective steps/epoch {eff_steps_per:.1f}")
    return _dataloader_gen(dataset, batch_size=batch_size, deterministic=deterministic, pin_memory=pin_memory,
                           prefetch_factor=prefetch_factor,
                           clip_probs_by_idxs=clip_probs_by_idxs,
                           clip_prob_middle_pkeep=clip_prob_middle_pkeep,
                           num_workers=num_workers)


class DropSampler(BatchSampler):
    def __init__(self, sampler, batch_size: int, drop_last: bool, clip_probs_by_idxs: dict, clip_prob_middle_pkeep=0.5):
        super().__init__(sampler, batch_size, drop_last)
        self.clip_probs_by_idxs = clip_probs_by_idxs
        self.clip_prob_middle_pkeep = clip_prob_middle_pkeep

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            if idx in self.clip_probs_by_idxs:
                this_probs = self.clip_probs_by_idxs[idx]
                pkeep = clip_pkeep(this_probs, middle_pkeep=self.clip_prob_middle_pkeep)
                if random.random() > pkeep:
                    continue

            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch


def seeding_worker_init_fn(worker_id):
    seed_th = th.utils.data.get_worker_info().seed
    seed_short = seed_th % (2**32 - 3)
    random.seed(seed_short + 1)
    np.random.seed(seed_short + 2)


def _dataloader_gen(dataset, batch_size, deterministic, pin_memory, prefetch_factor,
                    clip_probs_by_idxs=None,
                    clip_prob_middle_pkeep=0.5,
                    num_workers=1):
    print(f'_dataloader_gen: deterministic={deterministic}')
    kwargs = dict(batch_size=batch_size, drop_last=True, shuffle=not deterministic, )
    if clip_probs_by_idxs is not None:
        if not deterministic:
            sampler = RandomSampler(dataset, generator=None)
        else:
            sampler = SequentialSampler(dataset)
        batch_sampler = DropSampler(sampler=sampler, batch_size=batch_size, drop_last=True, clip_probs_by_idxs=clip_probs_by_idxs, clip_prob_middle_pkeep=clip_prob_middle_pkeep)
        kwargs = dict(batch_sampler=batch_sampler)

    loader = DataLoader(
        dataset, num_workers=num_workers, pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        worker_init_fn=seeding_worker_init_fn,
        **kwargs
    )
    while True:
        yield from loader


def load_superres_data(data_dir, batch_size, large_size, small_size, class_cond=False, txt=False, monochrome=False,
                       deterministic=False, offset=0, colorize=False,
                       blur_prob=0., blur_sigma_min=0.4, blur_sigma_max=0.6,
                       blur_width=5,  # paper used 3, i later learned. though that was for 64 -> 128 and 64 -> 256
                       min_filesize=0,
                       txt_pdrop=0., txt_drop_string='<mask><mask><mask><mask>',
                       crop_prob=0., crop_min_scale=0.75, crop_max_scale=1.,
                       use_special_crop_for_empty_string=False,
                       crop_prob_es=0., crop_min_scale_es=0.25, crop_max_scale_es=1.,
                       crop_without_resize=False,
                       safebox_path="",
                       use_random_safebox_for_empty_string=False,
                       flip_lr_prob_es=0.,
                       px_scales_path="",
                       pin_memory=False,
                       prefetch_factor=2,
                       num_workers=1,
                       min_imagesize=0,
                       clip_prob_path=None,
                       clip_prob_middle_pkeep=0.5,
                       capt_path="",
                       capt_pdrop=0.1,
                       require_capts=False,
                       all_pdrop=0.1,
                       class_map_path=None,
                       class_ix_unk=0,
                       class_ix_drop=999,
                       class_pdrop=0.1,
                       exclusions_data_path=None,
                       tokenizer=None,
                       antialias=False,
                       bicubic_down=False,
                       max_workers_dir_scan=32,
                       ):
    print(f'load_superres_data: deterministic={deterministic}')
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=large_size,
        class_cond=class_cond,
        txt=txt,
        monochrome=monochrome,
        deterministic=deterministic,
        offset=offset,
        min_filesize=min_filesize,
        txt_pdrop=txt_pdrop,
        txt_drop_string=txt_drop_string,
        crop_prob=crop_prob,
        crop_min_scale=crop_min_scale,
        crop_max_scale=crop_max_scale,
        use_special_crop_for_empty_string=use_special_crop_for_empty_string,
        crop_prob_es=crop_prob_es,
        crop_min_scale_es=crop_min_scale_es,
        crop_max_scale_es=crop_max_scale_es,
        crop_without_resize=crop_without_resize,
        safebox_path=safebox_path,
        use_random_safebox_for_empty_string=use_random_safebox_for_empty_string,
        flip_lr_prob_es=flip_lr_prob_es,
        px_scales_path=px_scales_path,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        num_workers=num_workers,
        min_imagesize=min_imagesize,
        clip_prob_path=clip_prob_path,
        clip_prob_middle_pkeep=clip_prob_middle_pkeep,
        capt_path=capt_path,
        capt_pdrop=capt_pdrop,
        require_capts=require_capts,
        all_pdrop=all_pdrop,
        class_map_path=class_map_path,
        class_ix_unk=class_ix_unk,
        class_ix_drop=class_ix_drop,
        class_pdrop=class_pdrop,
        exclusions_data_path=exclusions_data_path,
        tokenizer=tokenizer,
        max_workers_dir_scan=max_workers_dir_scan,
    )

    blurrer = T.RandomApply(transforms=[T.GaussianBlur(blur_width, sigma=(blur_sigma_min, blur_sigma_max))], p=blur_prob)

    is_power_of_2 = False
    top = large_size
    while top > small_size:
        top = top // 2
        if top == small_size:
            is_power_of_2 = True

    print(f"is_power_of_2: {is_power_of_2}")
    mode = "area" if is_power_of_2 else "bilinear"
    use_antialias = False

    if antialias:
        use_antialias = True
        mode = "bilinear"

    if bicubic_down:
        mode = 'bicubic'

    for large_batch, model_kwargs in data:
        model_kwargs["low_res"] = F.interpolate(large_batch, small_size, mode=mode, antialias=use_antialias)
        if colorize:
            model_kwargs["low_res"] = model_kwargs["low_res"].mean(dim=1, keepdim=True)
        if blur_prob > 0:
            model_kwargs["low_res"] = th.stack([blurrer(im) for im in model_kwargs["low_res"]], dim=0)

        yield large_batch, model_kwargs


def _list_image_files_recursively(data_dir, txt=False, min_filesize=0, min_imagesize=0, safeboxes=None, px_scales=None, capts=None, require_capts=False, excluded_paths=None, max_workers=32):
    results = []
    image_file_to_text_file = {}
    file_sizes = {}
    image_file_to_safebox = {}
    image_file_to_px_scales = {}
    image_file_to_capt = {}
    if safeboxes is None:
        safeboxes = {}
    if px_scales is None:
        px_scales = {}
    if capts is None:
        capts = {}
    if excluded_paths is None:
        excluded_paths = set()
    n_excluded_filesize = {'n': 0}
    n_excluded_imagesize = {'n': 0}
    n_excluded_path = {'n': 0}
    n_capts = {'n': 0}
    subdirectories = []
    def scan_entry(entry):
        full_path = bf.join(data_dir, entry)

        if full_path in excluded_paths:
            n_excluded_path['n'] += 1
            return

        prefix, _, ext = entry.rpartition(".")
        safebox_key = prefix.replace('/', '_')

        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            if require_capts and (safebox_key not in capts):
                return

            n_capts['n'] += int(safebox_key in capts)

            if min_filesize > 0:
                filesize = os.path.getsize(full_path)
                if filesize < min_filesize:
                    n_excluded_filesize['n'] += 1
                    return
                file_sizes[full_path] = filesize

            image_file_to_capt[full_path] = capts.get(safebox_key)

            if min_imagesize > 0:
                wh = imagesize.get(full_path)
                pxs = px_scales.get(safebox_key, (1, 1))
                edge = min(wh[0]/max(1, pxs[0]), wh[1]/max(pxs[1], 1))
                if edge < min_imagesize:
                    n_excluded_imagesize['n'] += 1
                    return
            results.append(full_path)
            if txt:
                prefix, _, ext = full_path.rpartition(".")
                path_txt = prefix + ".txt"
                # print(f'made path_txt={repr(path_txt)} from {repr(entry)}')

                if bf.exists(path_txt):
                    image_file_to_text_file[full_path] = path_txt
                    filesize = os.path.getsize(path_txt)
                    file_sizes[path_txt] = filesize

                    image_file_to_safebox[full_path] = safeboxes.get(safebox_key)
                    image_file_to_px_scales[full_path] = px_scales.get(safebox_key)
                else:
                    pass
                    # raise ValueError(path_txt)

        elif bf.isdir(full_path):
            subdirectories.append(full_path)

        # for entry in sorted(bf.listdir(data_dir)):
    if max_workers > 1:
        thread_map(scan_entry, sorted(bf.listdir(data_dir)), max_workers=32)
    else:
        for entry in tqdm(sorted(bf.listdir(data_dir)), mininterval=0.5):
            scan_entry(entry)

    n_excluded_filesize = n_excluded_filesize['n']
    n_excluded_imagesize = n_excluded_imagesize['n']
    n_excluded_path = n_excluded_path['n']
    n_capts = n_capts['n']

    for full_path in subdirectories:
        next_results, next_map, next_file_sizes, next_image_file_to_safebox, next_image_file_to_px_scales, next_image_file_to_capt = _list_image_files_recursively(
            full_path, txt=txt, min_filesize=min_filesize, min_imagesize=min_imagesize, safeboxes=safeboxes, px_scales=px_scales, capts=capts, require_capts=require_capts, excluded_paths=excluded_paths, max_workers=max_workers,
        )
        results.extend(next_results)
        image_file_to_text_file.update(next_map)
        file_sizes.update(next_file_sizes)
        image_file_to_safebox.update(next_image_file_to_safebox)
        image_file_to_px_scales.update(next_image_file_to_px_scales)
        image_file_to_capt.update(next_image_file_to_capt)
    print(f"_list_image_files_recursively: data_dir={data_dir}, n_excluded_filesize={n_excluded_filesize}, n_excluded_imagesize={n_excluded_imagesize},\n\tn_excluded_path={n_excluded_path}, n_capts={n_capts}")
    image_file_to_safebox = {k: v for k, v in image_file_to_safebox.items() if v is not None}
    image_file_to_px_scales = {k: v for k, v in image_file_to_px_scales.items() if v is not None}
    image_file_to_capt = {k: v for k, v in image_file_to_capt.items() if v is not None}
    return results, image_file_to_text_file, file_sizes, image_file_to_safebox, image_file_to_px_scales, image_file_to_capt


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths,
                 classes=None,
                 image_file_to_text_file=None,
                 txt=False,
                 monochrome=False,
                 file_sizes=None,
                 shard=0, num_shards=1,
                 txt_pdrop=0.,
                 txt_drop_string='<mask><mask><mask><mask>',
                 empty_string_to_drop_string=False,  # unconditional != no text
                 pre_resize_transform=None,
                 pre_resize_transform_for_empty_string=None,
                 image_file_to_safebox=None,
                 use_random_safebox_for_empty_string=False,
                 image_file_to_px_scales=None,
                 image_file_to_capt=None,
                 capt_pdrop=0.1,
                 capt_drop_string='unknown',
                 all_pdrop=0.1,
                 class_ix_drop=999,
                 class_pdrop=0.1,
                 tokenizer=None,
                 ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.txt = txt
        self.monochrome = monochrome
        self.file_sizes = file_sizes
        self.txt_pdrop = txt_pdrop
        self.txt_drop_string = txt_drop_string
        self.empty_string_to_drop_string = empty_string_to_drop_string
        self.pre_resize_transform = pre_resize_transform
        if pre_resize_transform_for_empty_string is None:
            pre_resize_transform_for_empty_string = pre_resize_transform
        self.pre_resize_transform_for_empty_string = pre_resize_transform_for_empty_string
        self.image_file_to_safebox = image_file_to_safebox
        if len(self.image_file_to_safebox) == 0:
            self.image_file_to_safebox = None
        self.use_random_safebox_for_empty_string = use_random_safebox_for_empty_string

        self.image_file_to_px_scales = image_file_to_px_scales
        if self.image_file_to_px_scales is None:
            self.image_file_to_px_scales = {}

        self.image_file_to_capt = image_file_to_capt
        self.using_capts = image_file_to_capt is not None
        if self.image_file_to_capt is None:
            self.image_file_to_capt = {}
        self.capt_pdrop = capt_pdrop
        self.capt_drop_string = capt_drop_string
        self.all_pdrop = all_pdrop
        self.class_ix_drop = class_ix_drop
        self.class_pdrop = class_pdrop

        self.tokenizer = tokenizer

        if (self.image_file_to_safebox is not None) and (self.pre_resize_transform is None):
            raise ValueError

        print(f"ImageDataset: self.pre_resize_transform={self.pre_resize_transform}")
        print(f"ImageDataset: self.pre_resize_transform_for_empty_string={self.pre_resize_transform_for_empty_string}")

        if image_file_to_safebox is not None:
            self.safebox_keys = list(image_file_to_safebox.keys())

        if self.txt:
            self.local_images = [p for p in self.local_images if p in image_file_to_text_file]
            self.local_texts = [image_file_to_text_file[p] for p in self.local_images]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        text, capt = None, None
        if self.txt:
            path_txt = self.local_texts[idx]
            with bf.BlobFile(path_txt, "r") as f:
                text = f.read()

        if not self.txt:
            if self.pre_resize_transform_for_empty_string is not None:
                pil_image = self.pre_resize_transform_for_empty_string(pil_image)
            elif self.pre_resize_transform is not None:
                pil_image = self.pre_resize_transform(pil_image)

        if self.txt and len(text) == 0:
            if self.pre_resize_transform_for_empty_string is not None:
                # eg lr flip -- this stacks on top of random safebox crop
                pil_image = self.pre_resize_transform_for_empty_string(pil_image)
            if self.use_random_safebox_for_empty_string and (self.image_file_to_safebox is not None):
                safebox = self.image_file_to_safebox[random.choice(self.safebox_keys)]
                px_scale = self.image_file_to_px_scales.get(path)
                pil_image = self.pre_resize_transform(pil_image, safebox, px_scale)
        elif self.txt:
            if self.image_file_to_safebox is not None:
                if path in self.image_file_to_safebox:
                    safebox = self.image_file_to_safebox[path]
                    px_scale = self.image_file_to_px_scales.get(path)
                    pil_image = self.pre_resize_transform(pil_image, safebox, px_scale)
            elif self.pre_resize_transform is not None:
                pil_image = self.pre_resize_transform(pil_image)

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        mode = "L" if self.monochrome else "RGB"
        arr = np.array(pil_image.convert(mode))
        if self.monochrome:
            arr = np.expand_dims(arr, 2)
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        arr = arr.astype(np.float32) / 127.5 - 1

        drop_txt = (self.txt_pdrop > 0) and (random.random() < self.txt_pdrop)
        drop_capt = (self.capt_pdrop > 0) and (random.random() < self.capt_pdrop)

        if (self.all_pdrop > 0) and (random.random() < self.all_pdrop):
            drop_txt = True
            drop_capt = True

        out_dict = {}
        if self.local_classes is not None:
            drop_class = (self.class_pdrop > 0) and (random.random() < self.class_pdrop)
            this_class = self.class_ix_drop if drop_class else self.local_classes[idx]
            out_dict["y"] = np.array(this_class, dtype=np.int64)

        if drop_txt:
            text = self.txt_drop_string
        if text is not None and (len(text) == 0) and self.empty_string_to_drop_string:
            text = self.txt_drop_string

        if self.txt:
            out_dict['txt'] = text
            # TODO: (low impact) tokenizer in dataloader

        capt = self.image_file_to_capt.get(path, self.capt_drop_string)
        if isinstance(capt, list):
            capt = random.choice(capt)
        if drop_capt:
            capt = self.capt_drop_string

        if self.using_capts:
            # out_dict['capt'] = capt
            out_dict['capt'] = clip.tokenize(capt, truncate=True)[0, :]

        return np.transpose(arr, [2, 0, 1]), out_dict


def to_visible(img):
    img = ((img + 1) * 127.5).clamp(0, 255).to(th.uint8)
    img = img.permute(0, 2, 3, 1)
    img = img.contiguous()
    return img


def save_first_batch(dataloader, path):
    from clip.clip import _tokenizer

    os.makedirs(path, exist_ok=True)
    batch, cond = next(dataloader)
    batch = to_visible(batch)
    txts = cond.get('txt')
    capts = cond.get('capt')
    ys = cond.get('y')

    low_ress = cond.get('low_res')
    if low_ress is not None:
        low_ress = to_visible(low_ress)

    if txts is not None and all(s == '' for s in txts):
        txts = None

    if capts is not None:
        capts = [_tokenizer.decode(c[1:c.argmax()]) for c in capts.cpu().numpy()]

    if capts is not None and all(s == '' for s in capts):
        capts = None

    for i in trange(len(batch)):
        img = batch[i]
        a = img.cpu().numpy()
        im = Image.fromarray(a)

        y = None
        if ys is not None:
            y = ys[i]
            y = y.cpu().numpy()
        y_segment = '_' + str(y) if y is not None else ''

        im.save(os.path.join(path, f'{i:04d}{y_segment}.png'))

        if low_ress is not None:
            low_res = low_ress[i]

            a = low_res.cpu().numpy()
            im = Image.fromarray(a)
            im.save(os.path.join(path, f'{i:04d}{y_segment}_lowres.png'))

        if txts is not None:
            txt = txts[i]

            with open(os.path.join(path, f'{i:04d}{y_segment}.txt'), 'w') as f:
                f.write(txt)

        if capts is not None:
            capt = capts[i]
            with open(os.path.join(path, f'{i:04d}{y_segment}_capt.txt'), 'w') as f:
                f.write(capt)
