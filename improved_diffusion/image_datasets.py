import string
from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchvision.transforms as T

import tokenizers


def make_char_level_tokenizer():
    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token="<unk>"))
    trainer = tokenizers.trainers.BpeTrainer(special_tokens=["<s>", "</s>", "<unk>", "<pad>", "<mask>"])
    tokenizer.train_from_iterator([[c] for c in string.printable], trainer)
    tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
        "<s> $0 </s>", special_tokens=[("<s>", 0), ("</s>", 1)]
    )
    return tokenizer


def load_tokenizer(tokenizer_path  = "tokenizer_file", max_seq_len=64, char_level=False):
    if char_level:
        tokenizer = make_char_level_tokenizer()
    else:
        tokenizer = tokenizers.Tokenizer.from_file(tokenizer_path)
    tokenizer.enable_truncation(max_seq_len)
    tokenizer.enable_padding()
    return tokenizer


def tokenize(tokenizer, txt):
    return [t.ids for t in tokenizer.encode_batch(txt)]


def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False,
    txt=False, monochrome=False, offset=0
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
    all_files, image_file_to_text_file = _list_image_files_recursively(data_dir, txt=txt)
    print(f"found {len(all_files)} images, {len(image_file_to_text_file)} texts")
    all_files = all_files[offset:]

    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for vpath in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        image_file_to_text_file=image_file_to_text_file,
        txt=txt,
        monochrome=monochrome,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True,
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True,
        )
    while True:
        yield from loader


def load_superres_data(data_dir, batch_size, large_size, small_size, class_cond=False, txt=False, monochrome=False,
                       deterministic=False, offset=0, colorize=False, blur_prob=0.):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=large_size,
        class_cond=class_cond,
        txt=txt,
        monochrome=monochrome,
        deterministic=deterministic,
        offset=offset
    )
    for large_batch, model_kwargs in data:
        blurrer = T.RandomApply(transforms=[T.GaussianBlur(5, sigma=(0.4, 1.4))], p=blur_prob)

        model_kwargs["low_res"] = F.interpolate(large_batch, small_size, mode="area")
        if colorize:
            model_kwargs["low_res"] = model_kwargs["low_res"].mean(dim=1, keepdim=True)
        if blur_prob > 0:
            model_kwargs["low_res"] = blurrer(model_kwargs["low_res"])

        yield large_batch, model_kwargs


def _list_image_files_recursively(data_dir, txt=False):
    results = []
    image_file_to_text_file = {}
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
            if txt:
                prefix, _, ext = full_path.rpartition(".")
                path_txt = prefix + ".txt"
                # print(f'made path_txt={repr(path_txt)} from {repr(entry)}')
                if bf.exists(path_txt):
                    image_file_to_text_file[full_path] = path_txt
                else:
                    pass
                    # raise ValueError(path_txt)
        elif bf.isdir(full_path):
            next_results, next_map = _list_image_files_recursively(full_path, txt=txt)
            results.extend(next_results)
            image_file_to_text_file.update(next_map)
    return results, image_file_to_text_file


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths,
                 classes=None,
                 image_file_to_text_file=None,
                 txt=False,
                 monochrome=False,
                 shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.txt = txt
        self.monochrome = monochrome

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

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        if self.txt:
            path_txt = self.local_texts[idx]
            with bf.BlobFile(path_txt, "r") as f:
                text = f.read()
            out_dict['txt'] = text
        return np.transpose(arr, [2, 0, 1]), out_dict
