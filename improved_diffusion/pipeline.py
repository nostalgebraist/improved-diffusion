import argparse
import os
from typing import Union, List, Optional

import numpy as np
import torch as th
import torch.nn as nn
import torch.distributed as dist
import blobfile as bf

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    load_config_to_model,
)
from improved_diffusion.image_datasets import load_tokenizer, tokenize

from improved_diffusion.unet import UNetModel
from improved_diffusion.respace import SpacedDiffusion


def _strip_space(s):
    return "\n".join([part.strip(" ") for part in s.split("\n")])


class SamplingModel(nn.Module):
    def __init__(
        self,
        model: UNetModel,
        diffusion_factory,
        tokenizer,
        timestep_respacing,
        is_super_res=False,
    ):
        super().__init__()
        self.model = model
        self.diffusion_factory = diffusion_factory
        self.tokenizer = tokenizer
        self.is_super_res = is_super_res

        self.set_timestep_respacing(timestep_respacing)

    def set_timestep_respacing(self, timestep_respacing):
        self.diffusion = self.diffusion_factory(timestep_respacing)

    @staticmethod
    def from_config(checkpoint_path, config_path, timestep_respacing=""):
        model, diffusion_factory, tokenizer, is_super_res = load_config_to_model(
            config_path,
        )
        model.load_state_dict(
            dist_util.load_state_dict(checkpoint_path, map_location="cpu")
        )
        model.to(dist_util.dev())
        model.eval()

        return SamplingModel(
            model=model,
            diffusion_factory=diffusion_factory,
            tokenizer=tokenizer,
            is_super_res=is_super_res,
            timestep_respacing=timestep_respacing,
        )

    def sample(
        self,
        text: Union[str, List[str]],
        batch_size: int,
        n_samples: int,
        clip_denoised=True,
        use_ddim=False,
        low_res=None,
        seed=None,
        to_visible=True,
        from_visible=True,
        clf_free_guidance=False,
        guidance_scale=0.,
        txt_drop_string='<mask><mask><mask><mask>',
        return_intermediates=False,
        use_prk=False,
        use_plms=False,
    ):
        dist_util.setup_dist()

        if self.is_super_res and low_res is None:
            raise ValueError("must pass low_res for super res")

        if return_intermediates and use_ddim:
            raise ValueError('not supported')

        if isinstance(text, str):
            batch_text = batch_size * [text]
        else:
            if len(text) != batch_size:
                raise ValueError(f"got {len(text)} texts for bs {batch_size}")
            batch_text = text

        n_batches = n_samples // batch_size

        if seed is not None:
            print(f"setting seed to {seed}")
            th.manual_seed(seed)

        if return_intermediates:
            def sample_fn_(*args, **kwargs):
                sample_array, xstart_array = [], []
                for out in self.diffusion.p_sample_loop_progressive(*args, **kwargs):
                    sample_array.append(out['sample'])
                    xstart_array.append(out['pred_xstart'])
                return {'sample': sample_array, 'xstart': xstart_array}

            sample_fn = sample_fn_
        elif use_plms:
            sample_fn = self.diffusion.plms_sample_loop
        elif use_prk:
            sample_fn = self.diffusion.prk_sample_loop
        elif use_ddim:
            sample_fn = self.diffusion.ddim_sample_loop
        else:
            sample_fn = self.diffusion.p_sample_loop

        model_kwargs = {}

        txt = tokenize(self.tokenizer, batch_text)
        txt = th.as_tensor(txt).to(dist_util.dev())
        model_kwargs["txt"] = txt

        if clf_free_guidance and (guidance_scale > 0):
            txt_uncon = batch_size * tokenize(self.tokenizer, [txt_drop_string])
            txt_uncon = th.as_tensor(txt_uncon).to(dist_util.dev())

            model_kwargs["guidance_scale"] = guidance_scale
            model_kwargs["unconditional_model_kwargs"] = {
                "txt": txt_uncon
            }

        all_low_res = []

        if self.is_super_res:
            # TODO: shape vs. text shape
            print(f"batch_size: {batch_size} vs low_res shape {low_res.shape}")

            low_res = th.from_numpy(low_res).float()

            if from_visible:
                low_res = low_res / 127.5 - 1.0
                low_res = low_res.permute(0, 3, 1, 2)

            all_low_res = low_res.to(dist_util.dev())
            print(
                f"batch_size: {batch_size} vs low_res kwarg shape {all_low_res.shape}"
            )

        image_channels = self.model.in_channels
        if self.is_super_res:
            image_channels -= all_low_res.shape[1]

        all_images = []
        all_sample_sequences = []
        all_xstart_sequences = []

        def _to_visible(img):
            img = ((img + 1) * 127.5).clamp(0, 255).to(th.uint8)
            img = img.permute(0, 2, 3, 1)
            img = img.contiguous()
            return img

        while len(all_images) * batch_size < n_samples:
            offset = len(all_images)
            if self.is_super_res:
                model_kwargs["low_res"] = all_low_res[offset : offset + batch_size]

                if "unconditional_model_kwargs" in model_kwargs:
                    model_kwargs["unconditional_model_kwargs"]["low_res"] = model_kwargs["low_res"]

            sample = sample_fn(
                self.model,
                (
                    batch_size,
                    image_channels,
                    self.model.image_size,
                    self.model.image_size,
                ),
                clip_denoised=clip_denoised,
                model_kwargs=model_kwargs,
            )

            if return_intermediates:
                sample_sequence = sample['sample']
                xstart_sequence = sample['xstart']
                sample = sample_sequence[-1]

            if to_visible:
                sample = _to_visible(sample)
                if return_intermediates:
                    # todo: vectorize
                    sample_sequence = [_to_visible(x) for x in sample_sequence]
                    xstart_sequence = [_to_visible(x) for x in xstart_sequence]

            all_images.append(sample.cpu().numpy())

            if return_intermediates:
                sample_sequence = th.stack(sample_sequence, dim=1)
                xstart_sequence = th.stack(xstart_sequence, dim=1)
                all_sample_sequences.append(sample_sequence.cpu().numpy())
                all_xstart_sequences.append(xstart_sequence.cpu().numpy())

        all_images = np.concatenate(all_images, axis=0)

        if return_intermediates:
            all_sample_sequences = np.concatenate(all_sample_sequences, axis=0)
            all_xstart_sequences = np.concatenate(all_xstart_sequences, axis=0)

            return all_images, all_sample_sequences, all_xstart_sequences

        return all_images


class SamplingPipeline(nn.Module):
    def __init__(self, base_model: SamplingModel, super_res_model: SamplingModel):
        super().__init__()
        self.base_model = base_model
        self.super_res_model = super_res_model

    def sample(
        self,
        text: Union[str, List[str]],
        batch_size: int,
        n_samples: int,
        clip_denoised=True,
        use_ddim=False,
        clf_free_guidance=False,
        guidance_scale=0.,
        txt_drop_string='<mask><mask><mask><mask>',
        low_res=None,
        seed=None,
        batch_size_sres=None,
        n_samples_sres=None,
        clf_free_guidance_sres=False,
        guidance_scale_sres=0.,
        strip_space=True,
        return_both_resolutions=False
    ):
        if isinstance(text, list):
            text = [_strip_space(s) for s in text]
        else:
            text = _strip_space(text)

        batch_size_sres = batch_size_sres or batch_size
        n_samples_sres = n_samples_sres or n_samples

        low_res = self.base_model.sample(
            text,
            batch_size,
            n_samples,
            clip_denoised=clip_denoised,
            use_ddim=use_ddim,
            clf_free_guidance=clf_free_guidance,
            guidance_scale=guidance_scale,
            txt_drop_string=txt_drop_string,
            seed=seed,
            to_visible=False,
        )
        high_res = self.super_res_model.sample(
            text,
            batch_size_sres,
            n_samples_sres,
            low_res=low_res,
            clip_denoised=clip_denoised,
            use_ddim=use_ddim,
            clf_free_guidance=clf_free_guidance_sres,
            guidance_scale=guidance_scale_sres,
            txt_drop_string=txt_drop_string,
            seed=seed,
            from_visible=False,
        )
        if return_both_resolutions:
            return high_res, low_res
        return high_res

    def sample_with_pruning(
        self,
        text: Union[str, List[str]],
        batch_size: int,
        n_samples: int,
        prune_fn,
        continue_if_all_pruned=True,
        clip_denoised=True,
        use_ddim=False,
        clf_free_guidance=False,
        guidance_scale=0.,
        txt_drop_string='<mask><mask><mask><mask>',
        low_res=None,
        seed=None,
        batch_size_sres=None,
        n_samples_sres=None,
        clf_free_guidance_sres=False,
        guidance_scale_sres=0.,
        strip_space=True,
        return_both_resolutions=False,
        use_plms=False,
        use_plms_sres=False,
    ):
        if strip_space:
            if isinstance(text, list):
                text = [_strip_space(s) for s in text]
            else:
                text = _strip_space(text)

        batch_size_sres = batch_size_sres or batch_size
        n_samples_sres = n_samples_sres or n_samples

        low_res = self.base_model.sample(
            text,
            batch_size,
            n_samples,
            clip_denoised=clip_denoised,
            use_ddim=use_ddim,
            clf_free_guidance=clf_free_guidance,
            guidance_scale=guidance_scale,
            txt_drop_string=txt_drop_string,
            seed=seed,
            use_plms=use_plms,
            to_visible=True,
        )
        low_res_pruned, text_pruned = prune_fn(low_res, text)
        if len(low_res_pruned) == 0:
            if continue_if_all_pruned:
                print(
                    f"all {len(low_res)} low res samples would be pruned, skipping prune"
                )
                low_res_pruned = low_res
                text_pruned = text
            else:
                return low_res_pruned

        # n_samples_sres = minimum we're OK sampling
        n_samples_sres = max(n_samples_sres, len(low_res_pruned))

        # TODO: n_samples > batch_size case
        tile_shape = [1] * low_res_pruned.ndim
        tile_shape[0] = n_samples_sres // len(low_res_pruned) + 1
        low_res_pruned = np.tile(low_res_pruned, tile_shape)[:n_samples_sres]

        # text_pruned = (text_pruned * tile_shape[0])[:n_samples_sres]
        text_pruned = text  # TODO: remove 'text_pruned' concept

        high_res = self.super_res_model.sample(
            text_pruned,
            batch_size_sres,
            n_samples_sres,
            low_res=low_res_pruned,
            clip_denoised=clip_denoised,
            use_ddim=use_ddim,
            clf_free_guidance=clf_free_guidance_sres,
            guidance_scale=guidance_scale_sres,
            txt_drop_string=txt_drop_string,
            seed=seed,
            use_plms=use_plms_sres,
            from_visible=True,
        )
        if return_both_resolutions:
            return high_res, low_res
        return high_res
