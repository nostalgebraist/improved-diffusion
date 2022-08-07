import argparse
import inspect
import json
from functools import partial

import numpy as np

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .unet import SuperResModel, UNetModel
from .image_datasets import load_tokenizer

NUM_CLASSES = 1000


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    return dict(
        image_size=64,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions="16,8",
        dropout=0.0,
        learn_sigma=False,
        sigma_small=False,
        class_cond=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
        resblock_updown=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        channels_per_head=0,
        channels_per_head_upsample=-1,
        channel_mult="",
        use_checkpoint_down=False,
        use_checkpoint_middle=False,
        use_checkpoint_up=False,
        txt=False,
        txt_dim=128,
        txt_depth=2,
        max_seq_len=64,
        txt_resolutions="8",
        cross_attn_channels_per_head=-1,
        cross_attn_init_gain=1.,
        cross_attn_gain_scale=200.,
        text_lr_mult=-1.,
        txt_output_layers_only=False,
        monochrome=False,
        monochrome_adapter=False,
        txt_attn_before_attn=False,
        txt_avoid_groupnorm=False,
        cross_attn_orth_init=False,
        cross_attn_q_t_emb=False,
        txt_rezero=False,
        cross_attn_rezero=False,
        cross_attn_rezero_keeps_prenorm=False,
        cross_attn_use_layerscale=False,
        tokenizer=None,
        verbose=False,
        txt_t5=False,
        txt_rotary=False,
        rgb_adapter=False,
        weave_attn=False,
        weave_use_ff=True,
        weave_ff_rezero=True,
        weave_ff_force_prenorm=False,
        weave_ff_mult=4,
        weave_ff_glu=False,
        weave_qkv_dim_always_text=False,
        channels_last_mem=False,
        txt_ff_glu=False,
        txt_ff_mult=4,
        weave_v2=False,
        use_checkpoint_lowcost=False,
        weave_use_ff_gain=False,
        return_diffusion_factory=False,
        use_balanced_loss=False,
        use_v_loss=False,
        use_snr_plus_one_loss=False,
        bread_adapter_at_ds=-1,
        bread_adapter_nearest_in=False,
        bread_adapter_zero_conv_in=False,
        bread_adapter_only=False,
        expand_timestep_base_dim=-1,
        silu_impl="torch",
        using_capt=False,
        xattn_capt=True,
        weave_capt=False,
        glide_style_capt_attn=False,
        glide_style_capt_emb=False,
        glide_style_capt_emb_init_scale=0.1,
        glide_style_capt_emb_nonlin=False,
        clipname='RN50',
        clip_use_penultimate_layer=False,
        use_checkpoint_below_res=-1,
        vb_loss_ratio=1000.,
        no_attn=False,
        no_attn_substitute_resblock=False,
        freeze_capt_encoder=False,
        clipmod=None,
        post_txt_image_attn='none',
    )


def create_model_and_diffusion(
    image_size,
    class_cond,
    learn_sigma,
    sigma_small,
    num_channels,
    num_res_blocks,
    num_heads,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    channels_per_head,
    channels_per_head_upsample,
    channel_mult="",
    verbose=False,
    use_checkpoint_up=False,
    use_checkpoint_middle=False,
    use_checkpoint_down=False,
    txt=False,
    txt_dim=128,
    max_seq_len=64,
    txt_depth=2,
    txt_resolutions="8",
    cross_attn_channels_per_head=-1,
    cross_attn_init_gain=1.,
    cross_attn_gain_scale=200.,
    text_lr_mult=-1.,
    txt_output_layers_only=False,
    monochrome=False,
    monochrome_adapter=False,
    txt_attn_before_attn=False,
    txt_avoid_groupnorm=False,
    cross_attn_orth_init=False,
    cross_attn_q_t_emb=False,
    txt_rezero=False,
    cross_attn_rezero=False,
    cross_attn_rezero_keeps_prenorm=False,
    cross_attn_use_layerscale=False,
    tokenizer=None,
    txt_t5=False,
    txt_rotary=False,
    rgb_adapter=False,
    weave_attn=False,
    weave_use_ff=True,
    weave_ff_rezero=True,
    weave_ff_force_prenorm=False,
    weave_ff_mult=4,
    weave_ff_glu=False,
    weave_qkv_dim_always_text=False,
    channels_last_mem=False,
    txt_ff_glu=False,
    txt_ff_mult=4,
    weave_v2=False,
    use_checkpoint_lowcost=False,
    weave_use_ff_gain=False,
    return_diffusion_factory=False,
    use_balanced_loss=False,
    use_v_loss=False,
    use_snr_plus_one_loss=False,
    bread_adapter_at_ds=-1,
    bread_adapter_nearest_in=False,
    bread_adapter_zero_conv_in=False,
    bread_adapter_only=False,
    expand_timestep_base_dim=-1,
    silu_impl="torch",
    using_capt=False,
    xattn_capt=True,
    weave_capt=False,
    glide_style_capt_attn=False,
    glide_style_capt_emb=False,
    glide_style_capt_emb_init_scale=0.1,
    glide_style_capt_emb_nonlin=False,
    clipname='RN50',
    clip_use_penultimate_layer=False,
    use_checkpoint_below_res=-1,
    vb_loss_ratio=1000.,
    no_attn=False,
    no_attn_substitute_resblock=False,
    freeze_capt_encoder=False,
    use_inference_caching=False,
    clipmod=None,
    post_txt_image_attn='none',
):
    print(f"create_model_and_diffusion: got txt={txt}")
    print(f"create_model_and_diffusion: use_checkpoint={use_checkpoint}")
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        use_checkpoint_down=use_checkpoint_down,
        use_checkpoint_middle=use_checkpoint_middle,
        use_checkpoint_up=use_checkpoint_up,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        dropout=dropout,
        channels_per_head=channels_per_head,
        channels_per_head_upsample=channels_per_head_upsample,
        channel_mult=channel_mult,
        txt=txt,
        txt_dim=txt_dim,
        max_seq_len=max_seq_len,
        txt_depth=txt_depth,
        txt_resolutions=txt_resolutions,
        cross_attn_channels_per_head=cross_attn_channels_per_head,
        cross_attn_init_gain=cross_attn_init_gain,
        cross_attn_gain_scale=cross_attn_gain_scale,
        text_lr_mult=text_lr_mult,
        txt_output_layers_only=txt_output_layers_only,
        monochrome=monochrome,
        monochrome_adapter=monochrome_adapter,
        txt_attn_before_attn=txt_attn_before_attn,
        txt_avoid_groupnorm=txt_avoid_groupnorm,
        cross_attn_orth_init=cross_attn_orth_init,
        cross_attn_q_t_emb=cross_attn_q_t_emb,
        txt_rezero=txt_rezero,
        cross_attn_rezero=cross_attn_rezero,
        cross_attn_rezero_keeps_prenorm=cross_attn_rezero_keeps_prenorm,
        cross_attn_use_layerscale=cross_attn_use_layerscale,
        tokenizer=tokenizer,
        txt_t5=txt_t5,
        txt_rotary=txt_rotary,
        rgb_adapter=rgb_adapter,
        weave_attn=weave_attn,
        weave_use_ff=weave_use_ff,
        weave_ff_rezero=weave_ff_rezero,
        weave_ff_force_prenorm=weave_ff_force_prenorm,
        weave_ff_mult=weave_ff_mult,
        weave_ff_glu=weave_ff_glu,
        weave_qkv_dim_always_text=weave_qkv_dim_always_text,
        channels_last_mem=channels_last_mem,
        txt_ff_glu=txt_ff_glu,
        txt_ff_mult=txt_ff_mult,
        weave_v2=weave_v2,
        use_checkpoint_lowcost=use_checkpoint_lowcost,
        weave_use_ff_gain=weave_use_ff_gain,
        bread_adapter_at_ds=bread_adapter_at_ds,
        bread_adapter_nearest_in=bread_adapter_nearest_in,
        bread_adapter_zero_conv_in=bread_adapter_zero_conv_in,
        bread_adapter_only=bread_adapter_only,
        expand_timestep_base_dim=expand_timestep_base_dim,
        verbose=verbose,
        silu_impl=silu_impl,
        using_capt=using_capt,
        xattn_capt=xattn_capt,
        weave_capt=weave_capt,
        glide_style_capt_attn=glide_style_capt_attn,
        glide_style_capt_emb=glide_style_capt_emb,
        glide_style_capt_emb_init_scale=glide_style_capt_emb_init_scale,
        glide_style_capt_emb_nonlin=glide_style_capt_emb_nonlin,
        clipname=clipname,
        clip_use_penultimate_layer=clip_use_penultimate_layer,
        use_checkpoint_below_res=use_checkpoint_below_res,
        no_attn=no_attn,
        no_attn_substitute_resblock=no_attn_substitute_resblock,
        freeze_capt_encoder=freeze_capt_encoder,
        use_inference_caching=use_inference_caching,
        clipmod=clipmod,
        post_txt_image_attn=post_txt_image_attn,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
        return_diffusion_factory=return_diffusion_factory,
        use_balanced_loss=use_balanced_loss,
        use_v_loss=use_v_loss,
        use_snr_plus_one_loss=use_snr_plus_one_loss,
        vb_loss_ratio=vb_loss_ratio,
    )
    if verbose:
        print(model)
    n_params = sum([np.product(p.shape) for p in model.parameters()])
    print(f"{n_params/1e6:.0f}M params")
    return model, diffusion


def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    resblock_updown,
    dropout,
    channels_per_head,
    channels_per_head_upsample,
    channel_mult="",
    use_checkpoint_up=False,
    use_checkpoint_middle=False,
    use_checkpoint_down=False,
    txt=False,
    txt_dim=128,
    max_seq_len=64,
    txt_depth=2,
    txt_resolutions="8",
    cross_attn_channels_per_head=-1,
    cross_attn_init_gain=1.,
    cross_attn_gain_scale=200.,
    text_lr_mult=-1.,
    txt_output_layers_only=False,
    monochrome=False,
    monochrome_adapter=False,
    txt_attn_before_attn=False,
    txt_avoid_groupnorm=False,
    cross_attn_orth_init=False,
    cross_attn_q_t_emb=False,
    txt_rezero=False,
    cross_attn_rezero=False,
    cross_attn_rezero_keeps_prenorm=False,
    cross_attn_use_layerscale=False,
    small_size=None,
    model_cls=UNetModel,
    tokenizer=None,
    txt_t5=False,
    txt_rotary=False,
    rgb_adapter=False,
    colorize=False,
    weave_attn=False,
    weave_use_ff=True,
    weave_ff_rezero=True,
    weave_ff_force_prenorm=False,
    weave_ff_mult=4,
    weave_ff_glu=False,
    weave_qkv_dim_always_text=False,
    channels_last_mem=False,
    txt_ff_glu=False,
    txt_ff_mult=4,
    up_interp_mode="bilinear",
    weave_v2=False,
    use_checkpoint_lowcost=False,
    weave_use_ff_gain=False,
    bread_adapter_at_ds=-1,
    bread_adapter_nearest_in=False,
    bread_adapter_zero_conv_in=False,
    bread_adapter_only=False,
    expand_timestep_base_dim=-1,
    verbose=False,
    silu_impl="torch",
    using_capt=False,
    xattn_capt=True,
    weave_capt=False,
    glide_style_capt_attn=False,
    glide_style_capt_emb=False,
    glide_style_capt_emb_init_scale=0.1,
    glide_style_capt_emb_nonlin=False,
    clipname='RN50',
    clip_use_penultimate_layer=False,
    use_checkpoint_below_res=-1,
    no_attn=False,
    no_attn_substitute_resblock=False,
    noise_cond=False,
    freeze_capt_encoder=False,
    use_inference_caching=False,
    clipmod=None,
    post_txt_image_attn='none',
):
    text_lr_mult = 1.
    print(
        f"create_model: got txt={txt}, num_heads={num_heads}, channels_per_head={channels_per_head}, cross_attn_channels_per_head={cross_attn_channels_per_head}, text_lr_mult={text_lr_mult}"
    )
    print(f"create_model: noise_cond={noise_cond}, use_checkpoint={use_checkpoint}, use_checkpoint_lowcost={use_checkpoint_lowcost}")
    if channel_mult != "":
        print(f"got channel_mult: {channel_mult}")
        try:
            channel_mult = tuple(float(v) for v in channel_mult.strip().split(','))
        except ValueError:
            pass

    if not isinstance(channel_mult, tuple):
        if image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        elif image_size == 32:
            channel_mult = (1, 2, 2, 2)
        else:
            raise ValueError(f"unsupported image size: {image_size}")

    print(f"channel_mult: {channel_mult}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    txt_ds = []
    for res in txt_resolutions.split(","):
        txt_ds.append(image_size // int(res))

    if monochrome and (not monochrome_adapter):
        in_channels = 1
        out_channels = (1 if not learn_sigma else 2)
    else:
        in_channels = 3
        out_channels = (3 if not learn_sigma else 6)

    return model_cls(
        in_channels=in_channels,
        model_channels=num_channels,
        out_channels=out_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_checkpoint_down=use_checkpoint_down,
        use_checkpoint_middle=use_checkpoint_middle,
        use_checkpoint_up=use_checkpoint_up,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        channels_per_head=channels_per_head,
        channels_per_head_upsample=channels_per_head_upsample,
        txt=txt,
        txt_dim=txt_dim,
        max_seq_len=max_seq_len,
        txt_depth=txt_depth,
        txt_resolutions=txt_ds,
        cross_attn_channels_per_head=cross_attn_channels_per_head,
        cross_attn_init_gain=cross_attn_init_gain,
        cross_attn_gain_scale=cross_attn_gain_scale,
        image_size=image_size,
        text_lr_mult=text_lr_mult,
        txt_output_layers_only=txt_output_layers_only,
        monochrome_adapter=monochrome_adapter,
        txt_attn_before_attn=txt_attn_before_attn,
        txt_avoid_groupnorm=txt_avoid_groupnorm,
        cross_attn_orth_init=cross_attn_orth_init,
        cross_attn_q_t_emb=cross_attn_q_t_emb,
        txt_rezero=txt_rezero,
        cross_attn_rezero=cross_attn_rezero,
        cross_attn_rezero_keeps_prenorm=cross_attn_rezero_keeps_prenorm,
        cross_attn_use_layerscale=cross_attn_use_layerscale,
        tokenizer=tokenizer,
        txt_t5=txt_t5,
        txt_rotary=txt_rotary,
        rgb_adapter=rgb_adapter,
        colorize=colorize,
        weave_attn=weave_attn,
        weave_use_ff=weave_use_ff,
        weave_ff_rezero=weave_ff_rezero,
        weave_ff_force_prenorm=weave_ff_force_prenorm,
        weave_ff_mult=weave_ff_mult,
        weave_ff_glu=weave_ff_glu,
        weave_qkv_dim_always_text=weave_qkv_dim_always_text,
        channels_last_mem=channels_last_mem,
        txt_ff_glu=txt_ff_glu,
        txt_ff_mult=txt_ff_mult,
        up_interp_mode=up_interp_mode,
        weave_v2=weave_v2,
        use_checkpoint_lowcost=use_checkpoint_lowcost,
        weave_use_ff_gain=weave_use_ff_gain,
        bread_adapter_at_ds=bread_adapter_at_ds,
        bread_adapter_nearest_in=bread_adapter_nearest_in,
        bread_adapter_zero_conv_in=bread_adapter_zero_conv_in,
        bread_adapter_only=bread_adapter_only,
        expand_timestep_base_dim=expand_timestep_base_dim,
        verbose=verbose,
        silu_impl=silu_impl,
        using_capt=using_capt,
        xattn_capt=xattn_capt,
        weave_capt=weave_capt,
        glide_style_capt_attn=glide_style_capt_attn,
        glide_style_capt_emb=glide_style_capt_emb,
        glide_style_capt_emb_init_scale=glide_style_capt_emb_init_scale,
        glide_style_capt_emb_nonlin=glide_style_capt_emb_nonlin,
        clipname=clipname,
        clip_use_penultimate_layer=clip_use_penultimate_layer,
        use_checkpoint_below_res=use_checkpoint_below_res,
        no_attn=no_attn,
        no_attn_substitute_resblock=no_attn_substitute_resblock,
        noise_cond=noise_cond,
        freeze_capt_encoder=freeze_capt_encoder,
        use_inference_caching=use_inference_caching,
        clipmod=clipmod,
        post_txt_image_attn=post_txt_image_attn,
    )


def sr_model_and_diffusion_defaults():
    res = model_and_diffusion_defaults()
    res["large_size"] = 256
    res["small_size"] = 64
    res["colorize"] = False
    res["up_interp_mode"] = "bilinear"
    res["noise_cond"] = False
    arg_names = inspect.getfullargspec(sr_create_model_and_diffusion)[0]
    for k in res.copy().keys():
        if k not in arg_names:
            del res[k]
    return res


def sr_create_model_and_diffusion(
    large_size,
    small_size,
    class_cond,
    learn_sigma,
    num_channels,
    num_res_blocks,
    num_heads,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    channels_per_head,
    channels_per_head_upsample,
    channel_mult="",
    verbose=False,
    use_checkpoint_up=False,
    use_checkpoint_middle=False,
    use_checkpoint_down=False,
    txt=False,
    txt_dim=128,
    max_seq_len=64,
    txt_depth=2,
    txt_resolutions="8",
    cross_attn_channels_per_head=-1,
    cross_attn_init_gain=1.,
    cross_attn_gain_scale=200.,
    text_lr_mult=-1.,
    txt_output_layers_only=False,
    monochrome=False,
    monochrome_adapter=False,
    txt_attn_before_attn=False,
    txt_avoid_groupnorm=False,
    cross_attn_orth_init=False,
    cross_attn_q_t_emb=False,
    txt_rezero=False,
    cross_attn_rezero=False,
    cross_attn_rezero_keeps_prenorm=False,
    cross_attn_use_layerscale=False,
    tokenizer=None,
    txt_t5=False,
    txt_rotary=False,
    colorize=False,
    rgb_adapter=False,
    weave_attn=False,
    weave_use_ff=True,
    weave_ff_rezero=True,
    weave_ff_force_prenorm=False,
    weave_ff_mult=4,
    weave_ff_glu=False,
    weave_qkv_dim_always_text=False,
    channels_last_mem=False,
    txt_ff_glu=False,
    txt_ff_mult=4,
    up_interp_mode='bilinear',
    weave_v2=False,
    use_checkpoint_lowcost=False,
    weave_use_ff_gain=False,
    return_diffusion_factory=False,
    use_balanced_loss=False,
    use_v_loss=False,
    use_snr_plus_one_loss=False,
    silu_impl="torch",
    using_capt=False,
    xattn_capt=True,
    weave_capt=False,
    glide_style_capt_attn=False,
    glide_style_capt_emb=False,
    glide_style_capt_emb_init_scale=0.1,
    glide_style_capt_emb_nonlin=False,
    clipname='RN50',
    clip_use_penultimate_layer=False,
    expand_timestep_base_dim=-1,
    use_checkpoint_below_res=-1,
    vb_loss_ratio=1000.,
    no_attn=False,
    no_attn_substitute_resblock=False,
    noise_cond=False,
    freeze_capt_encoder=False,
    use_inference_caching=False,
    clipmod=None,
    post_txt_image_attn='none',
):
    model = sr_create_model(
        large_size,
        small_size,
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        dropout=dropout,
        channels_per_head=channels_per_head,
        channels_per_head_upsample=channels_per_head_upsample,
        channel_mult=channel_mult,
        txt=txt,
        txt_dim=txt_dim,
        max_seq_len=max_seq_len,
        txt_depth=txt_depth,
        txt_resolutions=txt_resolutions,
        cross_attn_channels_per_head=cross_attn_channels_per_head,
        cross_attn_init_gain=cross_attn_init_gain,
        cross_attn_gain_scale=cross_attn_gain_scale,
        text_lr_mult=text_lr_mult,
        txt_output_layers_only=txt_output_layers_only,
        monochrome=monochrome,
        monochrome_adapter=monochrome_adapter,
        txt_attn_before_attn=txt_attn_before_attn,
        txt_avoid_groupnorm=txt_avoid_groupnorm,
        cross_attn_orth_init=cross_attn_orth_init,
        cross_attn_q_t_emb=cross_attn_q_t_emb,
        txt_rezero=txt_rezero,
        cross_attn_rezero=cross_attn_rezero,
        cross_attn_rezero_keeps_prenorm=cross_attn_rezero_keeps_prenorm,
        cross_attn_use_layerscale=cross_attn_use_layerscale,
        tokenizer=tokenizer,
        txt_t5=txt_t5,
        txt_rotary=txt_rotary,
        colorize=colorize,
        rgb_adapter=rgb_adapter,
        weave_attn=weave_attn,
        weave_use_ff=weave_use_ff,
        weave_ff_rezero=weave_ff_rezero,
        weave_ff_force_prenorm=weave_ff_force_prenorm,
        weave_ff_mult=weave_ff_mult,
        weave_ff_glu=weave_ff_glu,
        weave_qkv_dim_always_text=weave_qkv_dim_always_text,
        channels_last_mem=channels_last_mem,
        txt_ff_glu=txt_ff_glu,
        txt_ff_mult=txt_ff_mult,
        up_interp_mode=up_interp_mode,
        weave_v2=weave_v2,
        use_checkpoint_lowcost=use_checkpoint_lowcost,
        weave_use_ff_gain=weave_use_ff_gain,
        silu_impl=silu_impl,
        using_capt=using_capt,
        xattn_capt=xattn_capt,
        weave_capt=weave_capt,
        glide_style_capt_attn=glide_style_capt_attn,
        glide_style_capt_emb=glide_style_capt_emb,
        glide_style_capt_emb_init_scale=glide_style_capt_emb_init_scale,
        glide_style_capt_emb_nonlin=glide_style_capt_emb_nonlin,
        clipname=clipname,
        clip_use_penultimate_layer=clip_use_penultimate_layer,
        expand_timestep_base_dim=expand_timestep_base_dim,
        use_checkpoint_below_res=use_checkpoint_below_res,
        verbose=verbose,
        no_attn=no_attn,
        no_attn_substitute_resblock=no_attn_substitute_resblock,
        noise_cond=noise_cond,
        freeze_capt_encoder=freeze_capt_encoder,
        use_inference_caching=use_inference_caching,
        clipmod=clipmod,
        post_txt_image_attn=post_txt_image_attn,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
        return_diffusion_factory=return_diffusion_factory,
        use_balanced_loss=use_balanced_loss,
        use_v_loss=use_v_loss,
        use_snr_plus_one_loss=use_snr_plus_one_loss,
        vb_loss_ratio=vb_loss_ratio,
    )
    if verbose:
        print(model)
    n_params = sum([np.product(p.shape) for p in model.parameters()])
    print(f"{n_params/1e6:.0f}M params")
    return model, diffusion


def _sr_create_model(
    large_size,
    small_size,
    num_channels,
    num_res_blocks,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
):
    _ = small_size  # hack to prevent unused variable

    if large_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif large_size == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported large size: {large_size}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(large_size // int(res))

    return SuperResModel(
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
    )


def sr_create_model(large_size, small_size, **kwargs):
    return create_model(image_size=large_size, model_cls=SuperResModel, **kwargs)


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
    return_diffusion_factory=False,
    use_balanced_loss=False,
    use_v_loss=False,
    use_snr_plus_one_loss=False,
    vb_loss_ratio=1000.,
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_snr_plus_one_loss:
        loss_type = gd.LossType.RESCALED_MSE_SNR_PLUS_ONE
    elif use_v_loss:
        loss_type = gd.LossType.RESCALED_MSE_V
    elif use_balanced_loss:
        loss_type = gd.LossType.RESCALED_MSE_BALANCED
    elif use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    def diffusion_factory(timestep_respacing_=timestep_respacing):
        cls, kwargs = gd.GaussianDiffusion, {}
        if timestep_respacing_ != [steps]:
            cls = SpacedDiffusion
            kwargs['use_timesteps'] = space_timesteps(steps, timestep_respacing_)
        return cls(
            betas=betas,
            model_mean_type=(
                gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
            ),
            model_var_type=(
                (
                    gd.ModelVarType.FIXED_LARGE
                    if not sigma_small
                    else gd.ModelVarType.FIXED_SMALL
                )
                if not learn_sigma
                else gd.ModelVarType.LEARNED_RANGE
            ),
            loss_type=loss_type,
            rescale_timesteps=rescale_timesteps,
            vb_loss_ratio=vb_loss_ratio,
            **kwargs
        )
    if return_diffusion_factory:
        return diffusion_factory
    return diffusion_factory()


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def load_config_to_args(config_path, args, request_approval=False):
    is_super_res = None

    updates = {}

    with open(config_path, 'r') as f:
        conf = json.load(f)
    for k in conf:
        if k == 'is_super_res':
            is_super_res = conf[k]
        elif k == 'tokenizer_config':
            for k2 in conf[k]:
                updates[k2] = conf[k][k2]
        else:
            updates[k] = conf[k]

    changes = []
    for k in updates:
        cur = getattr(args, k, None)
        new = updates[k]
        if cur != new:
            changes.append((k, cur, new))

    use_config = True

    if request_approval and len(changes) > 0:
        print("Using config file would change these settings:")
        for (k, cur, new) in changes:
            print(f"\t{k}:\t\t{cur}\t\t--> {new}")
        response = input("Really use config?\n")

        use_config = (response.lower() == 'y')
        print(f"Using config?: {use_config}")

    if use_config:
        for k in updates:
            setattr(args, k, conf[k])

    return args, is_super_res


def load_config_to_model(config_path, overrides=None):
    if overrides is None:
        overrides = {}

    with open(config_path, 'r') as f:
        conf = json.load(f)

    conf['return_diffusion_factory'] = True

    is_super_res = conf['is_super_res']

    if is_super_res:
        defaults = sr_model_and_diffusion_defaults()
    else:
        defaults = model_and_diffusion_defaults()

    model_diffusion_args = {k: conf.get(k, defaults[k]) for k in defaults}

    tokenizer = None

    if model_diffusion_args['txt']:
        tokenizer_config = conf['tokenizer_config']
        tokenizer = load_tokenizer(**tokenizer_config)
        model_diffusion_args['tokenizer'] = tokenizer

    creator = sr_create_model_and_diffusion if is_super_res else create_model_and_diffusion

    model_diffusion_args.update(overrides)

    model, diffusion_factory = creator(**model_diffusion_args)

    return model, diffusion_factory, tokenizer, is_super_res


def save_config(config_path, model_diffusion_args, tokenizer_config, is_super_res):
    conf = dict(is_super_res=is_super_res, tokenizer_config=tokenizer_config)
    for k in model_diffusion_args:
        if k == "tokenizer":
            continue
        conf[k] = model_diffusion_args[k]
    with open(config_path, 'w') as f:
        json.dump(conf, f, indent=1)
