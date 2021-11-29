import argparse
import inspect

import numpy as np

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .unet import SuperResModel, UNetModel

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
        verbose=False,
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
):
    print(f"create_model_and_diffusion: got txt={txt}")
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
):
    text_lr_mult = 1.
    print(
        f"create_model: got txt={txt}, num_heads={num_heads}, channels_per_head={channels_per_head}, cross_attn_channels_per_head={cross_attn_channels_per_head}, text_lr_mult={text_lr_mult}"
    )
    if channel_mult != "":
        print(f"got channel_mult: {channel_mult}")
        try:
            channel_mult = tuple(int(v) for v in channel_mult.strip().split(','))
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

    return UNetModel(
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
        cross_attn_use_layerscale=cross_attn_use_layerscale
    )


def sr_model_and_diffusion_defaults():
    res = model_and_diffusion_defaults()
    res["large_size"] = 256
    res["small_size"] = 64
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
):
    model = sr_create_model(
        large_size,
        small_size,
        num_channels,
        num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
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
    )
    return model, diffusion


def sr_create_model(
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
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
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
    )


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
