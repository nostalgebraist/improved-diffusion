"""
Train a diffusion model on images.
"""

import argparse, os, json
import torch as th

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data, load_tokenizer, save_first_batch
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    load_config_to_args,
    save_config
)
from improved_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()
    print(f"args: got txt={args.txt}")

    dist_util.setup_dist()
    logger.configure()

    # if args.channels_last_mem:
    #     import improved_diffusion.channels_last_checker

    print(f"args.text_lr: {type(args.text_lr)}, {args.text_lr}")

    if isinstance(args.text_lr, float) and args.text_lr > 0:
        args.text_lr_mult = args.text_lr / args.lr
    else:
        args.text_lr_mult = None
    print(f"args.text_lr_mult: {args.text_lr_mult}")

    if args.text_lr < 0:
        args.text_lr = None

    if args.gain_lr < 0:
        args.gain_lr = None

    config_path = args.config_path
    have_config_path = config_path != ""
    using_config = have_config_path and os.path.exists(config_path)

    if using_config:
        args, _ = load_config_to_args(config_path, args)

    tokenizer = None
    tokenizer_config = dict(
        max_seq_len=getattr(args, 'max_seq_len', None),
        char_level=getattr(args, 'char_level', None),
    )
    if args.txt:
        tokenizer = load_tokenizer(**tokenizer_config)

    logger.log("creating model and diffusion...")
    print(f"image_train: use_checkpoint={args.use_checkpoint}")
    model_diffusion_args = args_to_dict(args, model_and_diffusion_defaults().keys())
    model_diffusion_args['tokenizer'] = tokenizer
    model, diffusion = create_model_and_diffusion(
        **model_diffusion_args
        # verbose=False
    )

    if have_config_path and (not using_config):
        save_config(config_path, model_diffusion_args, tokenizer_config, is_super_res=False)

    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    if args.text_encoder_warmstart != "" and os.path.exists(args.text_encoder_warmstart):
        sd = th.load(args.text_encoder_warmstart)
        sd = {k.partition("text_encoder.")[2]: v for k, v in sd.items() if k.startswith("text_encoder.")}
        ks = list(sd.keys())
        exk = ks[0]
        print(('state_dict', exk, sd[exk]))
        for n, p in model.text_encoder.named_parameters():
            if n == exk:
                print(('model (before)', n, p))
        model.text_encoder.load_state_dict(sd)
        for n, p in model.text_encoder.named_parameters():
            if n == exk:
                print(('model (after)', n, p))

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        txt=args.txt,
        monochrome=args.monochrome,
        min_filesize=args.min_filesize,
        txt_pdrop=args.txt_pdrop,
        crop_prob=args.crop_prob,
        crop_min_scale=args.crop_min_scale,
        crop_max_scale=args.crop_max_scale,
        use_special_crop_for_empty_string=args.use_special_crop_for_empty_string,
        crop_prob_es=args.crop_prob_es,
        crop_min_scale_es=args.crop_min_scale_es,
        crop_max_scale_es=args.crop_max_scale_es,
        safebox_path=args.safebox_path,
        use_random_safebox_for_empty_string=args.use_random_safebox_for_empty_string,
        flip_lr_prob_es=args.flip_lr_prob_es,
        px_scales_path=args.px_scales_path
    )

    if args.save_first_batch:
        save_first_batch(data, 'first_batch/')

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        text_lr=args.text_lr,
        gain_lr=args.gain_lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        tokenizer=tokenizer,
        lg_loss_scale=args.lg_loss_scale,
        beta1=args.beta1,
        beta2=args.beta2,
        weave_legacy_param_names=args.weave_legacy_param_names,
        state_dict_sandwich=args.state_dict_sandwich,
        state_dict_sandwich_manual_remaps=args.state_dict_sandwich_manual_remaps,
        master_on_cpu=args.master_on_cpu,
        use_amp=args.use_amp,
        use_profiler=args.use_profiler,
        autosave=args.autosave,
        arithmetic_avg_from_step=args.arithmetic_avg_from_step,
        arithmetic_avg_extra_shift=args.arithmetic_avg_extra_shift,
        gain_ff_setup_step=args.gain_ff_setup_step,
        use_bf16=args.use_bf16,
        onestep=args.onestep,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        lg_loss_scale=20,
        fp16_scale_growth=1e-3,
        channel_mult="",
        txt=False,
        txt_dim=128,
        txt_depth=2,
        max_seq_len=64,
        txt_resolutions="8",
        text_lr=-1.,
        gain_lr=-1.,
        beta1=0.9,
        beta2=0.999,
        verbose=False,
        char_level=False,
        text_encoder_warmstart="",
        weave_legacy_param_names=False,
        config_path="",
        state_dict_sandwich=0,
        state_dict_sandwich_manual_remaps="",
        min_filesize=0,
        txt_pdrop=0.,
        master_on_cpu=False,
        use_amp=False,
        use_bf16=False,
        use_profiler=False,
        onestep=False,
        autosave=True,
        arithmetic_avg_from_step='-1',
        arithmetic_avg_extra_shift='0',
        gain_ff_setup_step=False,
        crop_prob=0.,
        crop_min_scale=0.75,
        crop_max_scale=1.,
        use_special_crop_for_empty_string=False,
        crop_prob_es=0.,
        crop_min_scale_es=0.25,
        crop_max_scale_es=1.,
        safebox_path="",
        use_random_safebox_for_empty_string=False,
        flip_lr_prob_es=0.,
        px_scales_path="",
        save_first_batch=False
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
