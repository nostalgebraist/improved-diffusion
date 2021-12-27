"""
Train a super-resolution model.
"""

import argparse, os

import torch as th
import torch.nn.functional as F

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_superres_data, load_tokenizer
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    load_config_to_args,
    save_config
)
from improved_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model...")

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

    model_diffusion_args = args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    model_diffusion_args['tokenizer'] = tokenizer
    model, diffusion = sr_create_model_and_diffusion(
        **model_diffusion_args
    )

    if have_config_path and (not using_config):
        save_config(config_path, model_diffusion_args, tokenizer_config, is_super_res=True)

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
    data = load_superres_data(
        args.data_dir,
        args.batch_size,
        large_size=args.large_size,
        small_size=args.small_size,
        class_cond=args.class_cond,
        txt=args.txt,
        monochrome=args.monochrome,
        colorize=args.colorize,
        blur_prob=args.blur_prob,
        blur_sigma_min=args.blur_sigma_min,
        blur_sigma_max=args.blur_sigma_max,
        min_filesize=args.min_filesize,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
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
        beta1=args.beta1,
        beta2=args.beta2,
        weave_legacy_param_names=args.weave_legacy_param_names,
        state_dict_sandwich=args.state_dict_sandwich,
        state_dict_sandwich_manual_remaps=args.state_dict_sandwich_manual_remaps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        char_level=False,
        beta1=0.9,
        beta2=0.999,
        colorize=False,
        text_encoder_warmstart="",
        weave_legacy_param_names=False,
        config_path="",
        blur_prob=0.,
        blur_sigma_min=0.4,
        blur_sigma_max=0.6,
        up_interp_mode='bilinear',
        verbose=False,
        state_dict_sandwich=0,
        state_dict_sandwich_manual_remaps="",
        min_filesize=0,
        use_checkpoint_down=False,
        use_checkpoint_middle=False,
        use_checkpoint_up=False,
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
