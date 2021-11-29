"""
Train a diffusion model on images.
"""

import argparse

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data, load_tokenizer
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()
    print(f"args: got txt={args.txt}")

    dist_util.setup_dist()
    logger.configure()

    print(f"args.text_lr: {type(args.text_lr)}, {args.text_lr}")

    if isinstance(args.text_lr, float) and args.text_lr > 0:
        args.text_lr_mult = args.text_lr / args.lr
    else:
        args.text_lr_mult = None
    print(f"args.text_lr_mult: {args.text_lr_mult}")

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
        # verbose=False
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        txt=args.txt,
        monochrome=args.monochrome,
    )

    tokenizer = None
    if args.txt:
        tokenizer = load_tokenizer(max_seq_len=args.max_seq_len)

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        text_lr=args.text_lr,
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
        fp16_scale_growth=1e-3,
        channel_mult="",
        use_checkpoint_down=False,
        use_checkpoint_middle=False,
        use_checkpoint_up=False,
        txt=False,
        txt_dim=128,
        txt_depth=2,
        max_seq_len=64,
        txt_resolutions="8",
        text_lr=-1.,
        verbose=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
