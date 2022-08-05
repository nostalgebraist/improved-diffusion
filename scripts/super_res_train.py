"""
Train a super-resolution model.
"""

import argparse, os

import torch as th
import torch.nn.functional as F

from improved_diffusion import dist_util, logger, cuda_streams
from improved_diffusion.image_datasets import load_superres_data, load_tokenizer, save_first_batch
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

    th.backends.cudnn.benchmark = args.cudnn_benchmark
    print(f"using cudnn_benchmark: {th.backends.cudnn.benchmark}")

    if args.text_lr < 0:
        args.text_lr = None

    if args.gain_lr < 0:
        args.gain_lr = None

    if args.capt_lr < 0:
        args.capt_lr = None

    dist_util.setup_dist()
    logger.configure()

    if args.use_streams:
        cuda_streams.turn_streaming_on()

    logger.log("creating model...")

    config_path = args.config_path
    have_config_path = config_path != ""
    using_config = have_config_path and os.path.exists(config_path)

    if using_config:
        args, _ = load_config_to_args(config_path, args, request_approval=True)

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
        txt=args.txt or args.use_txt_in_dataload,
        monochrome=args.monochrome,
        colorize=args.colorize,
        blur_prob=args.blur_prob,
        blur_sigma_min=args.blur_sigma_min,
        blur_sigma_max=args.blur_sigma_max,
        min_filesize=args.min_filesize,
        txt_pdrop=args.txt_pdrop,
        crop_prob=args.crop_prob,
        crop_min_scale=args.crop_min_scale,
        crop_max_scale=args.crop_max_scale,
        use_special_crop_for_empty_string=args.use_special_crop_for_empty_string,
        crop_prob_es=args.crop_prob_es,
        crop_min_scale_es=args.crop_min_scale_es,
        crop_max_scale_es=args.crop_max_scale_es,
        crop_without_resize=args.crop_without_resize,
        safebox_path=args.safebox_path,
        use_random_safebox_for_empty_string=args.use_random_safebox_for_empty_string,
        flip_lr_prob_es=args.flip_lr_prob_es,
        px_scales_path=args.px_scales_path,
        pin_memory=args.perf_pin_memory,
        prefetch_factor=args.perf_prefetch_factor,
        num_workers=args.perf_num_workers,
        min_imagesize=args.min_imagesize,
        blur_width=args.blur_width,
        clip_prob_path=args.clip_prob_path,
        clip_prob_middle_pkeep=args.clip_prob_middle_pkeep,
        antialias=args.antialias,
        bicubic_down=args.bicubic_down,
        capt_path=args.capt_path,
        capt_pdrop=args.capt_pdrop,
        require_capts=args.require_capts,
        all_pdrop=args.all_pdrop,
        class_map_path=args.class_map_path,
        class_ix_unk=args.class_ix_unk,
        class_ix_drop=args.class_ix_drop,
        class_pdrop=args.class_pdrop,
        exclusions_data_path=args.exclusions_data_path,
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
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_warmup_shift=args.lr_warmup_shift,
        tokenizer=tokenizer,
        lg_loss_scale=args.lg_loss_scale,
        beta1=args.beta1,
        beta2=args.beta2,
        weave_legacy_param_names=args.weave_legacy_param_names,
        state_dict_sandwich=args.state_dict_sandwich,
        state_dict_sandwich_manual_remaps=args.state_dict_sandwich_manual_remaps,
        use_amp=args.use_amp,
        text_lr=args.text_lr,
        gain_lr=args.gain_lr,
        autosave=args.autosave,
        arithmetic_avg_from_step=args.arithmetic_avg_from_step,
        arithmetic_avg_extra_shift=args.arithmetic_avg_extra_shift,
        gain_ff_setup_step=args.gain_ff_setup_step,
        perf_no_ddl=args.perf_no_ddl,
        param_sandwich=args.param_sandwich,
        resize_mult=args.resize_mult,
        noise_cond=args.noise_cond,
        noise_cond_schedule=args.noise_cond_schedule,
        noise_cond_steps=args.noise_cond_steps,
        noise_cond_max_step=args.noise_cond_max_step,
        capt_lr=args.capt_lr,
        freeze_capt_encoder=args.freeze_capt_encoder,
        use_profiler=args.use_profiler,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        lr_warmup_steps=0,
        lr_warmup_shift=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        lg_loss_scale=20,
        char_level=False,
        text_lr=-1.,
        gain_lr=-1.,
        capt_lr=-1.,
        beta1=0.9,
        beta2=0.999,
        colorize=False,
        text_encoder_warmstart="",
        weave_legacy_param_names=False,
        config_path="",
        blur_prob=0.,
        blur_sigma_min=0.4,
        blur_sigma_max=0.6,
        blur_width=5,
        up_interp_mode='bilinear',
        verbose=False,
        state_dict_sandwich=0,
        state_dict_sandwich_manual_remaps="",
        min_filesize=0,
        txt_pdrop=0.,
        use_amp=False,
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
        crop_without_resize=False,
        safebox_path="",
        use_random_safebox_for_empty_string=False,
        flip_lr_prob_es=0.,
        px_scales_path="",
        perf_no_ddl=False,
        perf_pin_memory=False,
        perf_prefetch_factor=2,
        perf_num_workers=1,
        param_sandwich=0,
        min_imagesize=0,
        save_first_batch=False,
        capt_path="",
        capt_pdrop=0.1,
        all_pdrop=0.1,
        require_capts=False,
        class_map_path="",
        freeze_capt_encoder=False,
        class_ix_unk=0,
        class_ix_drop=999,
        class_pdrop=0.1,
        clip_prob_path="",
        clip_prob_middle_pkeep=0.5,
        resize_mult=1.,
        antialias=False,
        bicubic_down=False,
        use_txt_in_dataload=False,
        noise_cond=False,
        noise_cond_schedule='cosine',
        noise_cond_steps=1000,
        noise_cond_max_step=-1,
        cudnn_benchmark=False,
        exclusions_data_path="",
        use_streams=True,
        use_profiler=False,
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
h
