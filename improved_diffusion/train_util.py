import copy
import functools
import os
import time
import subprocess
from collections import defaultdict

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from transformer_utils.util.module_utils import get_child_module_by_names

from . import dist_util, logger
from .fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from .nn import update_ema, update_arithmetic_average, scale_module
from .resample import LossAwareSampler, UniformSampler, EarlyOnlySampler
from .gaussian_diffusion import SimpleForwardDiffusion, get_named_beta_schedule

from .image_datasets import tokenize

import clip

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        lr_warmup_steps=0,
        lr_warmup_shift=0,
        tokenizer=None,
        text_lr=None,
        gain_lr=None,
        bread_lr=None,
        capt_lr=None,
        lg_loss_scale = INITIAL_LOG_LOSS_SCALE,
        beta1=0.9,
        beta2=0.999,
        weave_legacy_param_names=False,
        state_dict_sandwich=0,
        state_dict_sandwich_manual_remaps="",
        master_on_cpu=False,
        use_amp=False,
        use_bf16=False,
        use_profiler=False,
        autosave=True,
        autosave_dir="gs://nost_ar_work/improved-diffusion/",
        arithmetic_avg_from_step=-1,
        arithmetic_avg_extra_shift=0,
        gain_ff_setup_step=False,
        only_optimize_bread=False,
        param_sandwich=-1,
        resize_mult=1.,
        perf_no_ddl=False,
        freeze_capt_encoder=False,
        noise_cond=False,
        noise_cond_schedule='cosine',
        noise_cond_steps=1000,
        noise_cond_max_step=-1,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.text_lr = text_lr if text_lr is not None else lr
        self.gain_lr = gain_lr if gain_lr is not None else lr
        self.bread_lr = bread_lr if bread_lr is not None else lr
        self.capt_lr = capt_lr if capt_lr is not None else lr
        print(f"train loop: text_lr={text_lr}, gain_lr={gain_lr}, bread_lr={bread_lr}")
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",") if len(x) > 0]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_warmup_shift = lr_warmup_shift
        self.tokenizer = tokenizer
        self.state_dict_sandwich = state_dict_sandwich
        self.state_dict_sandwich_manual_remaps = {kv.split(":")[0]: kv.split(":")[1]
                                                  for kv in state_dict_sandwich_manual_remaps.split(",")
                                                  if len(kv) > 0
                                                  }
        if param_sandwich < 0:
            param_sandwich = state_dict_sandwich
        self.master_device = 'cpu' if master_on_cpu else None
        self.use_amp = use_amp
        self.use_bf16 = use_bf16
        self.use_profiler = use_profiler
        self.autosave = autosave
        self.autosave_dir = autosave_dir
        self.anneal_log_flag = False
        self.arithmetic_avg_from_step = (
            [arithmetic_avg_from_step for _ in self.ema_rate]
            if isinstance(arithmetic_avg_from_step, float)
            else [float(x) for x in arithmetic_avg_from_step.split(",") if len(x) > 0]
        )
        self.arithmetic_avg_extra_shift = (
            [arithmetic_avg_extra_shift for _ in self.ema_rate]
            if isinstance(arithmetic_avg_extra_shift, float)
            else [float(x) for x in arithmetic_avg_extra_shift.split(",") if len(x) > 0]
        )
        self.only_optimize_bread = only_optimize_bread
        if self.only_optimize_bread:
            raise ValueError('only_optimize_bread no longer supported')
        self.resize_mult = resize_mult
        self.freeze_capt_encoder = freeze_capt_encoder

        self.noise_cond = noise_cond
        self.noise_cond_diffusion = None
        if self.noise_cond:
            betas = get_named_beta_schedule(noise_cond_schedule, noise_cond_steps)
            self.noise_cond_diffusion = SimpleForwardDiffusion(betas)
            # todo: other schedules
            if noise_cond_max_step < 0:
                noise_cond_max_step = noise_cond_steps
            self.noise_cond_schedule_sampler = EarlyOnlySampler(self.noise_cond_diffusion, noise_cond_max_step)

        print(f"TrainLoop self.master_device: {self.master_device}, use_amp={use_amp}, autosave={self.autosave}")
        print(f"TrainLoop self.arithmetic_avg_from_step: {self.arithmetic_avg_from_step}, self.arithmetic_avg_extra_shift={self.arithmetic_avg_extra_shift}")

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size # * dist.get_world_size()

        # text_params, self.text_param_names = [], []
        text_params, text_param_names = defaultdict(list), defaultdict(list)
        # xattn_params, self.xattn_param_names = [], []
        xattn_params, xattn_param_names = defaultdict(list), defaultdict(list)
        itot_params, itot_param_names = defaultdict(list), defaultdict(list)
        gain_params, self.gain_param_names = [], []
        other_params, self.other_param_names = [], []
        ff_gain_params, self.ff_gain_param_names = [], []
        bread_params, self.bread_param_names = [], []
        capt_params, self.capt_param_names = [], []
        cattn_params, self.cattn_param_names = [], []
        for n, p in model.named_parameters():
            if n.startswith('clipmod.'):
                if self.freeze_capt_encoder:
                    p.requires_grad_(False)
                else:
                    self.capt_param_names.append(n)
                    capt_params.append(p)
            elif '.encoder_kv' in n:
                self.cattn_param_names.append(n)
                cattn_params.append(p)
            elif 'text_encoder' in n:
                # subname = 'text'
                if 'text_encoder.model.layers.' in n:
                    subname = 'textl.' + '.'.join(n.partition('text_encoder.model.layers.')[2].split('.')[:2])
                else:
                    subname = 'text.' + n.partition('text_encoder.')[2].split('.')[0]
                text_param_names[subname].append(n)
                text_params[subname].append(p)
            elif ("cross_attn" in n or "weave_attn.text_to_image_layers") and "gain" in n:
                if 'gain_ff' in n:
                    self.ff_gain_param_names.append(n)
                    ff_gain_params.append(p)
                else:
                    self.gain_param_names.append(n)
                    gain_params.append(p)
            elif "cross_attn" in n or "weave_attn.text_to_image_layers" in n:
                # subname = 'xattn'
                prefix = "cross_attn." if "cross_attn." in n else "weave_attn.text_to_image_layers."
                nsegs = 2 if weave_legacy_param_names else 3
                subname = 'xattn.' + '.'.join(n.partition(prefix)[2].split('.')[:nsegs])
                xattn_param_names[subname].append(n)
                xattn_params[subname].append(p)
                # self.xattn_param_names.append(n)
                # xattn_params.append(p)
            elif "weave_attn.image_to_text_layers" in n:
                prefix = 'weave_attn.image_to_text_layers.'
                nsegs = 2 if weave_legacy_param_names else 3
                subname = 'itot.' + '.'.join(n.partition(prefix)[2].split('.')[:nsegs])
                itot_param_names[subname].append(n)
                itot_params[subname].append(p)
            else:
                is_bread = False

                if n.startswith('out.') or n.startswith('bread'):
                    is_bread = True
                elif 'input_blocks' in n:
                    num = int(n.split('.')[1])
                    is_bread = num < param_sandwich
                elif 'output_blocks' in n:
                    num = int(n.split('.')[1])
                    is_bread = (len(model.output_blocks) - num - 1) < param_sandwich

                if is_bread:
                    print(f"is_bread: {n}")
                    self.bread_param_names.append(n)
                    bread_params.append(p)
                else:
                    self.other_param_names.append(n)
                    other_params.append(p)

        self.text_mods = list(text_param_names.keys())
        text_params = [text_params[n] for n in self.text_mods]
        self.text_param_names = [text_param_names[n] for n in self.text_mods]

        self.xattn_mods = list(xattn_param_names.keys())
        xattn_params = [xattn_params[n] for n in self.xattn_mods]
        self.xattn_param_names = [xattn_param_names[n] for n in self.xattn_mods]

        self.itot_mods = list(itot_param_names.keys())
        itot_params = [itot_params[n] for n in self.itot_mods]
        self.itot_param_names = [itot_param_names[n] for n in self.itot_mods]

        group_names = [*self.text_mods, *self.xattn_mods, *self.itot_mods, 'xgain', 'bread', 'other', 'capt', 'cattn', 'xgainff']
        param_name_groups = [*self.text_param_names, *self.xattn_param_names, *self.itot_param_names, self.gain_param_names, self.bread_param_names, self.other_param_names, self.capt_param_names, self.cattn_param_names, self.ff_gain_param_names]
        model_params = [*text_params, *xattn_params, *itot_params, gain_params, bread_params, other_params, capt_params, cattn_params, ff_gain_params]
        group_lrs =  [
            *[self.text_lr for _ in self.text_mods],
            *[self.text_lr for _ in self.xattn_mods],
            *[self.text_lr for _ in self.itot_mods],
            self.gain_lr, self.bread_lr, self.lr, self.capt_lr, self.lr, self.gain_lr
        ]

        self.group_names = []
        self.param_name_groups = []
        self.model_params = []
        self.group_lrs = []
        for gn, n, p, glr in zip(group_names, param_name_groups, model_params, group_lrs):
            if isinstance(p, list) and len(p) == 0:
                print(f"skipping empty {gn} with lr {glr}, names {n}")
                continue
            self.group_names.append(gn)
            self.param_name_groups.append(n)
            self.model_params.append(p)
            self.group_lrs.append(glr)

        # self.param_name_groups = [*self.text_param_names, *self.xattn_param_names, *self.itot_param_names, self.gain_param_names, self.bread_param_names, self.other_param_names, self.capt_param_names, self.ff_gain_param_names]
        #
        # self.model_params = [*text_params, *xattn_params, *itot_params, gain_params, bread_params, other_params, capt_params, ff_gain_params]

        self.master_params = self.model_params
        self.lg_loss_scale = lg_loss_scale
        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()

        self.grad_scaler = None
        if self.use_amp:
            self.use_fp16 = False  # avoid all the manual fp16 steps
            self._setup_amp()

        if self.use_fp16:
            self._setup_fp16()

        text_nparams = 0
        xattn_nparams = 0
        itot_nparams = 0
        for p, name in zip(self.master_params, self.group_names):
            if isinstance(p, list):
                nparams = sum(np.product(pp.shape) for pp in p)
            else:
                nparams = np.product(p.shape)
            prefix = '\t'
            if name in self.text_mods:
                text_nparams += nparams
                prefix += '\t'
            if name in self.xattn_mods:
                xattn_nparams += nparams
                prefix += '\t'
            if name in self.itot_mods:
                itot_nparams += nparams
                prefix += '\t'
            print(f"{prefix}{nparams/1e6:.2f}M {name} params")
        print(f"\t{text_nparams/1e6:.2f}M text params")
        print(f"\t{xattn_nparams/1e6:.2f}M xattn params")
        print(f"\t{itot_nparams/1e6:.2f}M itot params")

        param_groups = [
            {"params": params, "lr": lr, "weight_decay": 0.}
            for params, lr in zip(
                self.master_params,
                self.group_lrs,
            )
        ]
        self.opt = AdamW(
            param_groups,
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(beta1, beta2)
        )
        print('groups')
        for pg in param_groups:
            print(f'\t {len(pg["params"])} params, lr {pg["lr"]}, wd {pg["weight_decay"]}')

        print('groups')
        print(sum(len(pg['params']) for pg in param_groups))

        print('model params')
        print(len(list(self.model.parameters())))

        print('master params')
        print(len(self.master_params))

        # if not gain_ff_setup_step and not self.only_optimize_bread and len(ff_gain_params) > 0:
        #     self.opt.add_param_group({"params": ff_gain_params, "lr": self.gain_lr, "weight_decay": 0.})

        if self.resume_step:
            try:
                self._load_optimizer_state()
            except ValueError as e:
                print("couldn't load opt")
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            try:
                self.ema_params = [
                    self._load_ema_parameters(rate) for rate in self.ema_rate
                ]
            except RuntimeError as e:
                raise e
                print("couldn't load ema")
                self.ema_params = [
                    copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
                ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        if gain_ff_setup_step:
            self.opt.add_param_group({"params": ff_gain_params, "lr": self.gain_lr, "weight_decay": 0.})

        if th.cuda.is_available() and not perf_no_ddl:
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if True: #dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                sd = dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                )
                newsd = apply_state_dict_sandwich(
                    self.model,
                    sd,
                    self.state_dict_sandwich,
                    self.state_dict_sandwich_manual_remaps,
                )

                newsd = apply_resize(
                    self.model,
                    newsd,
                    mult=self.resize_mult
                )

                incompatible_keys = self.model.load_state_dict(
                    newsd,
                    strict = False
                )
                print(incompatible_keys)

                # if self.state_dict_sandwich > 0:
                #     for n, p in self.model.named_parameters():
                #         print(f"{th.linalg.norm(p).item():.3f} | {n in newsd} | {n}")

        # dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if True: #dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self._state_dict_to_master_params(state_dict)

        # dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            ours = [v['params'] for v in self.opt.state_dict()['param_groups']]
            theirs = [v['params'] for v in state_dict['param_groups']]

            if not all(len(o) == len(t) for o, t in zip(ours, theirs)):
                # loading manual mp opt in amp
                their_exp_avg = [state_dict['state'][pg[0]]['exp_avg'] for pg in theirs]
                their_exp_avg_sq = [state_dict['state'][pg[0]]['exp_avg_sq'] for pg in theirs]

                their_exp_avg = unflatten_master_params(
                    self.model_params,
                    their_exp_avg
                )
                their_exp_avg_sq = unflatten_master_params(
                    self.model_params,
                    their_exp_avg_sq
                )

                for pg in theirs:
                    param_ix = pg[0]
                    state_dict['state'][param_ix]['exp_avg'] = their_exp_avg[param_ix]
                    state_dict['state'][param_ix]['exp_avg_sq'] = their_exp_avg_sq[param_ix]
                    state_dict['param_groups'][param_ix]['params'] = ours[param_ix]  # set param group to our enumeration
            try:
                self.opt.load_state_dict(state_dict)
            except ValueError as e:
                print(f"self.opt:\n{repr(ours)}\nloaded:\n{repr(theirs)}\n")
                raise e

    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params, master_device=self.master_device)
        self.model.convert_to_fp16()

    def _setup_amp(self):
        self.grad_scaler = th.cuda.amp.GradScaler(init_scale=2 ** self.lg_loss_scale, growth_interval=int(1 / self.fp16_scale_growth))

    def run_loop(self):
        t1 = time.time()
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)

            if self.use_profiler and (self.step > 0):
                with th.profiler.profile(with_stack=True, profile_memory=False, with_flops=False) as _p:
                    try:
                        self.run_step(batch, cond, verbose=True, single_fwd_only=True)
                    except Exception as e:
                        print(repr(e))
                print(_p.key_averages(
                    # group_by_stack_n=15
                ).table(sort_by="self_cuda_time_total", row_limit=50))
                _p.export_chrome_trace('chromeprof')
                raise ValueError('done saving')
            else:
                self.run_step(batch, cond, verbose = (self.step % self.log_interval == 0))

            if self.step % self.log_interval == 0:
                t2 = time.time()
                print(f"{t2-t1:.2f} sec")
                t1 = t2
                logger.dumpkvs()
            if (self.step % self.save_interval == 0) and (self.step > 0):
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond, verbose=False, single_fwd_only=False):
        self.forward_backward(batch, cond, verbose=verbose, single_fwd_only=single_fwd_only)
        if single_fwd_only:
            return
        if self.use_amp:
            self.optimize_amp()
        elif self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        self.log_step()

    def forward_backward(self, batch, cond, verbose=False, single_fwd_only=False):
        if self.use_amp:
            self.opt.zero_grad(set_to_none=True)
        else:
            zero_grad(self.model_params)
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                if k not in {'txt', 'capt'}
                else v[i : i + self.microbatch]
                for k, v in cond.items()
            }
            if not (self.model.txt) and 'txt' in micro_cond:
                del micro_cond['txt']
            if 'txt' in micro_cond:
                # micro_cond['txt'] = th.as_tensor(tokenize(self.tokenizer, micro_cond['txt']), device=dist_util.dev())

                txt = th.as_tensor(tokenize(self.tokenizer, micro_cond['txt']), device=dist_util.dev())
                capt = clip.tokenize(micro_cond['capt'], truncate=True).to(dist_util.dev())
                micro_cond['txt'] = txt
                micro_cond['capt'] = capt
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            if self.noise_cond:
                if 'low_res' not in micro_cond:
                    raise ValueError

                t_noise_cond, _, = self.noise_cond_schedule_sampler.sample(micro.shape[0], dist_util.dev())
                micro_cond['low_res'] = self.noise_cond_diffusion.q_sample(micro_cond['low_res'], t_noise_cond)
                micro_cond['cond_timesteps'] = t_noise_cond

            with th.cuda.amp.autocast(enabled=self.use_amp, dtype=th.bfloat16 if self.use_bf16 else th.float16):
                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.ddp_model,
                    micro,
                    t,
                    model_kwargs=micro_cond,
                )

                if last_batch or not self.use_ddp:
                    losses = compute_losses()
                else:
                    with self.ddp_model.no_sync():
                        losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )
                if i == 0:
                    warm = self.schedule_sampler._warmed_up(verbose=verbose)
                    if warm and verbose:
                        _weights = self.schedule_sampler.weights()
                        w_avg = np.average(np.arange(len(_weights)), weights=_weights)
                        w_avg_ref = np.average(np.arange(len(_weights)), weights=np.ones_like(_weights))
                        print(f"w_avg: {w_avg:.1f} (vs {w_avg_ref:.1f})")

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            if single_fwd_only:
                break
            grad_acc_scale = micro.shape[0] / self.batch_size
            if self.use_fp16:
                loss_scale = 2 ** self.lg_loss_scale
                (loss * loss_scale * grad_acc_scale).backward()
            elif self.use_amp:
                self.grad_scaler.scale(loss * grad_acc_scale).backward()
            else:
                (loss * grad_acc_scale).backward()

    def _update_ema(self, params, rate, arith_from_step=0, arith_extra_shift=0, verbose=True):
        def _vprint(*args, **kwargs):
            if verbose:
                print(*args, **kwargs)
        if arith_from_step >= 0:
            n = (self.step + self.resume_step) - arith_from_step + 2  # divisor is 1/2 at first step
            n = n + arith_extra_shift  # for after first save/load
            _vprint(f"using n={n}, vs 1/(1-rate) {1/(1-rate):.1f} | ", end="")
            if n >= 1/(1-rate):
                _vprint('update_ema')
                update_ema(params, self.master_params, rate=rate)
            else:
                _vprint('update_arithmetic_average')
                update_arithmetic_average(params, self.master_params, n=n)
        else:
            update_ema(params, self.master_params, rate=rate)

    def optimize_fp16(self):
        if any(not th.isfinite(p.grad).all() for ps in self.model_params for p in ps if p.grad is not None):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params, master_device=self.master_device)
        for mp in self.master_params:
            mp.grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        verboses = [False] * (len(self.ema_rate) - 1) + [True]
        for rate, params, arith_from_step, arith_extra_shift, verbose in zip(
            self.ema_rate,
            self.ema_params,
            self.arithmetic_avg_from_step,
            self.arithmetic_avg_extra_shift,
            verboses
        ):
            self._update_ema(params, rate=rate, arith_from_step=arith_from_step, arith_extra_shift=arith_extra_shift,
                             verbose=verbose)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        verboses = [False] * (len(self.ema_rate) - 1) + [True]
        for rate, params, arith_from_step, arith_extra_shift, verbose in zip(
            self.ema_rate,
            self.ema_params,
            self.arithmetic_avg_from_step,
            self.arithmetic_avg_extra_shift,
            verboses
        ):
            self._update_ema(params, rate=rate, arith_from_step=arith_from_step, arith_extra_shift=arith_extra_shift,
                             verbose=verbose)

    def optimize_amp(self):
        self.grad_scaler.unscale_(self.opt)
        self._log_grad_norm()
        self._anneal_lr()
        self.grad_scaler.step(self.opt)
        self.grad_scaler.update()
        verboses = [False] * (len(self.ema_rate) - 1) + [True]
        for rate, params, arith_from_step, arith_extra_shift, verbose in zip(
            self.ema_rate,
            self.ema_params,
            self.arithmetic_avg_from_step,
            self.arithmetic_avg_extra_shift,
            verboses
        ):
            self._update_ema(params, rate=rate, arith_from_step=arith_from_step, arith_extra_shift=arith_extra_shift,
                             verbose=verbose)

    def _log_grad_norm(self):
        sqsum = 0.0
        sqsum_text_encoder = 0.0
        has_text_encoder = False

        for p_ in self.master_params:
            if isinstance(p_, list):
                pp = p_
            else:
                pp = [p_]

            for p in pp:
                if p.grad is None:
                    continue
                sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

        gn_xattn, gn_text, gn_itot, gn_capt, gn_cattn = 0., 0., 0., 0., 0.

        # name_to_norm = {}
        # name_to_nparam = {}
        # for n, p in self.model.named_parameters():
        #     name_to_norm[n] = p.grad.float().norm().item()
        #     name_to_nparam[n] = int(np.product(p.shape))
        # for n in sorted(name_to_norm.keys(), key=lambda n_: name_to_norm[n_]):
        #     print(f"{name_to_norm[n]:.4e}\t | {name_to_nparam[n]:08d}\t | {n}")

        for p_, name in zip(self.master_params, self.group_names):
            if isinstance(p_, list):
                pp = p_
            else:
                pp = [p_]

            if len(pp) == 0:
                continue

            # vals = []
            gn = 0.
            for p in pp:
                if p.grad is None:
                    continue
                gn_sq = (p.grad.float() ** 2).sum().item()
                gn += gn_sq
                if name in self.text_mods:
                    gn_text += gn_sq
                elif name in self.xattn_mods:
                    gn_xattn += gn_sq
                elif name in self.itot_mods:
                    gn_itot += gn_sq
                # vals.append(gn_sq)
            gn = np.sqrt(gn)
            # vals = sorted(vals)
            # top = [np.sqrt(x) for x in vals[-3:]]
            # bottom = [np.sqrt(x) for x in vals[:3]]
            # print(f"grad_norm_{name}: {gn:.3f} for {len(pp)} params\n\ttop {top}\n\tbottom {bottom}")
            logger.logkv_mean(f"grad_norm_{name}", gn)
            if name == 'cattn':
                gn_cattn = gn

        gn_text = np.sqrt(gn_text)
        logger.logkv_mean(f"grad_norm_text", gn_text)

        gn_xattn = np.sqrt(gn_xattn)
        logger.logkv_mean(f"grad_norm_xattn", gn_xattn)

        if gn_itot > 0:
            gn_itot = np.sqrt(gn_itot)
            logger.logkv_mean(f"grad_norm_itot", gn_itot)

        if (gn_text is not None) and (gn_xattn is not None):
            logger.logkv_mean(f"grad_norm_xt_ratio", gn_xattn / max(gn_text, 1e-8))

        if gn_itot > 0:
            logger.logkv_mean(f"grad_norm_xi_ratio", gn_xattn / max(gn_itot, 1e-8))

        if gn_cattn > 0:
            logger.logkv_mean(f"grad_norm_xxc_ratio", gn_xattn / max(gn_cattn, 1e-8))

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            frac_done = 0.
        else:
            frac_done = max(0, self.step + self.resume_step - self.lr_warmup_steps) / self.lr_anneal_steps

        if not self.lr_warmup_steps:
            frac_warmup_done = 1.
        else:
            # +1 so we don't use zero lr on first step
            frac_warmup_done = min(1., (self.step + self.resume_step + 1 - self.lr_warmup_shift) / self.lr_warmup_steps)

        lr_variants = self.group_lrs
        # lr_variants = (len(self.opt.param_groups)-5) * [self.text_lr] + [self.gain_lr, self.bread_lr, self.lr, self.capt_lr, self.gain_lr]

        mult = frac_warmup_done if frac_warmup_done < 1 else (1 - frac_done)
        logger.logkv("learning_rate", self.lr * mult)
        # print(f"mult: {mult} | frac_warmup_done {frac_warmup_done} | frac_done {frac_done}")

        for param_group, lr_variant in zip(self.opt.param_groups, lr_variants):
            this_lr = lr_variant * mult
            state_lr = param_group["lr"]
            if not self.anneal_log_flag:
                print(f"for group with {len(param_group['params'])} params, setting lr to {this_lr:.4e} (was {state_lr:.4e})")
            param_group["lr"] = this_lr

        if not self.anneal_log_flag:
            self.anneal_log_flag = True

    def log_gain(self):
        for n, m in self.model.named_modules():
            for attrname, methodname, suffix in [
                ('gain', 'effective_gain', ''), ('gain_ff', 'effective_gain_ff', '_ff')
            ]:
                if hasattr(m, attrname):
                    gain_val = getattr(m, methodname)()
                    if gain_val.ndim < 1 or len(gain_val) == 1:
                        gain_val = gain_val.item()
                    else:
                        gain_val = gain_val.detach().abs().mean().item()
                    short_name = ".".join(seg[:3] for seg in n.split(".") if seg[:3] != 'cro')
                    logger.logkv(f"gain_{short_name}{attrname}", gain_val)

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)
        if self.use_amp:
            logger.logkv("lg_loss_scale", np.log2(self.grad_scaler.get_scale()))
        self.log_gain()

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            if self.freeze_capt_encoder:
                state_dict = {k: v for k, v in state_dict.items() if not k.startswith('clipmod')}
            if True: # dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        logger.log("saving opt...")

        if True: # dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        logger.log("done saving locally")

        if self.autosave:
            save_progress_to_gcs(step=self.step+self.resume_step, ema_rates=self.ema_rate, autosave_dir=self.autosave_dir)

        # dist.barrier()

    def _master_params_to_state_dict(self, master_params):
        if self.use_fp16:
            master_params = unflatten_master_params(
                #list(self.model.parameters()),
                self.model_params,
                master_params
            )
        state_dict = self.model.state_dict()
        if self.use_amp:
            for p, name_or_group in zip(master_params, self.param_name_groups):
                if isinstance(name_or_group, list):
                    for name, pp in zip(name_or_group, p):
                        state_dict[name] = pp
                else:
                    name = name_or_group
                    state_dict[name] = p
        else:
            names_flat = [name for names in self.param_name_groups for name in names]
            for i, name in enumerate(names_flat):
                    assert name in state_dict
                    state_dict[name] = master_params[i]

        # for i, (name, _value) in enumerate(self.model.named_parameters()):
        #     assert name in state_dict
        #     state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        # params = [[state_dict[name] for name in name_group] for name_group in self.param_name_groups]
        def _debug_get(sd, name, fallback):
            if name in sd:
                return sd[name]
            print(f'{repr(name)} not found, falling back to\n\t{repr(fallback)}\n')
            return copy.deepcopy(fallback)

        params = [
            [_debug_get(state_dict, name, p) for name, p in zip(name_group, param_group)]
            for name_group, param_group in zip(self.param_name_groups, self.master_params)
        ]

        if self.use_fp16:
            return make_master_params(params)
        else:
            return params


def save_progress_to_gcs(step, ema_rates, autosave_dir):
    def _run_and_log(command):
        print(f"running {repr(command)}")
        return subprocess.check_output(command, shell=True)

    # construct gcs dir
    logdir = get_blob_logdir()
    logdir_final = [s for s in logdir.split('/') if len(s) > 0][-1]

    if not autosave_dir.endswith('/'):
        autosave_dir = autosave_dir + '/'

    experiment_autosave_dir = autosave_dir + logdir_final
    logger.log(f"copying to {repr(experiment_autosave_dir)}")

    prefixd = f"{step:06d}"

    fn_progress_base = os.path.join(logdir, f"progress.csv")
    fn_progress = os.path.join(logdir, f"progress{step}.csv")
    _run_and_log(f"cp {fn_progress_base} {fn_progress}")

    fn_segs = [f'model{prefixd}.pt', f'opt{prefixd}.pt']
    fn_segs += [f'ema_{rate}_{prefixd}.pt' for rate in ema_rates]
    fn_segs = [os.path.join(logdir, s) for s in fn_segs]
    fn_segs.append(fn_progress)

    fns_joined = " ".join(fn_segs)
    gcs_up_command = f"gsutil -m cp {fns_joined} {experiment_autosave_dir}"
    _run_and_log(gcs_up_command)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

            if key == 'mse':
                sub_snr = diffusion.snr[int(sub_t)]
                # 0 = coarse, 1 = content, 2 = cleanup
                stage = 0 if sub_snr < 1e-2 else (1 if sub_snr < 1 else 2)
                logger.logkv_mean(f"{key}_s{stage}", sub_loss)


def apply_resize(model, sd, mult=1., debug=False):
    for n, p in model.named_parameters():
        if n not in sd:
            continue
        if p.shape != sd[n].shape:
            print(f"resize\t{n}\t\t{sd[n].shape} -> {p.shape}")
            slices = tuple(slice(0, i) for i in sd[n].shape)
            with th.no_grad():
                buffer = p.data.clone()

                mod = get_child_module_by_names(model, n.split('.')[:-1])
                if mod is None:
                    raise ValueError(n)
                # is_norm_w = n.endswith('ln.weight') or n.endswith('normalization.weight')
                is_norm_w = n.endswith('.weight') and (isinstance(mod, th.nn.GroupNorm) or isinstance(mod, th.nn.LayerNorm))

                if debug:
                    debug_slices = tuple(slice(max(0, i-2), min(j, i+2)) for i, j in zip(sd[n].shape, buffer.shape))
                    print(f"before {n}\t{repr(buffer[debug_slices].squeeze())}")
                if is_norm_w:
                    print(f'not scaling\t{n}')
                else:
                    buffer.mul_(mult)
                if debug:
                    print(f"after scale\t{n}\n{repr(buffer[debug_slices].squeeze())}")
                buffer.__setitem__(slices, sd[n])
                if debug:
                    print(f"after set\t{n}\n{repr(buffer[debug_slices].squeeze())}")
                sd[n] = buffer
    return sd


def apply_state_dict_sandwich(model, sd, state_dict_sandwich, state_dict_sandwich_manual_remaps=None):
    if state_dict_sandwich <= 0:
        return sd

    if state_dict_sandwich_manual_remaps is None:
        state_dict_sandwich_manual_remaps = {}

    ks = list(sd.keys())
    newsd = {}

    for k in ks:
        if k.startswith("input_blocks."):
            segs = k.split('.')
            num = int(segs[1])
            if num == 0:
                if hasattr(model, 'bread_adapter_in'):
                    # remap input transducer
                    v = sd[k]
                    newk = 'bread_adapter_in.transducer.' + '.'.join(segs[3:])
                    print(f'{v.shape} {k} -> {newk}')
                    newsd[newk] = v
                else:
                    # skip input transducer
                    print(f"skipping {k}")
                    continue
            else:
                v = sd[k]
                segs[1] = str(num + state_dict_sandwich)
                newk = '.'.join(segs)
                print(f'{v.shape} {k} -> {newk}')
                newsd[newk] = v
        elif k.startswith("out."):
            if hasattr(model, 'bread_adapter_out'):
                # remap input transducer
                v = sd[k]
                newk = 'bread_adapter_out.transducer.' + '.'.join(k.split('.')[1:])
                print(f'{v.shape} {k} -> {newk}')
                newsd[newk] = v
            else:
                # skip output transducer
                print(f"skipping {k}")
        else:
            newk = k
            for prefix in state_dict_sandwich_manual_remaps:
                if k.startswith(prefix):
                    newprefix = state_dict_sandwich_manual_remaps[prefix]
                    _, _, suffix = k.partition(prefix)
                    newk = newprefix + suffix
                    print(f'{sd[k].shape} {k} -> {newk}')
            newsd[newk] = sd[k]
    return newsd
