"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math
import random
from functools import lru_cache

import numpy as np
import torch as th

from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def get_schedule_fn(schedule_name, num_diffusion_timesteps):
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        def schedule_fn(t):
            ratio = t / (num_diffusion_timesteps - 1)
            return beta_start + ratio * (beta_end - beta_start)
        return schedule_fn
    else:
        return None
        # raise NotImplementedError(f"get_schedule_fn: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB
    RESCALED_MSE_BALANCED = enum.auto()
    RESCALED_MSE_V = enum.auto()
    RESCALED_MSE_SNR_PLUS_ONE = enum.auto()

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL

    def is_mse(self):
        return self == LossType.MSE or self == LossType.RESCALED_MSE or self == LossType.RESCALED_MSE_BALANCED or LossType.RESCALED_MSE_V or self == LossType.RESCALED_MSE_SNR_PLUS_ONE


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
        schedule_fn=None,
        vb_loss_ratio=1000.,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
        self.vb_loss_ratio = vb_loss_ratio

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.log_betas = np.log(self.betas)

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.one_minus_alphas_cumprod = 1.0 - self.alphas_cumprod

        #  1/(snr + 1) = 1 - alpha_cumprod, percept paper eqn 4 comment
        self.recip_snrp1_clipped = np.clip(self.one_minus_alphas_cumprod, a_min=1e-2, a_max=None)
        self.recip_snrp1_clipped_normalizer = self.recip_snrp1_clipped.mean()

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        self.schedule_fn = schedule_fn

        self.snr = 1.0 / (1 - self.alphas_cumprod) - 1

        self.tensorized_for = None

    def is_tensorized(self, device):
        # out = self.tensorized_for == device
        # if out:
        #     print(f"DIFF YES | {device}")
        # else:
        #     print(f"DIFF NO  | have {self.tensorized_for} want {device}")
        # return out
        return self.tensorized_for == device

    def tensorize(self, device):
        # print("GaussianDiffusion tensorize called")
        arrays = {name: getattr(self, name) for name in vars(self) if isinstance(getattr(self, name), np.ndarray)}

        for name, arr in arrays.items():
            setattr(self, name, th.from_numpy(arr).to(device=device, dtype=th.float))

        self.tensorized_for = device

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = self._extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)

        # are we doing clf free guide?
        guidance_scale = model_kwargs.get("guidance_scale", 0)
        unconditional_key = "unconditional_model_kwargs"
        t_py = set(t.cpu().tolist())
        if "txt_guidance_drop_ixs" in model_kwargs and (t_py.intersection(model_kwargs["txt_guidance_drop_ixs"]) != set()):
            unconditional_key = "unconditional_drop_model_kwargs"
        unconditional_model_kwargs = model_kwargs.get(unconditional_key)
        guidance_after_step = float(model_kwargs.get("guidance_after_step", 100000.))
        is_eps = self.model_mean_type == ModelMeanType.EPSILON
        effective_guidance_scale = th.where(t < guidance_after_step, float(guidance_scale), 0.)
        can_skip = (effective_guidance_scale <= 0).all()
        # can_skip = False
        is_guided = (guidance_scale is not None) and (unconditional_model_kwargs is not None) and is_eps and (not can_skip)
        # print(f"is_guided {is_guided} | can_skip {can_skip} | guidance_scale {guidance_scale} | is_eps {is_eps}")

        drop_args = {
            "guidance_scale", "guidance_after_step", "unconditional_model_kwargs",
            "unconditional_drop_model_kwargs", "txt_guidance_pdrop", "txt_guidance_drop_ixs"
        }
        model_kwargs_cond = {k: v for k, v in model_kwargs.items() if k not in drop_args}
        model_output = model(x, self._scale_timesteps(t), **model_kwargs_cond)

        unconditional_model_output = None
        if is_guided:
            unconditional_model_output = model(x, self._scale_timesteps(t), **unconditional_model_kwargs)

            # broadcast
            effective_guidance_scale = effective_guidance_scale.reshape([-1] + [1 for _ in model_output.shape[1:]])

            # print(effective_guidance_scale)
            model_output = (1 + effective_guidance_scale) * model_output - effective_guidance_scale * unconditional_model_output

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if is_guided:
                # don't guide variance
                _, model_var_values = th.split(unconditional_model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = self._extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = self._extract_into_tensor(self.log_betas, t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = self._extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
                # if is_guided:
                #     unconditional_pred_xstart = process_xstart(
                #         self._predict_xstart_from_eps(x_t=x, t=t, eps=unconditional_model_output)
                #     )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
            # if is_guided:
            #     # print(f'using guidance scale {guidance_scale}')
            #     unconditional_model_mean, _, _ = self.q_posterior_mean_variance(
            #         x_start=unconditional_pred_xstart, x_t=x, t=t
            #     )
            #     model_mean = (1 + guidance_scale) * model_mean - guidance_scale * unconditional_model_mean
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "model_var_values": model_var_values
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            self._extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - self._extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def p_sample(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = model.device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        trange = ts_index_range(shape[0], self.num_timesteps, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            # t = th.tensor([i] * shape[0], device=device)
            t = trange[i]
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = self._extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = self._extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t+1,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = self._extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def plms_steps(
        self,
        model,
        x,
        t,
        t2,
        old_eps,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
        ddim_fallback=False,
        use_model_var=True,
    ):
        def model_step(x_, t_):
            out = self.p_mean_variance(
                model,
                x_,
                t_,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            eps = self._predict_eps_from_xstart(x_, t_, out["pred_xstart"])
            model_var_values = out['model_var_values']
            return eps, model_var_values

        def transfer(x_, eps, t1_, t2_, model_var_values):
            xstart = self._predict_xstart_from_eps(x_, t1_, eps)
            if clip_denoised:
                xstart = xstart.clamp(-1, 1)

            alpha_bar_t1 = self._extract_into_tensor(self.alphas_cumprod, t1_, x.shape)
            alpha_bar_t2 = self._extract_into_tensor(self.alphas_cumprod, t2_, x.shape)

            frac = (model_var_values + 1) / 2
            if use_model_var:
                min_log = th.log(((1 - alpha_bar_t2) / (1 - alpha_bar_t1)) * (1 - alpha_bar_t1 / alpha_bar_t2))
                max_log = th.log((1 - alpha_bar_t1 / alpha_bar_t2))
                max_log = th.min(th.log(1 - alpha_bar_t2), max_log)  # prevent sqrt(neg)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                # print((min_log[0,0,0,0], max_log[0,0,0,0]))

                model_log_variance = frac * max_log + (1 - frac) * min_log
                sigma = th.sqrt(th.exp(model_log_variance))
            else:
                sigma = (
                    eta
                    * th.sqrt((1 - alpha_bar_t2) / (1 - alpha_bar_t1))
                    * th.sqrt(1 - alpha_bar_t1 / alpha_bar_t2)
                )

            coef_xstart = th.sqrt(alpha_bar_t2)
            coef_eps = th.sqrt(1 - alpha_bar_t2 - sigma ** 2)

            mean_pred = xstart * coef_xstart + coef_eps * eps
            noise = th.randn_like(x_)
            nonzero_mask = (
                (t1_ != 0).float().view(-1, *([1] * (len(x_.shape) - 1)))
            )  # no noise when t == 0
            sample = mean_pred + nonzero_mask * sigma * noise

            return sample, xstart

        eps, model_var_values = model_step(x, t)

        if ddim_fallback:
            eps_prime = eps
        else:
            eps_prime = (55 * eps - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24
        # eps_prime = eps  # debug
        x_new, pred = transfer(x, eps_prime, t, t2, model_var_values)
        return {"sample": x_new, "pred_xstart": pred, 'eps': eps}

    def prk_double_step(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
        ddim_fallback=False,
    ):
        def model_step(x_, t_):
            out = self.p_mean_variance(
                model,
                x_,
                t_,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            eps = self._predict_eps_from_xstart(x_, t_, out["pred_xstart"])
            return eps

        def transfer(x_, eps, t1_, t2_, eta_=0.0):
            xstart = self._predict_xstart_from_eps(x_, t1_, eps)
            if clip_denoised:
                xstart = xstart.clamp(-1, 1)

            alpha_bar_t1 = self._extract_into_tensor(self.alphas_cumprod, t1_, x.shape)
            alpha_bar_t2 = self._extract_into_tensor(self.alphas_cumprod, t2_, x.shape)

            sigma = (
                eta_
                * th.sqrt((1 - alpha_bar_t2) / (1 - alpha_bar_t1))
                * th.sqrt(1 - alpha_bar_t1 / alpha_bar_t2)
            )

            coef_xstart = th.sqrt(alpha_bar_t2)
            coef_eps = th.sqrt(1 - alpha_bar_t2 - sigma ** 2)

            mean_pred = xstart * coef_xstart + coef_eps * eps
            noise = th.randn_like(x_)
            nonzero_mask = (
                (t1_ != 0).float().view(-1, *([1] * (len(x_.shape) - 1)))
            )  # no noise when t == 0
            sample = mean_pred + nonzero_mask * sigma * noise

            return sample, xstart

        t1 = t
        t_mid = t-1
        t2 = t-2

        eps1 = model_step(x, t1)

        if ddim_fallback:
            eps_prime = eps1
        else:
            x1, _ = transfer(x, eps1, t1, t_mid)

            eps2 = model_step(x1, t_mid)
            x2, _ = transfer(x, eps2, t1, t_mid)

            eps3 = model_step(x2, t_mid)
            x3, _ = transfer(x, eps3, t1, t2)

            eps4 = model_step(x3, t2)

            eps_prime = (eps1 + 2 * eps2 + 2 * eps3 + eps4) / 6
        # eps_prime = eps1
        x_new, pred = transfer(x, eps_prime, t1, t2, eta_=eta)
        # eps_prime = eps1  # debug
        # x_new, pred = transfer(x, eps_prime, t1, t_mid)  # debug

        return {"sample": x_new, "pred_xstart": pred, 'eps': eps_prime}

    def prk_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = model.device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        # indices = list(range(self.num_timesteps))[::-1]
        trange = ts_index_range(shape[0], self.num_timesteps, device=device)
        indices = list(range(2, self.num_timesteps, 2))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        old_eps = []
        for i in indices:
            # t = th.tensor([i] * shape[0], device=device)
            t = trange[i]
            with th.no_grad():
                out = self.prk_double_step(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                old_eps.append(out['eps'])
                # print(('rk', i, [t[0, 0, 0, 0] for t in old_eps]))

                yield out
                img = out["sample"]
        # return img

    def prk_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        final = None
        for sample in self.prk_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]

    def plms_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        ddim_first_n=0,
        ddim_last_n=None,
    ):
        if device is None:
            device = model.device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        trange = ts_index_range(shape[0], self.num_timesteps, device=device)

        rk_indices = [self.num_timesteps - j for j in [1, 3, 5]]
        plms_indices = list(range(1, self.num_timesteps-6, 1))[::-1]

        step_counter = 0

        old_eps = []
        for i in rk_indices:
            t = trange[i]
            with th.no_grad():
                ddim_fallback = (step_counter < ddim_first_n) or (ddim_last_n is not None and (nsteps - step_counter) < ddim_last_n)
                out = self.prk_double_step(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    eta=eta if ddim_fallback else 0.0,
                    ddim_fallback=ddim_fallback
                )
                old_eps.append(out['eps'])
                step_counter += 1

                yield out
                img = out["sample"]
        for i in plms_indices:
            t = trange[i]
            with th.no_grad():
                ddim_fallback = (step_counter < ddim_first_n) or (ddim_last_n is not None and (nsteps - step_counter) < ddim_last_n)
                out = self.plms_steps(
                    model,
                    img,
                    t,
                    t2=(t-1).clamp(min=0),
                    old_eps=old_eps,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    eta=eta if ddim_fallback else 0.0,
                    ddim_fallback=ddim_fallback,
                    use_model_var=ddim_fallback,
                )
                old_eps.pop(0)
                old_eps.append(out['eps'])
                step_counter += 1

                yield out
                img = out["sample"]

        # final step
        with th.no_grad():
            out = self.p_mean_variance(
                model,
                img,
                trange[0],
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            yield {"sample": out["mean"], "pred_xstart": out["pred_xstart"]}

    def plms_sample_loop(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            eta=0.0,
            ddim_first_n=0,
            ddim_last_n=None,
        ):
            final = None
            for sample in self.plms_sample_loop_progressive(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                eta=eta,
                ddim_first_n=ddim_first_n,
                ddim_last_n=ddim_last_n
            ):
                final = sample
            return final["sample"]

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = model.device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        trange = ts_index_range(shape[0], self.num_timesteps, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            # t = th.tensor([i] * shape[0], device=device)
            t = trange[i]
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type.is_mse():
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                vb_out = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )
                terms["vb"] = vb_out["output"]
                if self.loss_type == LossType.RESCALED_MSE or self.loss_type == LossType.RESCALED_MSE_BALANCED or self.loss_type == LossType.RESCALED_MSE_SNR_PLUS_ONE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / self.vb_loss_ratio

            if self.loss_type == LossType.RESCALED_MSE_SNR_PLUS_ONE:
                ## OLD VERSION: (snr + 1) / snr

                # snr = self._extract_into_tensor(self.alphas_cumprod, t, model_output.shape) / self._extract_into_tensor(self.one_minus_alphas_cumprod, t, model_output.shape)
                # ratio = ((snr + 1.) / snr)
                # normalizer = (self.alphas_cumprod / self.one_minus_alphas_cumprod).mean()

                ## NEW VERSION: 1 / (snr + 1)
                #  1/(snr + 1) = 1 - alpha_cumprod, percept paper eqn 4 comment
                recip_snrp1_clipped = self._extract_into_tensor(self.recip_snrp1_clipped, t, model_output.shape)
                ratio = recip_snrp1_clipped
                normalizer = self.recip_snrp1_clipped_normalizer
                # normalizer = 1.

                target = noise
                mse_base = (target - model_output) ** 2
                ratio_weights = ratio / normalizer
                terms["mse"] = mean_flat(ratio_weights * mse_base)
            elif self.loss_type == LossType.RESCALED_MSE_V:
                # don't this this is correct...
                pred_xstart = self._predict_xstart_from_eps(x_t=x_t, t=t, eps=model_output)
                v_alpha = self._extract_into_tensor(self.sqrt_alphas_cumprod, t, pred_xstart.shape)
                v_sigma = self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, pred_xstart.shape)
                v = v_alpha * model_output - v_sigma * pred_xstart
                target = v_alpha * noise - v_sigma * x_start
                terms["mse"] = mean_flat((target - v) ** 2)
            elif self.loss_type == LossType.RESCALED_MSE_BALANCED:
                pred_xstart = self._predict_xstart_from_eps(x_t=x_t, t=t, eps=model_output)
                mse_eps = mean_flat((noise - model_output) ** 2)
                mse_xstart = mean_flat((x_start - pred_xstart) ** 2)
                terms['mse'] = (mse_eps + mse_xstart) / 2.
            else:
                target = {
                    ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                        x_start=x_start, x_t=x_t, t=t
                    )[0],
                    ModelMeanType.START_X: x_start,
                    ModelMeanType.EPSILON: noise,
                }[self.model_mean_type]
                assert model_output.shape == target.shape == x_start.shape
                terms["mse"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None, progress=False):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []

        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for t in indices:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.

        :param arr: the 1-D numpy array.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        if self.is_tensorized(timesteps.device):
            try:
                res = arr[timesteps]
            except TypeError as e:
                print(type(timesteps), type(arr))
                print((getattr(timesteps, 'device'), timesteps.dtype))
                print((getattr(arr, 'device'), arr.dtype))
                raise e
        else:
            res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
            self.tensorize(timesteps.device)
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)


class SimpleForwardDiffusion:
    def __init__(
        self,
        betas,
    ):
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.tensorized_for = None

    def is_tensorized(self, device):
        # out = self.tensorized_for == device
        # if out:
        #     print(f"DIFF YES | {device}")
        # else:
        #     print(f"DIFF NO  | have {self.tensorized_for} want {device}")
        # return out
        return self.tensorized_for == device

    def tensorize(self, device):
        arrays = {name: getattr(self, name) for name in vars(self) if isinstance(getattr(self, name), np.ndarray)}

        for name, arr in arrays.items():
            setattr(self, name, th.from_numpy(arr).to(device=device, dtype=th.float))

        self.tensorized_for = device

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.

        :param arr: the 1-D numpy array.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        if self.is_tensorized(timesteps.device):
            try:
                res = arr[timesteps]
            except TypeError as e:
                print(type(timesteps), type(arr))
                print((getattr(timesteps, 'device'), timesteps.dtype))
                print((getattr(arr, 'device'), arr.dtype))
                raise e
        else:
            res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
            self.tensorize(timesteps.device)
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)


def ts_index_range(batch_size, maxstep, device):
    return _ts_index_range(batch_size, maxstep, str(device))


@lru_cache(1)
def _ts_index_range(batch_size, nsteps, device):
    # print(f"_ts_index_range called for {(batch_size, nsteps, device)}")
    with th.no_grad():
        tblock = th.tile(th.arange(nsteps, device=device), (batch_size, 1))
        ts = []
        for i in range(0, nsteps):
            ts.append(tblock[:, i:i+1].view(-1))
    return ts
