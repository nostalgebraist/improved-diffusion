"""
Various utilities for neural networks.
"""

import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from axial_positional_embedding import AxialPositionalEmbedding


# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLUImplOpenAI(nn.Module):
    def __init__(self, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        return checkpoint(
            self._forward, (x,), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x):
        return x * th.sigmoid(x)

class SiLUImplTorch(nn.SiLU):
    def __init__(self, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        return checkpoint(
            super().forward, (x,), self.parameters(), self.use_checkpoint
        )

# from https://github.com/lukemelas/EfficientNet-PyTorch/blob/7e8b0d312162f335785fb5dcfa1df29a75a1783a/efficientnet_pytorch/utils.py
# A memory-efficient implementation of Swish function
class SwishImplementation(th.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * th.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = th.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class SiLUImplEfficientNet(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


def silu(impl="torch", use_checkpoint=False):
    if impl == "fused":
        return nn.Identity()
    elif impl == "openai":
        return SiLUImplOpenAI(use_checkpoint=use_checkpoint)
    elif impl == "torch":
        return SiLUImplTorch(use_checkpoint=use_checkpoint)
    elif impl == "efficientnet":
        return SiLUImplEfficientNet()
    else:
        raise ValueError(impl)



@th.jit.script
def groupnorm_silu(x, ng, w, b):
    return F.silu(th.group_norm(x.float(), ng, w, b).type(x.dtype))


@th.jit.script
def groupnorm_silu_32(x, w, b):
    return F.silu(th.group_norm(x.float(), 32, w, b).type(x.dtype))


@th.jit.script
def groupnorm_silu_24(x, w, b):
    return F.silu(th.group_norm(x.float(), 24, w, b).type(x.dtype))


@th.jit.script
def groupnorm_silu_20(x, w, b):
    return F.silu(th.group_norm(x.float(), 20, w, b).type(x.dtype))


@th.jit.script
def groupnorm_silu_8(x, w, b):
    return F.silu(th.group_norm(x.float(), 8, w, b).type(x.dtype))


@th.jit.script
def groupnorm_silu_6(x, w, b):
    return F.silu(th.group_norm(x.float(), 6, w, b).type(x.dtype))


@th.jit.script
def groupnorm_silu_1(x, w, b):
    return F.silu(th.group_norm(x.float(), 1, w, b).type(x.dtype))


class GroupNorm32(nn.GroupNorm):
    def __init__(self, *args, use_checkpoint=False, fused=False):
        super().__init__(*args)
        self.use_checkpoint = use_checkpoint
        self.fused = fused

    def forward(self, x):
        return checkpoint(
            self._forward, (x,), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x):
        if self.fused:
            if self.num_groups == 32:
                return groupnorm_silu_32(x, self.weight, self.bias)
            elif self.num_groups == 24:
                return groupnorm_silu_24(x, self.weight, self.bias)
            elif num_groups == 20:
                return groupnorm_silu_20(x, self.weight, self.bias)
            else:
                raise ValueError(self.num_groups)
        return super().forward(x.float()).type(x.dtype)


class AdaGN(nn.Module):
    def __init__(self, emb_channels, out_channels, num_groups, silu_in=True, do_norm=True, base_channels=-1, silu_impl="torch"):
        super().__init__()
        self.emb_layers = nn.Sequential(
            silu(impl="torch" if silu_impl == "fused" else silu_impl) if silu_in else nn.Identity(),
            nn.Linear(emb_channels, 2 * out_channels)
        )
        if not do_norm:
            self.normalization = nn.Identity()
        elif base_channels > 0:
            self.normalization = GroupNormExtended(num_groups, out_channels, num_channels_base=base_channels)
        else:
            self.normalization = nn.GroupNorm(num_groups, out_channels)

        self.base_channels = base_channels
        self.out_channels = out_channels
        self.base_out_channels = base_channels * out_channels // emb_channels

        self.fused = silu_impl == "fused"

    def forward(self, h, emb, side_emb=None):
        # print(f'got image shape {h.shape}')
        # print(f'got emb shape {emb.shape}')
        # print(f'got base_channels {selfbase_channels}')
        # print(f'got out_channels {selfout_channels}')
        # print(f'got base_out_channels {selfbase_out_channels}')

        emb_out = self.emb_layers(emb).type(h.dtype)

        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if side_emb is not None:
            emb_out = emb_out + side_emb.type(emb_out.dtype)

        if self.fused:
            if self.base_channels > 0:
                base, xtra = th.split(
                    emb_out,
                    [2 * self.base_out_channels, 2 * self.out_channels - 2 * self.base_out_channels],
                    dim=1
                )
                base_h, xtra_h = th.split(h, [self.normalization.num_channels_base, self.normalization.num_channels_xtra], dim=1)
                if self.normalization.num_groups_xtra == 1:
                    fn = adagn_extended_32_1
                elif self.normalization.num_groups_xtra == 8:
                    fn = adagn_extended_32_8
                else:
                    raise ValueError(self.normalization.num_groups_xtra)
                h = fn(
                    base_h, xtra_h,
                    base, xtra,
                    self.normalization.weight, self.normalization.bias,
                    self.normalization.weight_xtra, self.normalization.bias_xtra,
                )
            else:
                return adagn(h, emb_out, self.normalization.weight, self.normalization.bias)
        else:
            if self.base_channels > 0:
                base, xtra = th.split(
                    emb_out,
                    [2 * self.base_out_channels, 2 * self.out_channels - 2 * self.base_out_channels],
                    dim=1
                )
                base_scale, base_shift = th.chunk(base, 2, dim=1)
                xtra_scale, xtra_shift = th.chunk(xtra, 2, dim=1)
                scale = th.cat([base_scale, xtra_scale], dim=1)
                shift = th.cat([base_shift, xtra_shift], dim=1)
            else:
                scale, shift = th.chunk(emb_out, 2, dim=1)
            h = self.normalization(h) * (1 + scale) + shift
        return h


@th.jit.script
def adagn(h, emb_out, w, b):
    scale, shift = th.chunk(emb_out, 2, dim=1)
    h = th.group_norm(h.float(), 32, w, b).type(h.dtype) * (1 + scale) + shift
    return h


@th.jit.script
def adagn_silu(h, emb_out, w, b):
    scale, shift = th.chunk(emb_out, 2, dim=1)
    h = th.group_norm(h.float(), 32, w, b).type(h.dtype) * (1 + scale) + shift
    return F.silu(h)


@th.jit.script
def adagn_silu_extended_32_8(h, h2, emb_out, emb_out2, w, b, w2, b2):
    h = th.group_norm(h.float(), 32, w, b).type(h.dtype)
    h2 = th.group_norm(h2.float(), 8, w2, b2).type(h.dtype)

    h = th.cat([h, h2], dim=1)

    scale, shift = th.chunk(emb_out, 2, dim=1)
    scale2, shift2 = th.chunk(emb_out2, 2, dim=1)
    scale = th.cat([scale, scale2], dim=1)
    shift = th.cat([shift, shift2], dim=1)
    h = h * (1 + scale) + shift
    return F.silu(h)


@th.jit.script
def adagn_silu_extended_32_6(h, h2, emb_out, emb_out2, w, b, w2, b2):
    h = th.group_norm(h.float(), 32, w, b).type(h.dtype)
    h2 = th.group_norm(h2.float(), 6, w2, b2).type(h.dtype)

    h = th.cat([h, h2], dim=1)

    scale, shift = th.chunk(emb_out, 2, dim=1)
    scale2, shift2 = th.chunk(emb_out2, 2, dim=1)
    scale = th.cat([scale, scale2], dim=1)
    shift = th.cat([shift, shift2], dim=1)
    h = h * (1 + scale) + shift
    return F.silu(h)


@th.jit.script
def adagn_silu_extended_32_1(h, h2, emb_out, emb_out2, w, b, w2, b2):
    h = th.group_norm(h.float(), 32, w, b).type(h.dtype)
    h2 = th.group_norm(h2.float(), 1, w2, b2).type(h.dtype)

    h = th.cat([h, h2], dim=1)

    scale, shift = th.chunk(emb_out, 2, dim=1)
    scale2, shift2 = th.chunk(emb_out2, 2, dim=1)
    scale = th.cat([scale, scale2], dim=1)
    shift = th.cat([shift, shift2], dim=1)
    h = h * (1 + scale) + shift
    return F.silu(h)


@th.jit.script
def adagn_extended_32_1(h, h2, emb_out, emb_out2, w, b, w2, b2):
    h = th.group_norm(h.float(), 32, w, b).type(h.dtype)
    h2 = th.group_norm(h2.float(), 1, w2, b2).type(h.dtype)

    h = th.cat([h, h2], dim=1)

    scale, shift = th.chunk(emb_out, 2, dim=1)
    scale2, shift2 = th.chunk(emb_out2, 2, dim=1)
    scale = th.cat([scale, scale2], dim=1)
    shift = th.cat([shift, shift2], dim=1)
    h = h * (1 + scale) + shift
    return h


@th.jit.script
def adagn_extended_32_8(h, h2, emb_out, emb_out2, w, b, w2, b2):
    h = th.group_norm(h.float(), 32, w, b).type(h.dtype)
    h2 = th.group_norm(h2.float(), 8, w2, b2).type(h.dtype)

    h = th.cat([h, h2], dim=1)

    scale, shift = th.chunk(emb_out, 2, dim=1)
    scale2, shift2 = th.chunk(emb_out2, 2, dim=1)
    scale = th.cat([scale, scale2], dim=1)
    shift = th.cat([shift, shift2], dim=1)
    h = h * (1 + scale) + shift
    return h


class Conv2D(nn.Conv2d):
    def __init__(self, *args, use_checkpoint=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        return checkpoint(
            super().forward, (x,), self.parameters(), self.use_checkpoint
        )


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return Conv2D(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ_, src_ in zip(target_params, source_params):
        inner_targ  = targ_ if isinstance(targ_, list) else [targ_]
        inner_src  = src_ if isinstance(src_, list) else [src_]
        for targ, src in zip(inner_targ, inner_src):
            targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def update_arithmetic_average(target_params, source_params, n):
    if n == 0:
        raise ValueError
    rate = 1 - (1/n)
    update_ema(target_params, source_params, rate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels, use_checkpoint=False, base_channels=-1, fused=False):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    cls, kwargs = GroupNorm32, {}
    if base_channels > 0:
        cls, kwargs = GroupNormExtended, {"num_channels_base": base_channels}
    hack72 = channels % 72 == 0
    if base_channels > 0:
        hack72 = base_channels % 72 == 0
    if hack72:
        # hack
        return cls(24, channels, use_checkpoint=use_checkpoint, fused=fused, **kwargs)
    return cls(32, channels, use_checkpoint=use_checkpoint, fused=fused, **kwargs)


def normalization_1group(channels, base_channels=-1):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(1, channels, base_channels=base_channels)


# @th.jit.script
# def groupnorm_extended_silu(x, sizes, ng, nch, w, b, ng_xtra, nch_xtra, w_xtra, b_xtra):
#     dtype = x.type()
#     x = x.float()
#
#     base, xtra = th.split(x, sizes[nch, nch_xtra], dim=1)
#     base_out = F.silu(th.group_norm(base, ng, w, b))
#     xtra_out = F.silu(th.group_norm(xtra, ng_xtra, w_xtra, b_xtra))
#
#     return th.cat([base_out, xtra_out], dim=1).type(dtype)


class GroupNormExtended(GroupNorm32):
    def __init__(self, num_groups, num_channels, num_channels_base, use_checkpoint=False, fused=False):
        super().__init__(num_groups, num_channels_base, use_checkpoint=use_checkpoint)

        self.num_channels_base = num_channels_base
        self.num_channels_xtra = num_channels - num_channels_base

        self.num_groups_base = num_groups

        for channels_per_group_xtra in range(*sorted([num_channels // num_groups, self.num_channels_xtra])):
            ratio, mod = self.num_channels_xtra / channels_per_group_xtra, self.num_channels_xtra % channels_per_group_xtra
            if self.num_channels_xtra % channels_per_group_xtra == 0:
                self.num_groups_xtra = self.num_channels_xtra // channels_per_group_xtra
                break

        # print(f"base ch {self.num_channels_base} gr {self.num_groups_base}, xtra ch {self.num_channels_xtra} gr {self.num_groups_xtra}")

        self.weight_xtra = nn.Parameter(th.empty(self.num_channels_xtra))
        self.bias_xtra = nn.Parameter(th.empty(self.num_channels_xtra))

        nn.init.ones_(self.weight_xtra)
        nn.init.zeros_(self.bias_xtra)

        self.fused = fused

    def _forward(self, x):
        if self.fused:
            # return groupnorm_extended_silu(
            #     x,
            #     self._num_groups_base, self._num_channels_base, self.weight, self.bias,
            #     self._num_groups_xtra, self._num_channels_xtra, self.weight_xtra, self.bias_xtra,
            # )
            base, xtra = th.split(x, [self.num_channels_base, self.num_channels_xtra], dim=1)
            if self.num_groups_base == 32:
                base_out = groupnorm_silu_32(base, self.weight, self.bias)
            elif self.num_groups_base == 24:
                base_out = groupnorm_silu_24(base, self.weight, self.bias)
            else:
                raise ValueError(self.num_groups_base)

            if self.num_groups_xtra == 32:
                xtra_out = groupnorm_silu_32(xtra, self.weight_xtra, self.bias_xtra)
            elif self.num_groups_xtra == 24:
                xtra_out = groupnorm_silu_24(xtra, self.weight_xtra, self.bias_xtra)
            elif self.num_groups_xtra == 20:
                xtra_out = groupnorm_silu_20(xtra, self.weight_xtra, self.bias_xtra)
            elif self.num_groups_xtra == 8:
                xtra_out = groupnorm_silu_8(xtra, self.weight_xtra, self.bias_xtra)
            elif self.num_groups_xtra == 6:
                xtra_out = groupnorm_silu_6(xtra, self.weight_xtra, self.bias_xtra)
            elif self.num_groups_xtra == 1:
                xtra_out = groupnorm_silu_1(xtra, self.weight_xtra, self.bias_xtra)
            else:
                raise ValueError(self.num_groups_xtra)
            # base_out = groupnorm_silu(base, self._num_groups_base, self.weight, self.bias)
            # xtra_out = groupnorm_silu(xtra, self._num_groups_xtra, self.weight_xtra, self.bias_xtra)
            return th.cat([base_out, xtra_out], dim=1)
        else:
            dtype = x.type()
            x = x.float()

            base, xtra = th.split(x, [self.num_channels_base, self.num_channels_xtra], dim=1)

            base_out = th.group_norm(base, self.num_groups_base, self.weight, self.bias, self.eps)
            xtra_out = th.group_norm(xtra, self.num_groups_xtra, self.weight_xtra, self.bias_xtra, self.eps)

            out = th.cat([base_out, xtra_out], dim=1).type(dtype)
            return out


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def expanded_timestep_embedding(timesteps, dim, base_dim, max_period=10000):
    base_half = base_dim // 2

    base_logfreqs = -math.log(max_period) * th.arange(start=0, end=base_half, dtype=th.float32) / base_half

    step = base_dim//(dim-base_dim)
    if step <= 2:
        left = base_logfreqs[0::step]
        right = base_logfreqs[1::step]
    else:
        left = base_logfreqs[step//2::step]
        right = base_logfreqs[step//2+1::step]

    xtra_logfreqs = (right + left)/2

    base_freqs = th.exp(base_logfreqs).to(device=timesteps.device)
    xtra_freqs = th.exp(xtra_logfreqs).to(device=timesteps.device)

    freqs = th.cat([base_freqs, xtra_freqs])

    base_args = timesteps[:, None].float() * base_freqs[None]
    xtra_args = timesteps[:, None].float() * xtra_freqs[None]
    embedding = th.cat(
        [th.cos(base_args), th.sin(base_args),
         th.cos(xtra_args), th.sin(xtra_args)],
        dim=-1
    )
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag, final_nograd=0):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        # print(f"ckpt final_nograd: {final_nograd}")
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), final_nograd, *args)
    else:
        return func(*inputs)


class CheckpointFunction(th.autograd.Function):
    @staticmethod
    @th.cuda.amp.custom_fwd
    def forward(ctx, run_function, length, final_nograd, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        ctx.final_nograd = final_nograd
        # print(f"fwd fn: {repr(run_function)}")
        # print(f"fwd length: {length}")
        # print(f"fwd final_nograd: {final_nograd}")
        # print(f"fwd ctx.final_nograd: {ctx.final_nograd}")
        with th.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    @th.cuda.amp.custom_bwd
    def backward(ctx, *output_grads):
        # print(f"bwd ctx.final_nograd: {ctx.final_nograd}")
        if ctx.final_nograd:
            ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors[:-ctx.final_nograd]] + ctx.input_tensors[-ctx.final_nograd:]
            grad_input_tensors = ctx.input_tensors[:-ctx.final_nograd]
        else:
            isnone = [t is None for t in ctx.input_tensors] + [False for _ in ctx.input_params]
            ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors if x is not None]
            grad_input_tensors = ctx.input_tensors
        with th.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = th.autograd.grad(
            output_tensors,
            grad_input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        ng = len(grad_input_tensors)
        del ctx.input_tensors
        del grad_input_tensors
        del ctx.input_params
        del output_tensors
        if ctx.final_nograd:
            return (None, None, None) + input_grads[:ng] + (ctx.final_nograd * (None,)) + input_grads[ng:]

        ct = 0
        input_grads_with_nones = []
        for val in isnone:
            if val:
                input_grads_with_nones.append(None)
            else:
                input_grads_with_nones.append(input_grads[ct])
                ct += 1
        return (None, None, None) + tuple(input_grads_with_nones)
        # return (None, None, None) + input_grads


class AxialPositionalEmbeddingShape(AxialPositionalEmbedding):
    def forward(self, shape, device, dtype):
        b, t, e = shape
        assert (t <= self.max_seq_len), f'Sequence length ({t}) must be less than the maximum sequence length allowed ({self.max_seq_len})'
        embs = []

        for ax_emb in self.weights.to_list():
            axial_dim = ax_emb.shape[-1]
            expand_shape = (b, *self.shape, axial_dim)
            emb = ax_emb.expand(expand_shape).reshape(b, self.max_seq_len, axial_dim)
            embs.append(emb)

        pos_emb = sum(embs) if self.summed else th.cat(embs, dim=-1)
        return pos_emb[:, :t].to(device=device, dtype=dtype)
