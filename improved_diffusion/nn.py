"""
Various utilities for neural networks.
"""

import math

import torch as th
import torch.nn as nn


# # PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
# class SiLU(nn.Module):
#     def __init__(self, use_checkpoint=False):
#         super().__init__()
#         self.use_checkpoint = use_checkpoint
#
#     def forward(self, x):
#         return checkpoint(
#             self._forward, (x,), self.parameters(), self.use_checkpoint
#         )
#
#     def _forward(self, x):
#         return x * th.sigmoid(x)

class SiLU(nn.SiLU):
    def __init__(self, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        return checkpoint(
            super().forward, (x,), self.parameters(), self.use_checkpoint
        )

# # from https://github.com/lukemelas/EfficientNet-PyTorch/blob/7e8b0d312162f335785fb5dcfa1df29a75a1783a/efficientnet_pytorch/utils.py
# # A memory-efficient implementation of Swish function
# class SwishImplementation(th.autograd.Function):
#     @staticmethod
#     def forward(ctx, i):
#         result = i * th.sigmoid(i)
#         ctx.save_for_backward(i)
#         return result
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         i = ctx.saved_tensors[0]
#         sigmoid_i = th.sigmoid(i)
#         return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))
#
#
# class MemoryEfficientSwish(nn.Module):
#     def forward(self, x):
#         return SwishImplementation.apply(x)
#
# SiLU = MemoryEfficientSwish

# SiLU = nn.SiLU


class GroupNorm32(nn.GroupNorm):
    def __init__(self, *args, use_checkpoint=False,
                 # force_fp32=False,  # !!!!!!! changed for deepspeed, breaking change to non-deepspeed, TODO: fix
                 force_fp32=True,
                 ):
        super().__init__(*args)
        self.use_checkpoint = use_checkpoint
        self.force_fp32 = force_fp32

    def forward(self, x):
        return checkpoint(
            self._forward, (x,), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x):
        if self.force_fp32:
            return super().forward(x.float()).type(x.dtype)
        return super().forward(x)

    # def _apply(self, fn, is_inner=False):
    #     super()._apply(fn)
    #     if not is_inner:
    #         print("GroupNorm32 possibly being asked to halve itself, refusing ;)")
    #         self._apply(lambda t: t.float(), is_inner=True)


class AdaGN(nn.Module):
    def __init__(self, emb_channels, out_channels, num_groups, nonlin_in=True, do_norm=True):
        super().__init__()
        self.emb_layers = nn.Sequential(
            SiLU() if nonlin_in else nn.Identity(),
            nn.Linear(emb_channels, 2 * out_channels)
        )
        self.normalization = nn.GroupNorm(num_groups, out_channels) if do_norm else nn.Identity()

    def forward(self, h, emb, side_emb=None):
        emb_out = self.emb_layers(emb).type(h.dtype)

        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if side_emb is not None:
            emb_out = emb_out + side_emb.type(emb_out.dtype)

        scale, shift = th.chunk(emb_out, 2, dim=1)
        h = self.normalization(h) * (1 + scale) + shift
        return h


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
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


def normalization(channels, use_checkpoint=False):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    if channels % 72 == 0:
        # hack
        return GroupNorm32(24, channels, use_checkpoint=use_checkpoint)
    return GroupNorm32(32, channels, use_checkpoint=use_checkpoint)


def normalization_1group(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(1, channels)


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
            ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
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
        return (None, None, None) + input_grads
