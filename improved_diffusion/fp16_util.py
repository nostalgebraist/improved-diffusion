"""
Helpers to train with 16-bit precision.
"""

import torch
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from .nn import LayerNorm32
from .text_nn import TextEncoder, CrossAttention, ImageToTextCrossAttention


def convert_module_to_f16(l, bf16=False):
    """
    Convert primitive modules to float16.
    """
    dtype = torch.float16
    if bf16:
        dtype = torch.bfloat16
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d, )):
        l.weight.data = l.weight.data.to(dtype)
        if l.bias is not None:
            l.bias.data = l.bias.data.to(dtype)
    if isinstance(l, (CrossAttention, ImageToTextCrossAttention, TextEncoder)):
        for n, p in l.named_parameters():
            if 'src_ln' in n or 'tgt_ln' in n:
                p.data = p.data.to(th.float)
            else:
                p.data = p.data.to(dtype)


def convert_module_to_f32(l):
    """
    Convert primitive modules to float32, undoing convert_module_to_f16().
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d, )):
        l.weight.data = l.weight.data.float()
        if l.bias is not None:
            l.bias.data = l.bias.data.float()
    if isinstance(l, (CrossAttention, TextEncoder, ImageToTextCrossAttention)):
        for n, p in l.named_parameters():
            p.data = p.data.float()


def make_master_params(model_param_groups, master_device=None):
    """
    Copy model parameters into a (differently-shaped) list of full-precision
    parameters.
    """
    def _to_master_device(t):
        if master_device is not None:
            return t.to(master_device)
        return t

    if isinstance(model_param_groups[0], nn.Parameter):
        model_param_groups = [model_param_groups]

    master_params = [
        _flatten_dense_tensors(
            [_to_master_device(param.detach()).float() for param in model_params]
        )
        for model_params in model_param_groups
    ]
    master_params = [nn.Parameter(mp) for mp in master_params]
    for mp in master_params:
        mp.requires_grad = True
    return master_params


def model_grads_to_master_grads(model_param_groups, master_params, master_device=None):
    """
    Copy the gradients from the model parameters into the master parameters
    from make_master_params().
    """
    def _to_master_device(t):
        if master_device is not None:
            return t.to(master_device)
        return t

    if isinstance(model_param_groups[0], nn.Parameter):
        model_param_groups = [model_param_groups]

    for model_group, master_param in zip(model_param_groups, master_params):
        master_param.grad = _flatten_dense_tensors(
            [_to_master_device(param.grad.data.detach()).float()
             if param.grad is not None else None
             for param in model_group]
        )


def master_params_to_model_params(model_param_groups, master_params):
    """
    Copy the master parameter data back into the model parameters.
    """
    # Without copying to a list, if a generator is passed, this will
    # silently not copy any parameters.
    model_param_groups = list(model_param_groups)

    if isinstance(model_param_groups[0], nn.Parameter):
        model_param_groups = [model_param_groups]

    model_params = [p for pg in model_param_groups for p in pg]
    for param, master_param in zip(
        model_params, unflatten_master_params(model_param_groups, master_params)
    ):
        param.detach().copy_(master_param)


def unflatten_master_params(model_param_groups, master_params):
    """
    Unflatten the master parameters to look like model_params.
    """
    if isinstance(model_param_groups[0], nn.Parameter):
        model_param_groups = [model_param_groups]

    return [p
            for mp, model_params in zip(master_params, model_param_groups)
            for p in _unflatten_dense_tensors(mp.detach(), model_params)]


def zero_grad(model_param_groups):
    if isinstance(model_param_groups[0], nn.Parameter):
        model_param_groups = [model_param_groups]

    for model_params in model_param_groups:
        for param in model_params:
            # Taken from https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.add_param_group
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()
