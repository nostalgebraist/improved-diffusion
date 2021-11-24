"""
Helpers to train with 16-bit precision.
"""

import torch
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from .text_nn import TextEncoder, CrossAttention


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
    if isinstance(l, (CrossAttention, TextEncoder)):
        for n, p in l.named_parameters():
            if 'tgt_ln' in n and (not l.avoid_groupnorm):
                if 'normalization' not in n.partition('tgt_ln')[2]:
                    continue
            if 'src_ln' in n:
                continue
            p.data = p.data.to(dtype)


def convert_module_to_f32(l):
    """
    Convert primitive modules to float32, undoing convert_module_to_f16().
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d, )):
        l.weight.data = l.weight.data.float()
        if l.bias is not None:
            l.bias.data = l.bias.data.float()
    if isinstance(l, (CrossAttention, TextEncoder)):
        for n, p in l.named_parameters():
            p.data = p.data.float()


def make_master_params(model_params):
    """
    Copy model parameters into a (differently-shaped) list of full-precision
    parameters.
    """
    master_params = _flatten_dense_tensors(
        [param.detach().float() for param in model_params]
    )
    master_params = nn.Parameter(master_params)
    master_params.requires_grad = True
    return [master_params]


def model_grads_to_master_grads(model_params, master_params):
    """
    Copy the gradients from the model parameters into the master parameters
    from make_master_params().
    """
    master_params[0].grad = _flatten_dense_tensors(
        [param.grad.data.detach().float()
         if param.grad is not None else None
         for param in model_params]
    )


def master_params_to_model_params(model_params, master_params):
    """
    Copy the master parameter data back into the model parameters.
    """
    # Without copying to a list, if a generator is passed, this will
    # silently not copy any parameters.
    model_params = list(model_params)

    for param, master_param in zip(
        model_params, unflatten_master_params(model_params, master_params)
    ):
        param.detach().copy_(master_param)


def unflatten_master_params(model_params, master_params):
    """
    Unflatten the master parameters to look like model_params.
    """
    return _unflatten_dense_tensors(master_params[0].detach(), model_params)


def zero_grad(model_params):
    for param in model_params:
        # Taken from https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.add_param_group
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()
