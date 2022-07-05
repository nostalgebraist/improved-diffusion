from abc import abstractmethod
from functools import lru_cache

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from axial_positional_embedding import AxialPositionalEmbedding
from x_transformers.x_transformers import Rezero

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    silu,
    adagn_silu,
    adagn_silu_extended_32_8,
    adagn_silu_extended_32_6,
    adagn_silu_extended_32_1,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    checkpoint,
    expanded_timestep_embedding,
    scale_module
)

from .text_nn import TextEncoder, CrossAttention, WeaveAttention

import clip
from transformer_utils.partial_forward import partial_forward

def clip_encode_text_nopool(token_embedding, positional_embedding, transformer, toks, dtype=th.float32, out_format='nld',
                            ln_final=None,
                            use_penultimate_layer=False):
    clip_dtype = transformer.resblocks[0].attn.out_proj.weight.dtype
    x = token_embedding(toks).type(clip_dtype)  # [batch_size, n_ctx, d_model]

    x = x + positional_embedding.type(clip_dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND

    if use_penultimate_layer:
        # from imagen paper - note that cos sims between layer outputs get way lower in the final one
        out_name = 'resblocks.' + str(len(transformer.resblocks) - 2)
        x = partial_forward(transformer, [out_name], x)[out_name]
    else:
        x = transformer(x)

    if ln_final is not None:
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x.type(dtype)
        x = ln_final(x)
        if out_format == 'ndl':
            x = x.permute(0, 2, 1)  # NLD -> NDL
    else:
        if out_format == 'nld':
            x = x.permute(1, 0, 2)  # LND -> NLD
        elif out_format == 'ndl':
            x = x.permute(1, 2, 0)  # LND -> NDL
        else:
            raise ValueError(out_format)
    # x = ln_final(x)
    x = x.type(dtype)

    # x.shape = [batch_size, n_ctx, transformer.width]

    return x


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TextTimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb, txt, capt, attn_mask=None, tgt_pos_embs=None, capt_attn_mask=None, ):
        """
        Apply the module to `x` given `txt` texts.
        """


class GlideStyleBlock(nn.Module):
    @abstractmethod
    def forward(self, x, capt, capt_attn_mask):
        """
        Hack to let AttentionBlock just add capt to its usual x without adding too much code
        """


class CrossAttentionAdapter(TextTimestepBlock):
    def __init__(self, *args, use_capt=False, **kwargs):
        super().__init__()
        self.cross_attn = CrossAttention(*args, **kwargs)
        self.use_capt = use_capt

    def forward(self, x, emb, txt, capt, attn_mask=None, tgt_pos_embs=None, timesteps=None, capt_attn_mask=None):
        if self.use_capt:
            src = capt
            attn_mask_ = capt_attn_mask
        else:
            src = txt
            attn_mask_ = attn_mask

        x, orig_text = self.cross_attn.forward(src=src, tgt=x, attn_mask=attn_mask_, tgt_pos_embs=tgt_pos_embs, timestep_emb=emb)
        return x, txt


class WeaveAttentionAdapter(TextTimestepBlock):
    def __init__(self, *args, use_capt=False, **kwargs):
        super().__init__()
        self.weave_attn = WeaveAttention(*args, **kwargs)
        self.use_capt = use_capt

    def forward(self, x, emb, txt, capt, attn_mask=None, tgt_pos_embs=None, timesteps=None, capt_attn_mask=None):
        if self.use_capt:
            text = capt
            attn_mask_ = capt_attn_mask
        else:
            text = txt
            attn_mask_ = attn_mask

        x, orig_text = self.weave_attn.forward(text=text, image=x, attn_mask=attn_mask_, tgt_pos_embs=tgt_pos_embs, timestep_emb=emb)
        return x, txt


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, inps, emb, attn_mask=None, tgt_pos_embs=None, timesteps=None, capt_attn_mask=None):
        x, txt, capt = inps
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, TextTimestepBlock):
                x, txt = layer(x, emb, txt, capt, attn_mask=attn_mask, tgt_pos_embs=tgt_pos_embs, capt_attn_mask=capt_attn_mask)
            elif isinstance(layer, GlideStyleBlock):
                x = layer(x, capt, capt_attn_mask)
            else:
                x = layer(x)
        return x, txt, capt


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, use_checkpoint_lowcost=False, mode='nearest'):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        self.mode = mode
        self.use_checkpoint = use_checkpoint_lowcost and not use_conv
        if use_conv:
            self.conv = conv_nd(dims, channels, channels, 3, padding=1)

    def forward(self, x):
        return checkpoint(
            self._forward, (x,), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode=self.mode
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode=self.mode)
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, use_checkpoint_lowcost=False, use_nearest=False):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        self.use_checkpoint = use_checkpoint_lowcost and not use_conv
        self.use_nearest = use_nearest
        stride = 2 if dims != 3 else (1, 2, 2)
        if self.use_nearest:
            self.op = None
        elif use_conv:
            self.op = conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        return checkpoint(
            self._forward, (x,), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x):
        assert x.shape[1] == self.channels
        if self.use_nearest:
            return F.interpolate(x, scale_factor=0.5, mode="nearest")
        return self.op(x)


class BreadAdapterIn(nn.Module):
    def __init__(self, in_channels, model_channels, use_nearest=False, dims=2, use_checkpoint=False):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        self.down = Downsample(in_channels, False, dims, use_nearest=use_nearest)
        self.transducer = conv_nd(dims, in_channels, model_channels, 3, padding=1)

    def forward(self, x):
        return checkpoint(
            self._forward, (x,), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x):
        return self.transducer(self.down(x))


class BreadAdapterOut(nn.Module):
    def __init__(self, model_channels, out_channels, dims=2, use_checkpoint=False, silu_impl="torch"):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        # self.down = Downsample(model_channels, False, dims)
        self.up = Upsample(out_channels, False, dims)
        self.transducer = nn.Sequential(
            normalization(model_channels, fused=silu_impl=="fused"),
            silu(impl=silu_impl),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def forward(self, x):
        return checkpoint(
            self._forward, (x,), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x):
        return self.up(self.transducer(x))
        # return self.up(self.transducer(self.down(x)))


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        use_checkpoint_lowcost=False,
        base_channels=None,
        silu_impl="torch"
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        if use_checkpoint:
            use_checkpoint_lowcost = False

        self.fused = silu_impl=="fused"

        if base_channels > 0:
            self.base_channels = base_channels
            self.base_out_channels = self.base_channels * self.out_channels // channels
        else:
            self.base_channels = base_channels
            self.base_out_channels = base_channels

        self.in_layers = nn.Sequential(
            normalization(channels, base_channels=self.base_channels, fused=self.fused),
            silu(impl=silu_impl, use_checkpoint=use_checkpoint_lowcost),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims, use_checkpoint_lowcost=use_checkpoint_lowcost)
            self.x_upd = Upsample(channels, False, dims, use_checkpoint_lowcost=use_checkpoint_lowcost)
        elif down:
            self.h_upd = Downsample(channels, False, dims, use_checkpoint_lowcost=use_checkpoint_lowcost)
            self.x_upd = Downsample(channels, False, dims, use_checkpoint_lowcost=use_checkpoint_lowcost)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            silu(impl="torch" if self.fused else silu_impl, use_checkpoint=use_checkpoint_lowcost),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels, base_channels=self.base_out_channels, fused=self.fused),
            silu(impl=silu_impl, use_checkpoint=use_checkpoint_lowcost),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            # print(f'x shape: {x.shape}')
            # print(f'self.in_layers[0].weight shape: {self.in_layers[0].weight.shape}')
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            if self.fused:
                if self.base_channels > 0:
                    # AdaGN: fused, extended
                    base, xtra = th.split(
                        emb_out,
                        [2 * self.base_out_channels, 2 * self.out_channels - 2 * self.base_out_channels],
                        dim=1
                    )
                    base_h, xtra_h = th.split(h, [out_norm.num_channels_base, out_norm.num_channels_xtra], dim=1)
                    if out_norm.num_groups_xtra == 8:
                        fn = adagn_silu_extended_32_8
                    elif out_norm.num_groups_xtra == 6:
                        fn = adagn_silu_extended_32_6
                    elif out_norm.num_groups_xtra == 1:
                        fn = adagn_silu_extended_32_1
                    else:
                        raise ValueError(out_norm.num_groups_xtra)
                    h = fn(
                        base_h, xtra_h,
                        base, xtra,
                        out_norm.weight, out_norm.bias,
                        out_norm.weight_xtra, out_norm.bias_xtra,
                    )
                else:
                    # AdaGN: fused, not extended
                    h = adagn_silu(h, emb_out, out_norm.weight, out_norm.bias)
            else:
                if self.base_channels > 0:
                    # AdaGN: not fused, extended
                    base, xtra = th.split(
                        emb_out,
                        [2 * self.base_out_channels, 2 * self.out_channels - 2 * self.base_out_channels],
                        dim=1
                    )
                    base_scale, base_shift = th.chunk(base, 2, dim=1)
                    xtra_scale, xtra_shift = th.chunk(xtra, 2, dim=1)
                    scale = th.cat([base_scale, xtra_scale], dim=1)
                    shift = th.cat([base_shift, xtra_shift], dim=1)
                    h = out_norm(h) * (1 + scale) + shift
                else:
                    # AdaGN: not fused, not extended
                    scale, shift = th.chunk(emb_out, 2, dim=1)
                    h = out_norm(h) * (1 + scale) + shift
            # AdaGN: any
            h = out_rest(h)
        else:
            # not AdaGN
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(GlideStyleBlock):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.

    """
    # TODO: (nost) - why not try AdaGN for norm here?
    def __init__(self, channels, num_heads=1, use_checkpoint=False, use_checkpoint_lowcost=False, base_channels=None,
                 encoder_channels=None):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint
        use_checkpoint_lowcost = use_checkpoint_lowcost and not use_checkpoint

        self.norm = normalization(channels, base_channels=base_channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

        if encoder_channels is not None:
            self.encoder_kv = conv_nd(1, encoder_channels, channels * 2, 1)
            self.encoder_norm = None # normalization(encoder_channels)
        else:
            self.encoder_kv = None
            self.encoder_norm = None

    def forward(self, x, encoder_out=None, capt_attn_mask=None):
        return checkpoint(self._forward, (x, encoder_out, capt_attn_mask), self.parameters(), self.use_checkpoint)

    def _forward(self, x, encoder_out=None, attn_mask=None):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        if (encoder_out is not None) and (self.encoder_kv is not None):
            # encoder_out = self.encoder_norm(encoder_out)
            encoder_kv = self.encoder_kv(encoder_out)
            encoder_kv = encoder_kv.reshape(b * self.num_heads, -1, encoder_kv.shape[2])

            my_attn_mask = th.tile(attn_mask.unsqueeze(1), (self.num_heads, qkv.shape[2], 1))
            # image_attn_mask_name = f"mask_{qkv.shape[0]}_{qkv.shape[2]}"
            # if not hasattr(self, image_attn_mask_name):
            #     self.register_buffer(image_attn_mask_name, th.ones((qkv.shape[0], qkv.shape[2], qkv.shape[2]), dtype=bool, device=qkv.device), persistent=False)
            #
            # image_attn_mask = getattr(self, image_attn_mask_name)
            #
            # my_attn_mask = th.cat([image_attn_mask, my_attn_mask], dim=2)

            my_attn_mask = th.cat([th.ones((qkv.shape[0], qkv.shape[2], qkv.shape[2]), dtype=bool, device=qkv.device), my_attn_mask], dim=2)
            my_attn_mask = (~my_attn_mask).to(encoder_kv.dtype) * -10000.

            h = self.attention(qkv, encoder_kv, my_attn_mask)
        else:
            h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def forward(self, qkv, encoder_kv=None, attn_mask=None):
        """
        Apply QKV attention.

        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = th.split(qkv, ch, dim=1)
        if encoder_kv is not None:
            ek, ev = encoder_kv.split(ch, dim=1)
            k = th.cat([ek, k], dim=-1)
            v = th.cat([ev, v], dim=-1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = weight.float()
        if attn_mask is not None:
            weight = weight + attn_mask
        weight = th.softmax(weight, dim=-1).type(weight.dtype)
        # if encoder_kv is not None:
        #     l_base = qkv.shape[2]
        #     print(f'l_base {l_base}')
        #     weight_on_capt = weight[:, :, l_base:].sum(dim=-1)
        #     wmin = weight_on_capt.min().item()
        #     wmean = weight_on_capt.mean().item()
        #     wmax = weight_on_capt.max().item()
        #     print(f"weight_on_capt min {wmin:.3f} | mean {wmean:.3f} | max {wmax:.3f}")
        return th.einsum("bts,bcs->bct", weight, v)

    @staticmethod
    def count_flops(model, _x, y):
        """
        A counter for the `thop` package to count the operations in an
        attention operation.

        Meant to be used like:

            macs, params = thop.profile(
                model,
                inputs=(inputs, timestamps),
                custom_ops={QKVAttention: QKVAttention.count_flops},
            )

        """
        b, c, *spatial = y[0].shape
        num_spatial = int(np.prod(spatial))
        # We perform two matmuls with the same number of ops.
        # The first computes the weight matrix, the second computes
        # the combination of the value vectors.
        matmul_ops = 2 * b * (num_spatial ** 2) * c
        model.total_ops += th.DoubleTensor([matmul_ops])


class MonochromeAdapter(nn.Module):
    def __init__(self, to_mono=True, needs_var=False):
        super().__init__()
        dims = (3, 1) if to_mono else (1, 3)
        w_init = 1/3. if to_mono else 1.

        self.linear_mean = nn.Linear(*dims)
        nn.init.constant_(self.linear_mean.weight, w_init)
        nn.init.constant_(self.linear_mean.bias, 0.)

        self.needs_var = needs_var
        if needs_var:
            self.linear_var = nn.Linear(*dims)
            nn.init.constant_(self.linear_var.weight, w_init)
            nn.init.constant_(self.linear_var.bias, 0.)

    def forward(self, x):
        segs = th.split(x, 3, dim=1)
        out = self.linear_mean(segs[0].transpose(1, 3))
        if self.needs_var and len(segs) > 1:
            out_var = self.linear_var(segs[1].transpose(1, 3))
            out = th.cat([out, out_var], dim=3)
        out = out.transpose(1, 3)
        return out


class DropinRGBAdapter(nn.Module):
    def __init__(self, needs_var=False, scale=1.0e0, diag_w=0.5):
        super().__init__()
        self.scale = scale
        dims = (3, 3)
        w_init = diag_w * th.eye(3) + (1 - diag_w) * (1/3.) * th.ones((3, 3))
        w_init = w_init / self.scale

        self.linear_mean_w = nn.Parameter(w_init)
        self.linear_mean_b = nn.Parameter(th.zeros((3,)))
        # self.linear_mean = nn.Linear(*dims)
        # nn.init.constant_(self.linear_mean.weight, w_init)
        # nn.init.constant_(self.linear_mean.bias, 0.)

        self.needs_var = needs_var
        if needs_var:
            self.linear_var_w = nn.Parameter(w_init)
            self.linear_var_b = nn.Parameter(th.zeros((3,)))
            # self.linear_var = nn.Linear(*dims)
            # nn.init.constant_(self.linear_var.weight, w_init)
            # nn.init.constant_(self.linear_var.bias, 0.)

    def forward(self, x):
        segs = th.split(x, 3, dim=1)
        # out = self.linear_mean(segs[0].transpose(1, 3))
        out = F.linear(
            segs[0].transpose(1, 3),
            self.scale * self.linear_mean_w,
            self.linear_mean_b
        )
        if self.needs_var and len(segs) > 1:
            # out_var = self.linear_var(segs[1].transpose(1, 3))
            out_var = F.linear(
                segs[0].transpose(1, 3),
                self.scale * self.linear_var_w,
                self.linear_var_b
            )
            out = th.cat([out, out_var], dim=3)
        out = out.transpose(1, 3)
        return out


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_checkpoint_up=False,
        use_checkpoint_middle=False,
        use_checkpoint_down=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        channels_per_head=0,
        channels_per_head_upsample=-1,
        txt=False,
        txt_dim=128,
        txt_depth=2,
        max_seq_len=64,
        txt_resolutions=(8,),
        cross_attn_channels_per_head=-1,
        cross_attn_init_gain=1.,
        cross_attn_gain_scale=200,
        image_size=None,
        text_lr_mult=-1.,
        txt_output_layers_only=False,
        monochrome_adapter=False,
        txt_attn_before_attn=False,
        txt_avoid_groupnorm=False,
        cross_attn_orth_init=False,
        cross_attn_q_t_emb=False,
        txt_rezero=False,
        txt_ff_glu=False,
        txt_ff_mult=4,
        cross_attn_rezero=False,
        cross_attn_rezero_keeps_prenorm=False,
        cross_attn_use_layerscale=False,
        tokenizer=None,
        verbose=False,
        txt_t5=False,
        txt_rotary=False,
        colorize=False,
        rgb_adapter=False,
        weave_attn=False,
        weave_use_ff=True,
        weave_ff_rezero=True,
        weave_ff_force_prenorm=False,
        weave_ff_mult=4,
        weave_ff_glu=False,
        weave_qkv_dim_always_text=False,
        channels_last_mem=False,
        up_interp_mode="bilinear",
        weave_v2=False,
        use_checkpoint_lowcost=False,
        weave_use_ff_gain=False,
        bread_adapter_at_ds=-1,
        bread_adapter_only=False,
        bread_adapter_nearest_in=False,
        bread_adapter_zero_conv_in=False,
        expand_timestep_base_dim=-1,
        silu_impl="torch",
        using_capt=False,
        xattn_capt=True,
        weave_capt=True,
        glide_style_capt_attn=False,
        glide_style_capt_emb=False,
        glide_style_capt_emb_init_scale=0.1,
        glide_style_capt_emb_nonlin=False,
        label_emb_init_scale=0.,
        clipname='RN50',
        clip_use_penultimate_layer=False,
        use_checkpoint_below_res=-1,
        no_attn=False,
        no_attn_substitute_resblock=False,
        noise_cond=False,
        freeze_capt_encoder=False,
        use_inference_caching=False,
        clipmod=None,
    ):
        super().__init__()

        print(f"unet: got txt={txt}, text_lr_mult={text_lr_mult}, txt_output_layers_only={txt_output_layers_only}, colorize={colorize} | weave_attn {weave_attn} | up_interp_mode={up_interp_mode} | weave_v2={weave_v2}")

        if text_lr_mult < 0:
            text_lr_mult = None

        print(f"unet: have text_lr_mult={text_lr_mult}")
        print(f"unet: got use_scale_shift_norm={use_scale_shift_norm}, resblock_updown={resblock_updown}")
        print(f"unet: got use_checkpoint={use_checkpoint}, use_checkpoint_up={use_checkpoint_up}, use_checkpoint_middle={use_checkpoint_middle}, use_checkpoint_down={use_checkpoint_down}, use_checkpoint_lowcost={use_checkpoint_lowcost}")
        print(f"unet: have noise_cond={noise_cond}, freeze_capt_encoder={freeze_capt_encoder}")

        def vprint(*args):
            if verbose:
                print(*args)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if channels_per_head_upsample == -1:
            channels_per_head_upsample = channels_per_head

        use_attn = not no_attn
        if use_attn:
            no_attn_substitute_resblock = False

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample

        self.txt = txt
        self.txt_resolutions = txt_resolutions
        self.image_size = image_size

        self.using_capt = using_capt
        self.xattn_capt = xattn_capt
        self.glide_style_capt_attn = glide_style_capt_attn
        self.glide_style_capt_emb = glide_style_capt_emb
        self.clip_use_penultimate_layer = clip_use_penultimate_layer
        self.use_inference_caching = use_inference_caching

        self.noise_cond = noise_cond

        # if use_checkpoint_below_res < 0:
        #     use_checkpoint_below_res = self.image_size * 2
        self.use_checkpoint_below_res = use_checkpoint_below_res
        print(("self.use_checkpoint_below_res", self.use_checkpoint_below_res))

        if monochrome_adapter and rgb_adapter:
            print("using both monochrome_adapter and rgb_adapter, make sure this is intentional!")
        self.monochrome_adapter = monochrome_adapter
        self.rgb_adapter = rgb_adapter
        self.colorize = colorize
        self.channels_last_mem = channels_last_mem
        self.up_interp_mode = up_interp_mode
        self.expand_timestep_base_dim = expand_timestep_base_dim

        self.cached_timestep_embs = None

        if self.txt:
            self.text_encoder = TextEncoder(
                inner_dim=txt_dim,
                depth=txt_depth,
                max_seq_len=max_seq_len,
                lr_mult=text_lr_mult,
                use_rezero=txt_rezero,
                use_scalenorm=not txt_rezero,
                tokenizer=tokenizer,
                rel_pos_bias=txt_t5,
                rotary_pos_emb=txt_rotary,
                ff_glu=txt_ff_glu,
                ff_mult=txt_ff_mult,
                use_checkpoint=use_checkpoint,
                silu_impl=silu_impl
            )

        if self.using_capt:
            if clipmod is None:
                clipmod, _ = clip.load(name=clipname)
                del clipmod.visual
            if not freeze_capt_encoder:
                clipmod.float()
            self.clipmod = clipmod
            self.clipmod.positional_embedding = clipmod.positional_embedding
            self.clipmod.transformer = clipmod.transformer
            self.capt_embd_dim = clipmod.ln_final.weight.shape[0]

            if self.clip_use_penultimate_layer:
                self.capt_ln_final = nn.LayerNorm(self.capt_embd_dim)
            else:
                self.capt_ln_final = clipmod.ln_final

        self.tgt_pos_embs = nn.ModuleDict({})

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            silu(impl="torch" if silu_impl == "fused" else silu_impl, use_checkpoint=use_checkpoint_lowcost),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.noise_cond:
            self.time_embed_noise_cond = nn.Sequential(
                linear(model_channels, time_embed_dim),
                silu(impl="torch" if silu_impl == "fused" else silu_impl, use_checkpoint=use_checkpoint_lowcost),
                linear(time_embed_dim, time_embed_dim),
            )
        else:
            self.time_embed_noise_cond = None

        if self.glide_style_capt_emb:
            if glide_style_capt_emb_nonlin:
                self.capt_embed = nn.Sequential(
                    linear(self.capt_embd_dim, time_embed_dim),
                    silu(impl="torch" if silu_impl == "fused" else silu_impl, use_checkpoint=use_checkpoint_lowcost),
                    scale_module(linear(time_embed_dim, time_embed_dim), glide_style_capt_emb_init_scale)
                )
            else:
                self.capt_embed = scale_module(linear(self.capt_embd_dim, time_embed_dim), glide_style_capt_emb_init_scale)

        if self.num_classes is not None:
            self.label_emb = scale_module(nn.Embedding(num_classes, time_embed_dim), label_emb_init_scale)

        if monochrome_adapter:
            self.mono_to_rgb = MonochromeAdapter(to_mono=False, needs_var=False)

        if rgb_adapter:
            self.rgb_to_input = DropinRGBAdapter(needs_var=False)

        self.using_bread_adapter = bread_adapter_at_ds >= 1
        bread_adapter_in_added = False
        bread_adapter_out_added = False
        self.bread_adapter_only = bread_adapter_only
        print(f'unet self.bread_adapter_only: {self.bread_adapter_only}')

        mapper = lambda x: x
        if self.using_bread_adapter and bread_adapter_zero_conv_in:
            mapper = zero_module
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    mapper(conv_nd(dims, in_channels, model_channels, 3, padding=1, use_checkpoint=image_size <= use_checkpoint_below_res))
                )
            ]
        )

        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint or use_checkpoint_down or ((image_size // ds) <= use_checkpoint_below_res),
                        use_scale_shift_norm=use_scale_shift_norm,
                        use_checkpoint_lowcost=use_checkpoint_lowcost,
                        base_channels=expand_timestep_base_dim * ch // model_channels,
                        silu_impl=silu_impl,
                    )
                ]
                ch = mult * model_channels
                if (ds in attention_resolutions):
                    if no_attn_substitute_resblock:
                        layers.append(
                            ResBlock(
                                ch,
                                time_embed_dim,
                                dropout,
                                out_channels=mult * model_channels,
                                dims=dims,
                                use_checkpoint=use_checkpoint or use_checkpoint_down or ((image_size // ds) <= use_checkpoint_below_res),
                                use_scale_shift_norm=use_scale_shift_norm,
                                use_checkpoint_lowcost=use_checkpoint_lowcost,
                                base_channels=expand_timestep_base_dim * ch // model_channels,
                                silu_impl=silu_impl,
                            )
                        )
                    elif use_attn:
                        num_heads_here = num_heads
                        if channels_per_head > 0:
                            num_heads_here = ch // channels_per_head
                        layers.append(
                            AttentionBlock(
                                ch, use_checkpoint=use_checkpoint or use_checkpoint_down or ((image_size // ds) <= use_checkpoint_below_res),
                                num_heads=num_heads_here,
                                use_checkpoint_lowcost=use_checkpoint_lowcost,
                                base_channels=expand_timestep_base_dim * ch // model_channels,
                                encoder_channels=self.capt_embd_dim if self.glide_style_capt_attn else None
                            )
                        )
                    else:
                        layers.append(nn.Identity())
                if self.txt and ds in self.txt_resolutions and (not txt_output_layers_only):
                    num_heads_here = num_heads
                    if cross_attn_channels_per_head > 0:
                        num_heads_here = txt_dim // cross_attn_channels_per_head

                    emb_res = image_size // ds
                    if emb_res not in self.tgt_pos_embs:
                        pos_emb_dim = ch
                        # pos emb in AdaGN
                        if (not txt_avoid_groupnorm) and cross_attn_q_t_emb:
                            pos_emb_dim *= 2
                        self.tgt_pos_embs[str(emb_res)] = AxialPositionalEmbedding(
                            dim=pos_emb_dim,
                            axial_shape=(emb_res, emb_res),
                        )
                    caa_args = dict(
                        use_checkpoint=use_checkpoint or use_checkpoint_down,
                        dim=ch,
                        time_embed_dim=time_embed_dim,
                        heads=num_heads_here,
                        text_dim=txt_dim,
                        emb_res = image_size // ds,
                        init_gain = cross_attn_init_gain,
                        gain_scale = cross_attn_gain_scale,
                        lr_mult=text_lr_mult,
                        needs_tgt_pos_emb=False,
                        avoid_groupnorm=txt_avoid_groupnorm,
                        orth_init=cross_attn_orth_init,
                        q_t_emb=cross_attn_q_t_emb,
                        use_rezero=cross_attn_rezero,
                        rezero_keeps_prenorm=cross_attn_rezero_keeps_prenorm,
                        use_layerscale=cross_attn_use_layerscale,
                        image_base_channels=expand_timestep_base_dim * ch // model_channels,
                        silu_impl=silu_impl,
                    )
                    if weave_attn:
                        caa_args['image_dim'] = caa_args.pop('dim')
                        caa_args.update(dict(
                            use_ff=weave_use_ff,
                            ff_rezero=weave_ff_rezero,
                            ff_force_prenorm=weave_ff_force_prenorm,
                            ff_mult=weave_ff_mult,
                            ff_glu=weave_ff_glu,
                            qkv_dim_always_text=weave_qkv_dim_always_text,
                            weave_v2=weave_v2,
                            use_ff_gain=weave_use_ff_gain,
                        ))
                        caa = WeaveAttentionAdapter(**caa_args) if use_attn else nn.Identity()
                    else:
                        caa = CrossAttentionAdapter(**caa_args) if use_attn else nn.Identity()
                    if txt_attn_before_attn and (ds in attention_resolutions):
                        layers.insert(-1, caa)
                    else:
                        layers.append(caa)


                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
                vprint(f"up   | {level} of {len(channel_mult)} | ch {ch} | ds {ds}")
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint or use_checkpoint_down or ((image_size // ds) <= use_checkpoint_below_res),
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            use_checkpoint_lowcost=use_checkpoint_lowcost,
                            base_channels=expand_timestep_base_dim * ch // model_channels,
                            silu_impl=silu_impl,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims
                        )
                    )
                )
                input_block_chans.append(ch)
                ds *= 2
                vprint(f"up   | ds {ds // 2} -> {ds}")

                if (bread_adapter_at_ds == ds) and (not bread_adapter_in_added):
                    vprint(f"adding bread_adapter_in at {ds}")
                    self.bread_adapter_in = BreadAdapterIn(in_channels=self.in_channels, model_channels=ch,
                                                           use_nearest=bread_adapter_nearest_in)
                    bread_adapter_in_added = True
                    self.input_blocks[-1].bread_adapter_in_pt = True

        vprint(f"input_block_chans: {input_block_chans}")

        def _middle_resblock():
            return ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint or use_checkpoint_middle or ((image_size // ds) <= use_checkpoint_below_res),
                use_scale_shift_norm=use_scale_shift_norm,
                use_checkpoint_lowcost=use_checkpoint_lowcost,
                base_channels=expand_timestep_base_dim * ch // model_channels,
                silu_impl=silu_impl,
            )

        def _middle_attnblock():
            return AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint or use_checkpoint_middle or ((image_size // ds) <= use_checkpoint_below_res),
                num_heads=num_heads,
                use_checkpoint_lowcost=use_checkpoint_lowcost,
                base_channels=expand_timestep_base_dim * ch // model_channels,
                encoder_channels=self.capt_embd_dim if self.glide_style_capt_attn else None
            )

        self.middle_block = TimestepEmbedSequential(
            _middle_resblock(),
            _middle_resblock() if no_attn_substitute_resblock else (_middle_attnblock() if use_attn else nn.Identity()),
            _middle_resblock(),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                this_ch = ch + input_block_chans.pop()
                layers = [
                    ResBlock(
                        this_ch,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint or use_checkpoint_up or ((image_size // ds) <= use_checkpoint_below_res),
                        use_scale_shift_norm=use_scale_shift_norm,
                        use_checkpoint_lowcost=use_checkpoint_lowcost,
                        base_channels=expand_timestep_base_dim * this_ch // model_channels,
                        silu_impl=silu_impl,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if no_attn_substitute_resblock:
                        layers.append(
                            ResBlock(
                                ch,
                                time_embed_dim,
                                dropout,
                                out_channels=model_channels * mult,
                                dims=dims,
                                use_checkpoint=use_checkpoint or use_checkpoint_up or ((image_size // ds) <= use_checkpoint_below_res),
                                use_scale_shift_norm=use_scale_shift_norm,
                                use_checkpoint_lowcost=use_checkpoint_lowcost,
                                base_channels=expand_timestep_base_dim * this_ch // model_channels,
                                silu_impl=silu_impl,
                            )
                        )
                    elif use_attn:
                        num_heads_here = num_heads_upsample
                        if channels_per_head_upsample > 0:
                            num_heads_here = ch // channels_per_head_upsample
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint or use_checkpoint_up or ((image_size // ds) <= use_checkpoint_below_res),
                                num_heads=num_heads_here,
                                use_checkpoint_lowcost=use_checkpoint_lowcost,
                                base_channels=expand_timestep_base_dim * ch // model_channels,
                                encoder_channels=self.capt_embd_dim if self.glide_style_capt_attn else None
                            )
                        )
                    else:
                        layers.append(nn.Identity())
                if self.txt and ds in self.txt_resolutions:
                    use_capts = [False, True] if (self.using_capt and self.xattn_capt) else [False]
                    if self.glide_style_capt_attn:
                        use_capts = [False]
                    for use_capt in use_capts:
                        num_heads_here = num_heads
                        if cross_attn_channels_per_head > 0:
                            num_heads_here = txt_dim // cross_attn_channels_per_head

                        emb_res = image_size // ds
                        if emb_res not in self.tgt_pos_embs:
                            pos_emb_dim = ch
                            # pos emb in AdaGN
                            if (not txt_avoid_groupnorm) and cross_attn_q_t_emb:
                                pos_emb_dim *= 2
                            self.tgt_pos_embs[str(emb_res)] = AxialPositionalEmbedding(
                                dim=pos_emb_dim,
                                axial_shape=(emb_res, emb_res),
                            )
                        caa_args = dict(
                            use_checkpoint=use_checkpoint or use_checkpoint_up,
                            dim=ch,
                            time_embed_dim=time_embed_dim,
                            heads=num_heads_here,
                            text_dim=self.capt_embd_dim if use_capt else txt_dim,
                            emb_res = emb_res,
                            init_gain = cross_attn_init_gain,
                            gain_scale = cross_attn_gain_scale,
                            lr_mult=text_lr_mult,
                            needs_tgt_pos_emb=False,
                            avoid_groupnorm=txt_avoid_groupnorm,
                            orth_init=cross_attn_orth_init,
                            q_t_emb=cross_attn_q_t_emb,
                            use_rezero=cross_attn_rezero,
                            rezero_keeps_prenorm=cross_attn_rezero_keeps_prenorm,
                            use_layerscale=cross_attn_use_layerscale,
                            image_base_channels=expand_timestep_base_dim * ch // model_channels,
                            silu_impl=silu_impl,
                            use_capt=use_capt,
                            txt_already_normed=use_capt and (self.clipmod.ln_final is not None)
                        )
                        if weave_attn:
                            caa_args['image_dim'] = caa_args.pop('dim')
                            caa_args.update(dict(
                                use_ff=weave_use_ff,
                                ff_rezero=weave_ff_rezero,
                                ff_force_prenorm=weave_ff_force_prenorm or use_capt, # ! testing this out
                                ff_mult=weave_ff_mult,
                                ff_glu=weave_ff_glu,
                                qkv_dim_always_text=weave_qkv_dim_always_text,
                                weave_v2=weave_v2,
                                use_ff_gain=weave_use_ff_gain,
                                no_itot=use_capt and (not weave_capt),
                            ))
                            caa = WeaveAttentionAdapter(**caa_args) if use_attn else nn.Identity()
                        else:
                            caa = CrossAttentionAdapter(**caa_args) if use_attn else nn.Identity()
                        if txt_attn_before_attn and (ds in attention_resolutions):
                            layers.insert(-1, caa)
                        else:
                            layers.append(caa)
                vprint(f"down | {level} of {len(channel_mult)} | ch {ch} | ds {ds}")
                if level and i == num_res_blocks:
                    if (bread_adapter_at_ds == ds) and (not bread_adapter_out_added):
                        vprint(f"adding bread_adapter_out at {ds}")
                        self.bread_adapter_out = BreadAdapterOut(silu_impl=silu_impl, out_channels=out_channels, model_channels=ch)
                        bread_adapter_out_added = True
                        self.output_blocks.append(TimestepEmbedSequential(*layers))
                        layers = []
                        self.output_blocks[-1].bread_adapter_out_pt = True

                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint or use_checkpoint_up or ((image_size // ds) <= use_checkpoint_below_res),
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            use_checkpoint_lowcost=use_checkpoint_lowcost,
                            base_channels=expand_timestep_base_dim * ch // model_channels,
                            silu_impl=silu_impl,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims)
                    )
                    ds //= 2
                    vprint(f"down | ds {ds * 2} -> {ds}")
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                # if (bread_adapter_at_ds == (ds*2)) and (not bread_adapter_out_added):
                #     vprint(f"adding bread_adapter_out at {ds*2}")
                #     self.bread_adapter_out = BreadAdapterOut(out_channels=out_channels, model_channels=ch)
                #     bread_adapter_out_added = True
                #     self.output_blocks[-1].bread_adapter_out_pt = True

        self.out = nn.Sequential(
            normalization(ch, base_channels=self.expand_timestep_base_dim, fused=silu_impl=="fused"),
            silu(impl=silu_impl, use_checkpoint=use_checkpoint_lowcost),
            zero_module(conv_nd(dims, channel_mult[0] * model_channels, out_channels, 3, padding=1)),
        )

        if monochrome_adapter:
            self.rgb_to_mono = MonochromeAdapter(to_mono=True, needs_var=out_channels>3)

        if rgb_adapter:
            self.output_to_rgb = DropinRGBAdapter(needs_var=out_channels>3)

    def timestep_embedding(self, timesteps):
        if self.expand_timestep_base_dim > 0:
            return expanded_timestep_embedding(timesteps, self.model_channels, self.expand_timestep_base_dim)
        return timestep_embedding(timesteps, self.model_channels)

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

        if self.channels_last_mem:
            self.input_blocks.to(memory_format=th.channels_last)
            self.middle_block.to(memory_format=th.channels_last)
            self.output_blocks.to(memory_format=th.channels_last)

        # if hasattr(self, 'text_encoder'):
        #     self.text_encoder.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)
        # if hasattr(self, 'text_encoder'):
        #     self.text_encoder.apply(convert_module_to_f32)

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    @lru_cache(2)
    def embed_capt_cached(self, capt_toks):
        return self.embed_capt(capt_toks)

    def embed_capt(self, capt_toks):
        capt_attn_mask = capt_toks != 0
        capt = clip_encode_text_nopool(
            self.clipmod.token_embedding, self.clipmod.positional_embedding, self.clipmod.transformer,
            capt_toks,
            ln_final=self.capt_ln_final if self.glide_style_capt_attn else None,
            use_penultimate_layer=self.clip_use_penultimate_layer,
            out_format='ndl' if self.glide_style_capt_attn else 'nld',
            dtype = self.inner_dtype,
            )
        return capt, capt_attn_mask

    def timestep_embed_with_cache(self, timesteps, save_to_cache=False):
        if self.cached_timestep_embs is not None:
            return self.cached_timestep_embs[timesteps]

        emb = self.time_embed(self.timestep_embedding(timesteps))

        if save_to_cache:
            self.cached_timestep_embs = emb

        return emb

    def unset_timestep_embed_cache(self):
        self.cached_timestep_embs = None
        if hasattr(self, 'text_encoder'):
            self.text_encoder.cached_timestep_embs = None

    def forward(self, x, timesteps, y=None, txt=None, capt=None, cond_timesteps=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # print(f"forward: txt passed = {txt is not None}, model txt = {self.txt}")
        # if isinstance(txt, dict):
        #     capt = txt['capt']
        #     txt = txt['txt']
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        assert (txt is not None) == (
            self.txt
        ), "must specify txt if and only if the model is text-conditional"

        assert (cond_timesteps is not None) == (
            self.noise_cond
        ), "must specify noise_cond if and only if the model uses noise cond"


        hs = []
        # emb = self.time_embed(self.timestep_embedding(timesteps))
        emb = self.timestep_embed_with_cache(timesteps)

        if cond_timesteps is not None:
            emb = emb + self.time_embed_noise_cond(self.timestep_embedding(cond_timesteps))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        attn_mask = None
        if txt is not None:
            txt, attn_mask = self.text_encoder(txt, timesteps=timesteps)
            txt = txt.type(self.inner_dtype)

        capt_attn_mask = None
        if self.using_capt and capt is not None:
            capt, capt_attn_mask = self.embed_capt_cached(capt) if self.use_inference_caching else self.embed_capt(capt)
            # capt_toks = capt
            # capt_attn_mask = capt_toks != 0
            # capt = clip_encode_text_nopool(
            #     self.clipmod.token_embedding, self.clipmod.positional_embedding, self.clipmod.transformer,
            #     capt_toks,
            #     ln_final=self.capt_ln_final if self.glide_style_capt_attn else None,
            #     use_penultimate_layer=self.clip_use_penultimate_layer,
            #     out_format='ndl' if self.glide_style_capt_attn else 'nld',
            #     dtype = self.inner_dtype,
            #     )
            # capt = capt.type(self.inner_dtype)
            if self.glide_style_capt_emb:
                eos = capt[th.arange(capt_toks.shape[0]), :, capt_toks.argmax(dim=-1)]
                emb = emb + self.capt_embed(eos)

        h = x

        if self.monochrome_adapter:
            h = self.mono_to_rgb(h)
        if self.rgb_adapter:
            h = self.rgb_to_input(h)

        # print(f'x type: {x.dtype}')
        # h = h.type(self.inner_dtype)
        # print(f'h type: {h.dtype}')
        if self.channels_last_mem:
            h = h.to(memory_format=th.channels_last)
        if self.using_bread_adapter:
            h_bread_in = self.bread_adapter_in(h)
            h_bread_out = None
        for module in self.input_blocks:
            # print(module)
            # print(f"h: {h.shape} | hs: {[t.shape for t in hs]}")
            h, txt, capt = module((h, txt, capt), emb, attn_mask=attn_mask, tgt_pos_embs=self.tgt_pos_embs, capt_attn_mask=capt_attn_mask)
            if getattr(module, 'bread_adapter_in_pt', False):
                if self.bread_adapter_only:
                    h = h_bread_in
                else:
                    h = h + h_bread_in
            hs.append(h)
            # print(f'\th type: {h.dtype}')
        h, txt, capt = self.middle_block((h, txt, capt), emb, attn_mask=attn_mask, tgt_pos_embs=self.tgt_pos_embs, capt_attn_mask=capt_attn_mask)
        skip_pop = False
        for module in self.output_blocks:
            # print(module)
            # print(f"h: {h.shape} | hs: {[t.shape for t in hs]}")
            if skip_pop:
                cat_in = h
                skip_pop = False
            else:
                if self.expand_timestep_base_dim > 0:
                    ch = h.shape[1]
                    mult = ch // self.model_channels
                    h_base, h_xtra = th.split(
                        h,
                        [mult * self.expand_timestep_base_dim, ch - mult * self.expand_timestep_base_dim],
                        dim=1
                    )

                    popped = hs.pop()
                    ch = popped.shape[1]
                    mult = ch // self.model_channels
                    popped_base, popped_xtra = th.split(
                        popped,
                        [mult * self.expand_timestep_base_dim, ch - mult * self.expand_timestep_base_dim],
                        dim=1
                    )

                    cat_in = th.cat([h_base, popped_base, h_xtra, popped_xtra], dim=1)
                else:
                    cat_in = th.cat([h, hs.pop()], dim=1)
            h, txt, capt = module((cat_in, txt, capt), emb, attn_mask=attn_mask, tgt_pos_embs=self.tgt_pos_embs, capt_attn_mask=capt_attn_mask)
            if getattr(module, 'bread_adapter_out_pt', False):
                h_bread_out = self.bread_adapter_out(h)
                skip_pop = True
            # print(f'\th type: {h.dtype}')
        # h = h.type(x.dtype)
        # print(f'h type: {h.dtype}')

        h = checkpoint(self.out.forward, (h,), self.out.parameters(), self.image_size <= self.use_checkpoint_below_res)

        # print(f'\th type: {h.dtype}')

        if self.using_bread_adapter:
            if self.bread_adapter_only:
                h = h_bread_out
            else:
                h = h + h_bread_out

        if self.rgb_adapter:
            h = self.output_to_rgb(h)
        if self.monochrome_adapter:
            h = self.rgb_to_mono(h)

        return h

    def get_feature_vectors(self, x, timesteps, y=None):
        """
        Apply the model and return all of the intermediate tensors.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: a dict with the following keys:
                 - 'down': a list of hidden state tensors from downsampling.
                 - 'middle': the tensor of the output of the lowest-resolution
                             block in the model.
                 - 'up': a list of hidden state tensors from upsampling.
        """
        hs = []
        emb = self.time_embed(self.timestep_embedding(timesteps))
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        result = dict(down=[], up=[])
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            result["down"].append(h.type(x.dtype))
        h = self.middle_block(h, emb)
        result["middle"] = h.type(x.dtype)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
            result["up"].append(h.type(x.dtype))
        return result


class SuperResModel(UNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(in_channels + 1 if kwargs.get('colorize') else in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode=self.up_interp_mode)
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)

    def get_feature_vectors(self, x, timesteps, low_res=None, **kwargs):
        _, new_height, new_width, _ = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode=self.up_interp_mode)
        x = th.cat([x, upsampled], dim=1)
        return super().get_feature_vectors(x, timesteps, **kwargs)
