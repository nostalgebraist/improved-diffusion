from abc import abstractmethod

import math
import io
import sys

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from axial_positional_embedding import AxialPositionalEmbedding
from x_transformers.x_transformers import Rezero

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    SiLU,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    checkpoint,
)

from .text_nn import TextEncoder, CrossAttention, WeaveAttention

ORIG_EINSUM = th.einsum  # for deepspeed


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
    def forward(self, x, emb, txt, attn_mask=None, tgt_pos_embs=None):
        """
        Apply the module to `x` given `txt` texts.
        """


class CrossAttentionAdapter(TextTimestepBlock):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # if self.use_checkpoint:
        #     raise ValueError('grad ckpt for xattn not working yet')
        self.cross_attn = CrossAttention(*args, **kwargs)

    def forward(self, x, emb, txt, attn_mask=None, tgt_pos_embs=None, timesteps=None):
        return self.cross_attn.forward(src=txt, tgt=x, attn_mask=attn_mask, tgt_pos_embs=tgt_pos_embs, timestep_emb=emb)

    # def forward(self, x, emb, txt, attn_mask=None, tgt_pos_embs=None, timesteps=None):
    #     return checkpoint(self._forward, (x, emb, txt, attn_mask, tgt_pos_embs, timesteps), self.parameters(), self.use_checkpoint)
    #
    # def _forward(self, x, emb, txt, attn_mask=None, tgt_pos_embs=None, timesteps=None):
    #     return self.cross_attn.forward(src=txt, tgt=x, attn_mask=attn_mask, tgt_pos_embs=tgt_pos_embs, timestep_emb=emb)


class WeaveAttentionAdapter(TextTimestepBlock):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # if self.use_checkpoint:
        #     raise ValueError('grad ckpt for xattn not working yet')
        self.weave_attn = WeaveAttention(*args, **kwargs)

    def forward(self, x, emb, txt, attn_mask=None, tgt_pos_embs=None, timesteps=None):
        return self.weave_attn.forward(text=txt, image=x, attn_mask=attn_mask, tgt_pos_embs=tgt_pos_embs, timestep_emb=emb)

    # def forward(self, x, emb, txt, attn_mask=None, tgt_pos_embs=None, timesteps=None):
    #     return checkpoint(self._forward, (x, emb, txt, attn_mask, tgt_pos_embs, timesteps), self.parameters(), self.use_checkpoint)
    #
    # def _forward(self, x, emb, txt, attn_mask=None, tgt_pos_embs=None, timesteps=None):
    #     return self.weave_attn.forward(text=txt, image=x, attn_mask=attn_mask, tgt_pos_embs=tgt_pos_embs, timestep_emb=emb)


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, inps, emb, attn_mask=None, tgt_pos_embs=None, timesteps=None):
        x, txt = inps
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, TextTimestepBlock):
                x, txt = layer(x, emb, txt, attn_mask=attn_mask, tgt_pos_embs=tgt_pos_embs)
            else:
                x = layer(x)
        return x, txt


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, use_checkpoint_lowcost=False):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
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
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
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

    def __init__(self, channels, use_conv, dims=2, use_checkpoint_lowcost=False):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        self.use_checkpoint = use_checkpoint_lowcost and not use_conv
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        return checkpoint(
            self._forward, (x,), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


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

        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(use_checkpoint=use_checkpoint_lowcost),
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
            SiLU(use_checkpoint=use_checkpoint_lowcost),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(use_checkpoint=use_checkpoint_lowcost),
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
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, channels, num_heads=1, use_checkpoint=False, use_checkpoint_lowcost=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint
        use_checkpoint_lowcost = use_checkpoint_lowcost and not use_checkpoint

        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class FakeStream(io.IOBase):
    def write(self, *args, **kwargs): pass


FAKE_STREAM = FakeStream()


def einsum_deepspeed_safe(*args):
    # silence spam that gets print()ed every time deepspeed profiler runs on th.einsum 9_9
    #
    # cf. https://github.com/microsoft/DeepSpeed/blob/bea701a1fc87a13f26f9d2f97ff4fd779b5d8b77/deepspeed/profiling/flops_profiler/profiler.py#L719-L739
    real_stdout = sys.stdout
    sys.stdout = FAKE_STREAM

    # out = th.einsum(*args)

    # completely disable _einsum_flops_compute
    out = ORIG_EINSUM(*args)

    sys.stdout = real_stdout

    return out


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = th.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))


        weight = einsum_deepspeed_safe(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)

        return einsum_deepspeed_safe("bts,bcs->bct", weight, v)

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
        use_checkpoint_lowcost=False
    ):
        super().__init__()

        print(f"unet: got txt={txt}, text_lr_mult={text_lr_mult}, txt_output_layers_only={txt_output_layers_only}, colorize={colorize} | weave_attn {weave_attn} | up_interp_mode={up_interp_mode} | weave_v2={weave_v2}")

        if text_lr_mult < 0:
            text_lr_mult = None

        print(f"unet: have text_lr_mult={text_lr_mult}")
        print(f"unet: got use_scale_shift_norm={use_scale_shift_norm}, resblock_updown={resblock_updown}")
        print(f"unet: got use_checkpoint={use_checkpoint}, use_checkpoint_up={use_checkpoint_up}, use_checkpoint_middle={use_checkpoint_middle}, use_checkpoint_down={use_checkpoint_down}, use_checkpoint_lowcost={use_checkpoint_lowcost}")

        def vprint(*args):
            if verbose:
                print(*args)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if channels_per_head_upsample == -1:
            channels_per_head_upsample = channels_per_head

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

        if monochrome_adapter and rgb_adapter:
            print("using both monochrome_adapter and rgb_adapter, make sure this is intentional!")
        self.monochrome_adapter = monochrome_adapter
        self.rgb_adapter = rgb_adapter
        self.colorize = colorize
        self.channels_last_mem = channels_last_mem
        self.up_interp_mode = up_interp_mode

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
                use_checkpoint=use_checkpoint
            )

        self.tgt_pos_embs = nn.ModuleDict({})

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(use_checkpoint=use_checkpoint_lowcost),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        if monochrome_adapter:
            self.mono_to_rgb = MonochromeAdapter(to_mono=False, needs_var=False)

        if rgb_adapter:
            self.rgb_to_input = DropinRGBAdapter(needs_var=False)

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
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
                        use_checkpoint=use_checkpoint or use_checkpoint_down,
                        use_scale_shift_norm=use_scale_shift_norm,
                        use_checkpoint_lowcost=use_checkpoint_lowcost,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    num_heads_here = num_heads
                    if channels_per_head > 0:
                        num_heads_here = ch // channels_per_head
                    layers.append(
                        AttentionBlock(
                            ch, use_checkpoint=use_checkpoint or use_checkpoint_down, num_heads=num_heads_here,
                            use_checkpoint_lowcost=use_checkpoint_lowcost
                        )
                    )
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
                        ))
                        caa = WeaveAttentionAdapter(**caa_args)
                    else:
                        caa = CrossAttentionAdapter(**caa_args)
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
                            use_checkpoint=use_checkpoint or use_checkpoint_down,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            use_checkpoint_lowcost=use_checkpoint_lowcost,
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

        vprint(f"input_block_chans: {input_block_chans}")

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint or use_checkpoint_middle,
                use_scale_shift_norm=use_scale_shift_norm,
                use_checkpoint_lowcost=use_checkpoint_lowcost,
            ),
            AttentionBlock(ch, use_checkpoint=use_checkpoint or use_checkpoint_middle, num_heads=num_heads,
                           use_checkpoint_lowcost=use_checkpoint_lowcost),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint or use_checkpoint_middle,
                use_scale_shift_norm=use_scale_shift_norm,
                use_checkpoint_lowcost=use_checkpoint_lowcost,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint or use_checkpoint_up,
                        use_scale_shift_norm=use_scale_shift_norm,
                        use_checkpoint_lowcost=use_checkpoint_lowcost,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    num_heads_here = num_heads_upsample
                    if channels_per_head_upsample > 0:
                        num_heads_here = ch // channels_per_head_upsample
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint or use_checkpoint_up,
                            num_heads=num_heads_here,
                            use_checkpoint_lowcost=use_checkpoint_lowcost
                        )
                    )
                if self.txt and ds in self.txt_resolutions:
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
                        text_dim=txt_dim,
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
                        ))
                        caa = WeaveAttentionAdapter(**caa_args)
                    else:
                        caa = CrossAttentionAdapter(**caa_args)
                    if txt_attn_before_attn and (ds in attention_resolutions):
                        layers.insert(-1, caa)
                    else:
                        layers.append(caa)
                vprint(f"down | {level} of {len(channel_mult)} | ch {ch} | ds {ds}")
                if level and i == num_res_blocks:
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint or use_checkpoint_up,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            use_checkpoint_lowcost=use_checkpoint_lowcost,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims)
                    )
                    ds //= 2
                    vprint(f"down | ds {ds * 2} -> {ds}")
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(use_checkpoint=use_checkpoint_lowcost),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

        if monochrome_adapter:
            self.rgb_to_mono = MonochromeAdapter(to_mono=True, needs_var=out_channels>3)

        if rgb_adapter:
            self.output_to_rgb = DropinRGBAdapter(needs_var=out_channels>3)

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

    def forward(self, x, timesteps, y=None, txt=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # print(f"forward: txt passed = {txt is not None}, model txt = {self.txt}")
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        assert (txt is not None) == (
            self.txt
        ), "must specify txt if and only if the model is text-conditional"

        hs = []
        emb = timestep_embedding(timesteps, self.model_channels).to(self.time_embed[0].weight.dtype)
        emb = self.time_embed(emb)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        attn_mask = None
        if txt is not None:
            txt, attn_mask = self.text_encoder(txt, timesteps=timesteps)
            txt = txt.type(self.inner_dtype)

        h = x

        if self.monochrome_adapter:
            h = self.mono_to_rgb(h)
        if self.rgb_adapter:
            h = self.rgb_to_input(h)

        h = h.type(self.inner_dtype)
        if self.channels_last_mem:
            h = h.to(memory_format=th.channels_last)
        for module in self.input_blocks:
            print(("h.device before", h.device))
            h, txt = module((h, txt), emb, attn_mask=attn_mask, tgt_pos_embs=self.tgt_pos_embs)
            print(("h.device after", h.device))
            hs.append(h)
            print(("hs[-1].device after", hs[-1].device))
        print(('h.device', h.device))
        for hix, hh in enumerate(hs):
            print((f'hs[{hix}].device', hs[hix].device))
        h, txt = self.middle_block((h, txt), emb, attn_mask=attn_mask, tgt_pos_embs=self.tgt_pos_embs)
        # h = h.to(hs[0].device)  # deepspeed
        print(('h.device', h.device))
        for hix, hh in enumerate(hs):
            print((f'hs[{hix}].device', hs[hix].device))
        for hix, module in enumerate(self.output_blocks):
            tocat = hs.pop()
            # h = h.to(tocat.device)
            print(('h.device', h.device))
            print(('tocat.device', tocat.device))
            cat_in = th.cat([h, tocat], dim=1)
            # cat_in = th.cat([h, hs[len(hs) - hix - 1]], dim=1)
            h, txt = module((cat_in, txt), emb, attn_mask=attn_mask, tgt_pos_embs=self.tgt_pos_embs)

        # # !!!!!!! changed for deepspeed, breaking change to non-deepspeed, TODO: fix
        # if False:
        #     h = h.type(x.dtype)
        h = h.type(x.dtype)

        h = self.out(h)

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
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
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
