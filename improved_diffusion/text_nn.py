import numpy as np

import torch
import torch.nn as nn

from einops import rearrange
from x_transformers import TransformerWrapper, Encoder, XTransformer
from x_transformers.x_transformers import AbsolutePositionalEmbedding, Attention, FeedForward, Rezero

import improved_diffusion.monkeypatch

from .nn import normalization, normalization_1group, timestep_embedding, silu, AdaGN, checkpoint, AxialPositionalEmbeddingShape


def make_grad_mult_hook(mult, debug=False):
    def grad_mult_hook(grad):
        if debug:
            print(f"grad: {grad}")
        new_grad = mult * grad
        if debug:
            print(f"new_grad: {new_grad}")
        return new_grad
    return grad_mult_hook

def multiply_lr_via_hooks(m: nn.Module, mult: float, debug=False) -> nn.Module:
    m._lr_hook_handles = {}

    for n, p in m.named_parameters():
        handle = p.register_hook(make_grad_mult_hook(mult, debug=debug))
        m._lr_hook_handles[n] = handle

    return m


class LineEmbedding(nn.Module):
    def __init__(self, dim, line_sep_id=5, max_lines=32):
        super().__init__()
        self.line_sep_id = line_sep_id
        self.max_lines = max_lines

        self.scale = dim ** -0.5
        self.emb = nn.Embedding(max_lines, dim)

    def forward(self, x):
        n = (x == self.line_sep_id).to(torch.int).cumsum(dim=1).clamp(max=self.max_lines - 1)
        pos_emb = self.emb(n)
        return pos_emb * self.scale


class TextEncoder(nn.Module):
    def __init__(self,
        inner_dim,           # model dim (default = w_dim)
        depth = 2,
        head_dim = 64,
        num_tokens = 2500,
        max_seq_len = 64,
        rotary_pos_emb = False,
        ff_glu = False,
        ff_mult = 4,
        use_scalenorm = True,
        use_rezero = False,
        use_encoder_decoder = False,
        decoder_sqrt_ntok = 32,
        encoder_kwargs = {},
        return_sequences=True,
        lr_mult=None,
        use_line_emb=True,
        tokenizer=None,
        rel_pos_bias=False,
        use_checkpoint=False,
        silu_impl="torch",
    ):
        super().__init__()

        head_dim = min(head_dim, inner_dim)

        assert inner_dim % head_dim == 0
        self.n_heads = inner_dim // head_dim

        self.use_encoder_decoder = use_encoder_decoder
        self.return_sequences = return_sequences
        self.use_line_emb = use_line_emb
        self.dim = inner_dim
        self.use_checkpoint = use_checkpoint

        if tokenizer is not None:
            num_tokens = tokenizer.get_vocab_size()
        print(f"TextEncoder: using num_tokens={num_tokens}, rotary_pos_emb={rotary_pos_emb}, rel_pos_bias={rel_pos_bias}")

        if self.use_encoder_decoder:
            raise ValueError('no longer supported')
        else:
            self.token_emb = nn.Embedding(num_tokens, inner_dim)
            self.pos_emb = AbsolutePositionalEmbedding(inner_dim, max_seq_len)
            if self.use_line_emb:
                self.line_emb = LineEmbedding(dim=inner_dim, line_sep_id=tokenizer.get_vocab()['\n'])
            self.model = Encoder(
                dim = inner_dim,
                depth = depth,
                heads = self.n_heads,
                rotary_pos_emb = rotary_pos_emb,
                ff_glu = ff_glu,
                ff_mult = ff_mult,
                use_scalenorm = use_scalenorm,
                use_rezero = use_rezero,
                rel_pos_bias = rel_pos_bias
            )

            nn.init.kaiming_normal_(self.token_emb.weight)

        if hasattr(self.model, "to_logits"):
            del self.model.to_logits

        self.time_embed_scale = inner_dim ** -0.5
        self.time_embed = nn.Sequential(
            silu(impl="torch" if silu_impl == "fused" else silu_impl),
            nn.Linear(inner_dim, inner_dim),
        )

    def model_forward(self, x, attn_mask):
        return checkpoint(
            self._model_forward, (x, attn_mask, ), self.parameters(), self.use_checkpoint,
            final_nograd=1
        )

    def _model_forward(self, x, attn_mask):
        return self.model.forward(x, attn_mask=attn_mask)

    def forward(self, tokens, timesteps=None):
        if self.use_encoder_decoder:
            raise ValueError('no longer supported')
        else:
            x = tokens
            x = self.token_emb(x)
            # tok_norm = (x ** 2).sum().sqrt().item()
            pe = self.pos_emb(x)
            # pe_norm = (pe ** 2).sum().sqrt().item()
            x = x + pe
            if self.use_line_emb:
                le = self.line_emb(tokens)
                x = x + le

            if timesteps is not None:
                emb = self.time_embed_scale * self.time_embed(timestep_embedding(timesteps, self.dim))
                emb = emb.unsqueeze(1).tile((1, x.shape[1], 1))
                x = x + emb

            # TODO: workaround for HF tokenizers setting PAD and CLS to id 0
            # cf. pad_id arg of enable_padding()
            attn_mask = tokens != 0
            my_attn_mask = torch.tile(attn_mask.unsqueeze(1).unsqueeze(1), (self.n_heads, tokens.shape[1], 1))

            out = self.model_forward(x, attn_mask=my_attn_mask)
            if not self.return_sequences:
                out = out[:, 0, :], attn_mask
            return out, attn_mask


class BetterMultiheadAttention(torch.nn.MultiheadAttention):
    def __init__(self, src_embed_dim, tgt_embed_dim, num_heads, qkv_dim=None, dropout=0., batch_first=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(torch.nn.MultiheadAttention, self).__init__()
        self.src_embed_dim = src_embed_dim
        self.tgt_embed_dim = tgt_embed_dim
        self.embed_dim = self.src_embed_dim
        if qkv_dim is None:
            qkv_dim = src_embed_dim
        self.qkv_dim = qkv_dim
        # self._qkv_same_embed_dim = self.src_embed_dim == self.tgt_embed_dim  # ??

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = self.qkv_dim // num_heads
        assert self.head_dim * num_heads == self.qkv_dim, "qkv_dim must be divisible by num_heads"

        self.q = torch.nn.Linear(tgt_embed_dim, self.qkv_dim, bias=False)
        self.k = torch.nn.Linear(src_embed_dim, self.qkv_dim, bias=False)
        self.v = torch.nn.Linear(src_embed_dim, self.qkv_dim, bias=False)

        # self.scale = self.num_heads ** 0.5

        self.register_parameter('in_proj_weight', None)
        self.register_parameter('in_proj_bias', None)

        self.out_proj = torch.nn.modules.linear.NonDynamicallyQuantizableLinear(self.qkv_dim, tgt_embed_dim, bias=False, **factory_kwargs)

        self.bias_k = self.bias_v = None
        self.add_zero_attn = False

        self.register_buffer("fake_proj_weight", torch.eye(self.qkv_dim, **factory_kwargs), persistent=False)

        # self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.q.weight)
        torch.nn.init.xavier_uniform_(self.k.weight)
        torch.nn.init.xavier_uniform_(self.v.weight)
        torch.nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, query, key, value,
                attn_mask=None,
                need_weights: bool = True):
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        query = self.q(query)
        key = self.k(key)
        value = self.v(value)

        in_dtype = query.dtype

        attn_output, attn_output_weights = torch.nn.functional.multi_head_attention_forward(
            query, key, value, self.qkv_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            self.bias_k, self.bias_v, self.add_zero_attn,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            key_padding_mask=None, need_weights=need_weights,
            attn_mask=attn_mask, use_separate_proj_weight=True,
            q_proj_weight=self.fake_proj_weight, k_proj_weight=self.fake_proj_weight,
            v_proj_weight=self.fake_proj_weight)

        attn_output = attn_output.to(in_dtype)

        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        emb_res,
        time_embed_dim,
        text_dim=512,
        init_gain=1.,
        gain_scale=200.,
        resid=True,
        lr_mult=None,
        needs_tgt_pos_emb=True,
        avoid_groupnorm=False,
        groupnorm_1group=True,
        orth_init=False,
        q_t_emb=False,
        use_rezero=False,
        rezero_keeps_prenorm=False,
        use_layerscale=False,
        layerscale_init=1e-5,
        qkv_dim=None,
        use_checkpoint=False,
        image_base_channels=-1,
        silu_impl="torch",
        txt_already_normed=False,
        post_txt_image_attn=None,
    ):
        super().__init__()
        print(
            f"xattn: emb_res {emb_res} | dim {dim} | qkv_dim {qkv_dim} | heads {heads} | avoid_groupnorm {avoid_groupnorm} | q_t_emb {q_t_emb} | use_rezero {use_rezero}"
        )
        self.dim = dim
        self.heads = heads
        self.text_dim = text_dim

        # self.q = torch.nn.Linear(self.dim, self.dim, bias=False)
        # self.kv = torch.nn.Linear(self.text_dim, 2*self.dim, bias=False)
        self.attn = BetterMultiheadAttention(self.text_dim, self.dim, self.heads, qkv_dim=qkv_dim, batch_first=True)

        self.avoid_groupnorm = avoid_groupnorm
        self.q_t_emb = q_t_emb
        self.use_rezero = use_rezero
        self.use_layerscale = use_layerscale
        self.no_prenorm = txt_already_normed or (use_rezero and not rezero_keeps_prenorm)
        self.use_checkpoint = use_checkpoint

        if self.no_prenorm:
            self.src_ln = nn.Identity()
        else:
            self.src_ln = torch.nn.LayerNorm(self.text_dim)

        self.emb_res = emb_res
        self.tgt_pos_emb = None
        if needs_tgt_pos_emb:
            pos_emb_dim = self.dim // 2
            # pos emb in AdaGN
            if (not avoid_groupnorm) and self.q_t_emb:
                pos_emb_dim *= 2
            self.tgt_pos_emb = AxialPositionalEmbeddingShape(
                dim=self.dim,
                axial_shape=(emb_res, emb_res),
                axial_dims=(pos_emb_dim, pos_emb_dim),
            )

        if self.q_t_emb:
            self.tgt_ln = AdaGN(
                emb_channels=time_embed_dim,
                out_channels=self.dim,
                num_groups=1 if groupnorm_1group else 32,
                silu_in=False,  # pre-silu
                do_norm=not self.no_prenorm,
                base_channels=image_base_channels,
                silu_impl=silu_impl
            )
        elif self.no_prenorm:
            self.tgt_ln = nn.Identity()
        elif avoid_groupnorm:
            self.tgt_ln = torch.nn.LayerNorm(self.dim)
        else:
            cls = normalization_1group if groupnorm_1group else normalization
            self.tgt_ln = cls(self.dim, base_channels=image_base_channels)

        self.gain_scale = gain_scale
        if self.use_layerscale:
            self.gain = torch.nn.Parameter(layerscale_init * torch.ones(self.dim))
        elif self.use_rezero:
            self.gain = torch.nn.Parameter(torch.zeros(1))
        else:
            self.gain = torch.nn.Parameter(torch.as_tensor(np.log(init_gain) / gain_scale))

        self.resid = resid

        self.post_txt_image_attn = post_txt_image_attn

        if orth_init:
            torch.nn.init.orthogonal_(self.attn.q.weight)
            torch.nn.init.orthogonal_(self.attn.k.weight)
            torch.nn.init.orthogonal_(self.attn.v.weight)
            torch.nn.init.orthogonal_(self.attn.out_proj.weight)

        # if lr_mult is not None:
        #     multiply_lr_via_hooks(self, lr_mult)

    def effective_gain(self):
        g = self.gain_scale * self.gain
        if not (self.use_rezero or self.use_layerscale):
            g = g.exp()
        return g

    def forward(self, src, tgt, attn_mask=None, tgt_pos_embs=None, timestep_emb=None):
        tgt_pos_emb = tgt_pos_embs[str(self.emb_res)]
        b, c, *spatial = tgt.shape
        tgt_in_shape  = (b, spatial[0]*spatial[1], c)
        tgt_pos_emb_val = tgt_pos_emb(tgt_in_shape, device=tgt.device, dtype=tgt.dtype)

        return checkpoint(
            self._forward, (src, tgt, tgt_pos_emb_val, timestep_emb, attn_mask, ), self.parameters(), self.use_checkpoint,
            final_nograd=1
        )

    def _forward(self, src, tgt, tgt_pos_emb_val, timestep_emb, attn_mask):
        def _to_b_hw_c(x, retdims=True):
            b, c, *spatial = x.shape
            xt = x.reshape(b, c, -1).transpose(1, 2)
            if retdims:
                return xt, b, c, spatial
            return xt

        def _to_b_c_h_w(x, spatial):
            return rearrange(x, 'b (h w) c -> b c h w', h=spatial[0])

        if self.avoid_groupnorm:
            tgt_in, b, c, spatial = _to_b_hw_c(tgt)
            tgt_in = tgt_in + tgt_pos_emb_val
            tgt_in = self.tgt_ln(tgt_in)
        elif self.q_t_emb:
            tgt_in = tgt

            b, c, *spatial = tgt_in.shape

            tgt_in = self.tgt_ln(h=tgt_in, emb=timestep_emb, side_emb=_to_b_c_h_w(tgt_pos_emb_val, spatial))

            tgt_in, b, c, spatial = _to_b_hw_c(tgt_in)
        else:
            tgt_in = tgt
            tgt_in = self.tgt_ln(tgt_in)
            tgt_in, b, c, spatial = _to_b_hw_c(tgt_in)
            # pos emb after ln, so the GroupNorm doesn't avg it away
            tgt_in = tgt_in + tgt_pos_emb_val

        # q = self.q(tgt_in)
        q = tgt_in

        src_in = self.src_ln(src)
        # kv = self.kv(src)
        # k, v = kv.chunk(2, dim=-1)
        k = src_in
        v = src_in

        my_attn_mask = None
        if attn_mask is not None:
            my_attn_mask = torch.tile(attn_mask.unsqueeze(1), (self.heads, q.shape[1], 1))
            my_attn_mask = (~my_attn_mask).to(q.dtype) * -10000.

        attn_output, attn_output_weights = self.attn(q, k, v, attn_mask=my_attn_mask)
        attn_output = attn_output * self.effective_gain()
        attn_output = _to_b_c_h_w(attn_output, spatial)

        if False and self.heads == 1:
            # attn_output_weights: (bsz, num_heads, tgt_len, src_len)
            max_over_src = attn_output_weights.max(dim=-1).values

            eql = 1./attn_output_weights.shape[-1]
            print(('max_over_src',
                   'avg', f"{(max_over_src.mean()).item():.4f}",
                   'max', f"{max_over_src.max().item() / eql:.4f}",
                   'min', f"{max_over_src.min().item() / eql:.4f}",
                   'tmx', f"{max_over_src.max().item():.4f}",
                   'tmn', f"{max_over_src.min().item():.4f}",
                   ))

            tgt_norm1 = (tgt.float() ** 2).sum().sqrt().item()
            attn_norm = (attn_output.float() ** 2).sum().sqrt().item()

        if self.resid:
            tgt = tgt + attn_output

            if self.post_txt_image_attn is not None:
                tgt = self.post_txt_image_attn(tgt)
            return tgt, src

        return attn_output, src


class ImageToTextCrossAttention(nn.Module):
    def __init__(
        self,
        image_dim,
        heads,
        emb_res,
        time_embed_dim,
        text_dim,
        gain_scale=1.,
        init_gain=1.,
        orth_init=False,
        q_t_emb=False,
        use_rezero=False,
        use_layerscale=False,
        layerscale_init=1e-5,
        use_ff=True,
        ff_rezero=True,
        ff_force_prenorm=False,
        ff_mult=4,
        ff_glu=False,
        qkv_dim=None,
        use_checkpoint=False,
        use_ff_gain=False,
        image_base_channels=-1,
        silu_impl="torch",
        txt_already_normed=False,
        groupnorm_1group=True,
    ):
        super().__init__()
        if qkv_dim is None:
            qkv_dim = image_dim
        print(
            f"itot:  emb_res {emb_res} | image_dim {image_dim} | text_dim {text_dim} | qkv_dim {qkv_dim} | heads {heads} | dim_head {qkv_dim // heads}"
        )

        self.image_dim = image_dim
        self.heads = heads
        self.text_dim = text_dim

        self.attn = BetterMultiheadAttention(self.image_dim, self.text_dim, self.heads, qkv_dim=qkv_dim, batch_first=True)

        self.use_rezero = use_rezero
        self.use_layerscale = use_layerscale
        self.use_checkpoint = use_checkpoint

        if txt_already_normed:
            self.tgt_ln = torch.nn.Identity()
        else:
            self.tgt_ln = torch.nn.LayerNorm(self.text_dim)

        self.emb_res = emb_res

        self.src_ln = AdaGN(
            emb_channels=time_embed_dim,
            out_channels=self.image_dim,
            num_groups=1 if groupnorm_1group else 32,
            silu_in=False,  # pre-silu
            do_norm=True,
            base_channels=image_base_channels,
            silu_impl=silu_impl
        )

        self.gain_scale = gain_scale
        if self.use_layerscale:
            self.gain = torch.nn.Parameter(layerscale_init * torch.ones(self.text_dim))
            if use_ff and use_ff_gain:
                self.gain_ff = torch.nn.Parameter(layerscale_init * torch.ones(self.text_dim))
        elif self.use_rezero:
            self.gain = torch.nn.Parameter(torch.zeros(1))
            if use_ff and use_ff_gain:
                self.gain_ff = torch.nn.Parameter(torch.zeros(1))
        else:
            self.gain = torch.nn.Parameter(torch.as_tensor(np.log(init_gain) / gain_scale))
            if use_ff:
                if use_ff_gain:
                    self.gain_ff = torch.nn.Parameter(torch.as_tensor(np.log(init_gain) / gain_scale))
                else:
                    self.gain_ff = np.log(init_gain)

        self.use_ff = use_ff
        self.use_ff_gain = use_ff_gain
        self.ff = None
        if use_ff:
            ff = FeedForward(dim=text_dim, mult=ff_mult, glu=ff_glu)
            if ff_rezero:
                ff = Rezero(ff)
            self.ff = ff

            if ff_force_prenorm or (not ff_rezero):
                self.ff_ln = torch.nn.LayerNorm(self.text_dim)
            else:
                self.ff_ln = nn.Identity()

        if orth_init:
            torch.nn.init.orthogonal_(self.attn.q.weight)
            torch.nn.init.orthogonal_(self.attn.k.weight)
            torch.nn.init.orthogonal_(self.attn.v.weight)
            torch.nn.init.orthogonal_(self.attn.out_proj.weight)

    def _effective_gain(self, base):
        g = self.gain_scale * base
        if not (self.use_rezero or self.use_layerscale):
            g = g.exp()
        return g

    def effective_gain(self):
        return self._effective_gain(self.gain)

    def effective_gain_ff(self):
        if not self.use_ff_gain:
            return 1.
        return self._effective_gain(self.gain_ff)

    def forward(self, src, tgt, attn_mask=None, image_pos_embs=None, timestep_emb=None):
        src_pos_emb = image_pos_embs[str(self.emb_res)]
        b, c, *spatial = src.shape

        src_in_shape  = (b, spatial[0]*spatial[1], c)
        src_pos_emb_val = src_pos_emb(src_in_shape, device=src.device, dtype=src.dtype)

        return checkpoint(
            self._forward, (src, tgt, src_pos_emb_val, timestep_emb, attn_mask, ), self.parameters(), self.use_checkpoint,
            final_nograd=1
        )

    def _forward(self, src, tgt, src_pos_emb_val, timestep_emb, attn_mask, ):
        def _to_b_hw_c(x, retdims=True):
            b, c, *spatial = x.shape
            xt = x.reshape(b, c, -1).transpose(1, 2)
            if retdims:
                return xt, b, c, spatial
            return xt

        def _to_b_c_h_w(x, spatial):
            return rearrange(x, 'b (h w) c -> b c h w', h=spatial[0])

        # image
        src_in = src

        b, c, *spatial = src_in.shape
        pos_emb = src_pos_emb_val
        pos_emb = _to_b_c_h_w(pos_emb, spatial)

        src_in = self.src_ln(h=src_in, emb=timestep_emb, side_emb=pos_emb)

        src_in, b, c, spatial = _to_b_hw_c(src_in)

        k = src_in
        v = src_in

        # text
        tgt_in = self.tgt_ln(tgt)
        q = tgt_in

        my_attn_mask = None
        if attn_mask is not None:
            my_attn_mask = torch.tile(attn_mask.unsqueeze(2), (self.heads, 1, k.shape[1]))
            my_attn_mask = (~my_attn_mask).to(q.dtype) * -10000.

        attn_output, attn_output_weights = self.attn(q, k, v, attn_mask=my_attn_mask)
        attn_output = attn_output * self.effective_gain()

        tgt = tgt + attn_output

        if self.use_ff:
            ff_output = self.ff(self.ff_ln(tgt))
            ff_output = ff_output * self.effective_gain_ff()
            tgt = tgt + ff_output
        return tgt, src


class WeaveAttention(nn.Module):
    def __init__(
        self,
        image_dim,
        heads,
        emb_res,
        time_embed_dim,
        text_dim,
        n_layers=1,
        gain_scale=1.,
        init_gain=1.,
        orth_init=True,
        q_t_emb=True,
        use_rezero=False,
        rezero_keeps_prenorm=False,
        use_layerscale=False,
        layerscale_init=1e-5,
        use_ff=True,
        ff_rezero=True,
        ff_force_prenorm=False,
        ff_mult=4,
        ff_glu=False,
        qkv_dim_always_text=False,
        weave_v2=False,
        use_checkpoint=False,
        use_ff_gain=False,
        image_base_channels=-1,
        silu_impl="torch",
        no_itot=False,
        txt_already_normed=False,
        post_txt_image_attn=None,
        groupnorm_1group=True,
        **text_to_image_kwargs,
    ):
        super().__init__()

        self.weave_v2 = weave_v2

        shared_args = dict(
            heads=heads,
            emb_res=emb_res,
            time_embed_dim=time_embed_dim,
            text_dim=text_dim,
            gain_scale=gain_scale,
            init_gain=init_gain,
            orth_init=orth_init,
            use_rezero=use_rezero,
            use_layerscale=use_layerscale,
            layerscale_init=layerscale_init,
            use_checkpoint=use_checkpoint,
            image_base_channels=image_base_channels,
            silu_impl=silu_impl,
            txt_already_normed=txt_already_normed,
            groupnorm_1group=groupnorm_1group,
        )

        text_to_image_kwargs.update(
            dict(
                dim=image_dim,
                q_t_emb=q_t_emb,
                rezero_keeps_prenorm=rezero_keeps_prenorm,
                qkv_dim=text_dim if qkv_dim_always_text else None,
                post_txt_image_attn=post_txt_image_attn,
                **shared_args
            )
        )

        image_to_text_kwargs = dict(
            image_dim=image_dim,
            use_ff=use_ff,
            ff_rezero=ff_rezero,
            ff_force_prenorm=ff_force_prenorm,
            ff_mult=ff_mult,
            ff_glu=ff_glu,
            qkv_dim=text_dim if qkv_dim_always_text else None,
            use_ff_gain=use_ff_gain,
            **shared_args
        )

        self.no_itot = no_itot

        if no_itot:
            self.image_to_text_layers = nn.Sequential(
                *[nn.Identity() for _ in range(n_layers)]
            )
        else:
            self.image_to_text_layers = nn.Sequential(
                *[ImageToTextCrossAttention(**image_to_text_kwargs) for _ in range(n_layers)]
            )

        self.text_to_image_layers = nn.Sequential(
            *[CrossAttention(**text_to_image_kwargs) for _ in range(n_layers)]
        )


    def forward(self, text, image, attn_mask=None, tgt_pos_embs=None, timestep_emb=None):
        shared_kwargs = dict(attn_mask=attn_mask, timestep_emb=timestep_emb)

        orig_text = text

        for i_to_t, t_to_i in zip(self.image_to_text_layers, self.text_to_image_layers):
            if not self.no_itot:
                text, image = i_to_t(src=image, tgt=text, image_pos_embs=tgt_pos_embs, **shared_kwargs)
            image, text = t_to_i(src=text, tgt=image, tgt_pos_embs=tgt_pos_embs, **shared_kwargs)

        if self.weave_v2:
            return image, text
        return image, orig_text
