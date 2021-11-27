import numpy as np

import torch
import torch.nn as nn

from axial_positional_embedding import AxialPositionalEmbedding
from einops import rearrange
from x_transformers import TransformerWrapper, Encoder, XTransformer
from x_transformers.x_transformers import AbsolutePositionalEmbedding

from .nn import normalization_1group, timestep_embedding, SiLU, AdaGN


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


class TextEncoder(nn.Module):
    def __init__(self,
        inner_dim,           # model dim (default = w_dim)
        depth = 2,
        head_dim = 64,
        num_tokens = 2500,
        max_seq_len = 64,
        rotary_pos_emb = False,
        ff_glu = True,
        use_scalenorm = True,
        use_rezero = False,
        use_encoder_decoder = False,
        decoder_sqrt_ntok = 32,
        encoder_kwargs = {},
        return_sequences=True,
        lr_mult=None,
    ):
        super().__init__()

        head_dim = min(head_dim, inner_dim)

        assert inner_dim % head_dim == 0
        n_heads = inner_dim // head_dim

        self.use_encoder_decoder = use_encoder_decoder
        self.return_sequences = return_sequences
        self.dim = inner_dim

        if self.use_encoder_decoder:
            enc_kwargs = dict(
                depth = depth,
                heads = n_heads,
                rotary_pos_emb = rotary_pos_emb,
                ff_glu = ff_glu,
                use_scalenorm = use_scalenorm,
                use_rezero = use_rezero,
            )
            enc_kwargs = {k: encoder_kwargs.get(k, v) for k, v in enc_kwargs.items()}
            enc_kwargs = {'enc_' + k: v for k, v in enc_kwargs.items()}

            self.decoder_sqrt_ntok = decoder_sqrt_ntok
            self.dec_max_seq_len = decoder_sqrt_ntok ** 2

            self.model = XTransformer(
                enc_num_tokens = num_tokens,
                dec_num_tokens = 1,
                enc_max_seq_len = max_seq_len,
                dec_max_seq_len = self.dec_max_seq_len,
                dim = inner_dim,
                dec_depth = depth,
                dec_heads = n_heads,
                dec_rotary_pos_emb = rotary_pos_emb,
                dec_ff_glu = ff_glu,
                **enc_kwargs
            )
        else:
            self.token_emb = nn.Embedding(num_tokens, inner_dim)
            self.pos_emb = AbsolutePositionalEmbedding(inner_dim, max_seq_len)
            self.model = Encoder(
                dim = inner_dim,
                depth = depth,
                heads = n_heads,
                rotary_pos_emb = rotary_pos_emb,
                ff_glu = ff_glu,
                use_scalenorm = use_scalenorm,
                use_rezero = use_rezero,
            )
        if hasattr(self.model, "to_logits"):
            del self.model.to_logits

        self.time_embed = nn.Sequential(
            nn.Linear(inner_dim, inner_dim),
            SiLU(),
            nn.Linear(inner_dim, inner_dim),
        )

        if lr_mult is not None:
            multiply_lr_via_hooks(self, lr_mult)

    def forward(self, tokens, timesteps=None):
        if self.use_encoder_decoder:
            tgt = torch.zeros((tokens.shape[0], self.dec_max_seq_len), device=tokens.device, dtype=torch.int)
            enc = self.model.encoder(tokens, return_embeddings = True)
            out = self.model.decoder.net(tgt, context=enc, return_embeddings=True)
            out = rearrange(out, 'b (h w) c -> b h w c', h=self.decoder_sqrt_ntok)
            # out = self.proj(out)
            return out
        else:
            x = tokens
            x = self.token_emb(x)
            x = x + self.pos_emb(x)

            if timesteps is not None:
                emb = self.time_embed(timestep_embedding(timesteps, self.dim))
                emb = emb.unsqueeze(1).tile((1, x.shape[1], 1))
                x = x + emb

            out = self.model(x)
            if not self.return_sequences:
                out = out[:, 0, :]
            return out


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
        orth_init=False,
        q_t_emb=False,
        use_rezero=False,
        rezero_keeps_prenorm=False,
        use_layerscale=False,
        layerscale_init=1e-5,
        qk_dim=None
    ):
        super().__init__()
        print(
            f"xattn: emb_res {emb_res} | dim {dim} | heads {heads} | avoid_groupnorm {avoid_groupnorm} | q_t_emb {q_t_emb} | use_rezero {use_rezero}"
        )
        self.dim = dim
        self.heads = heads
        self.text_dim = text_dim
        self.qk_dim = qk_dim
        if self.qk_dim is None:
            self.qk_dim = self.dim

        self.q = torch.nn.Linear(self.dim, self.qk_dim, bias=False)
        self.kv = torch.nn.Linear(self.text_dim, 2*self.qk_dim, bias=False)
        self.attn = torch.nn.MultiheadAttention(self.dim, self.heads, batch_first=True, vdim=self.dim)

        self.avoid_groupnorm = avoid_groupnorm
        self.q_t_emb = q_t_emb
        self.use_rezero = use_rezero
        self.use_layerscale = use_layerscale
        self.no_prenorm = use_rezero and not rezero_keeps_prenorm

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
            self.tgt_pos_emb = AxialPositionalEmbedding(
                dim=self.dim,
                axial_shape=(emb_res, emb_res),
                axial_dims=(pos_emb_dim, pos_emb_dim),
            )


        if self.q_t_emb:
            self.tgt_ln = AdaGN(
                emb_channels=time_embed_dim,
                out_channels=self.dim,
                num_groups=1,
                nonlin_in=True,  # TODO: does this matter?
                do_norm=not self.no_prenorm,
            )
        elif self.no_prenorm:
            self.tgt_ln = nn.Identity()
        elif avoid_groupnorm:
            self.tgt_ln = torch.nn.LayerNorm(self.dim)
        else:
            self.tgt_ln = normalization_1group(self.dim)

        self.gain_scale = gain_scale
        if self.use_layerscale:
            self.gain = torch.nn.Parameter(layerscale_init * torch.ones(self.dim))
        elif self.use_rezero:
            self.gain = torch.nn.Parameter(torch.zeros(1))
        else:
            self.gain = torch.nn.Parameter(torch.as_tensor(np.log(init_gain) / gain_scale))

        self.resid = resid

        if orth_init:
            torch.nn.init.orthogonal_(self.q.weight)
            torch.nn.init.orthogonal_(self.kv.weight)
            torch.nn.init.orthogonal_(self.attn.out_proj.weight)

        if lr_mult is not None:
            multiply_lr_via_hooks(self, lr_mult)

    def effective_gain(self):
        g = self.gain_scale * self.gain
        if not (self.use_rezero or self.use_layerscale):
            g = g.exp()
        return g

    def forward(self, src, tgt, tgt_pos_embs=None, timestep_emb=None):
        def _to_b_hw_c(x, retdims=True):
            b, c, *spatial = x.shape
            xt = x.reshape(b, c, -1).transpose(1, 2)
            if retdims:
                return xt, b, c, spatial
            return xt

        def _to_b_c_h_w(x, spatial):
            return rearrange(x, 'b (h w) c -> b c h w', h=spatial[0])

        if tgt_pos_embs is None:
            tgt_pos_emb = self.tgt_pos_emb
        else:
            tgt_pos_emb = tgt_pos_embs[str(self.emb_res)]
        if tgt_pos_emb is None:
            raise ValueError('must pass tgt_pos_emb')

        if self.avoid_groupnorm:
            tgt_in, b, c, spatial = _to_b_hw_c(tgt)
            tgt_in = tgt_in + tgt_pos_emb(tgt_in)
            tgt_in = self.tgt_ln(tgt_in)
        elif self.q_t_emb:
            tgt_in = tgt

            b, c, *spatial = tgt_in.shape
            pos_emb = tgt_pos_emb(_to_b_hw_c(tgt_in, retdims=False))
            pos_emb = _to_b_c_h_w(pos_emb, spatial)

            tgt_in = self.tgt_ln(h=tgt_in, emb=timestep_emb, side_emb=pos_emb)

            tgt_in, b, c, spatial = _to_b_hw_c(tgt_in)
        else:
            tgt_in = tgt
            tgt_in = self.tgt_ln(tgt_in)
            tgt_in, b, c, spatial = _to_b_hw_c(tgt_in)
            # pos emb after ln, so the GroupNorm doesn't avg it away
            tgt_in = tgt_in + tgt_pos_emb(tgt_in)

        q = self.q(tgt_in)

        src = self.src_ln(src)
        kv = self.kv(src)

        k, v = kv.chunk(2, dim=-1)

        attn_output, attn_output_weights = self.attn(q, k, v)
        attn_output = attn_output * self.effective_gain()
        attn_output = _to_b_c_h_w(attn_output, spatial)

        if self.resid:
            return tgt + attn_output

        return attn_output
