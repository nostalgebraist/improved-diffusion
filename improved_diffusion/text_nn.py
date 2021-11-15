import numpy as np

import torch
import torch.nn as nn

from axial_positional_embedding import AxialPositionalEmbedding
from einops import rearrange
from x_transformers import TransformerWrapper, Encoder, XTransformer

from .nn import normalization, checkpoint


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
            self.model = TransformerWrapper(
                num_tokens = num_tokens,
                max_seq_len = max_seq_len,
                attn_layers = Encoder(
                    dim = inner_dim,
                    depth = depth,
                    heads = n_heads,
                    rotary_pos_emb = rotary_pos_emb,
                    ff_glu = ff_glu,
                    use_scalenorm = use_scalenorm,
                    use_rezero = use_rezero,
                )
            )
        if hasattr(self.model, "to_logits"):
            del self.model.to_logits

        if lr_mult is not None:
            multiply_lr_via_hooks(self, lr_mult)

    def forward(self, tokens):
        if self.use_encoder_decoder:
            tgt = torch.zeros((tokens.shape[0], self.dec_max_seq_len), device=tokens.device, dtype=torch.int)
            enc = self.model.encoder(tokens, return_embeddings = True)
            out = self.model.decoder.net(tgt, context=enc, return_embeddings=True)
            out = rearrange(out, 'b (h w) c -> b h w c', h=self.decoder_sqrt_ntok)
            # out = self.proj(out)
            return out
        else:
            out = self.model(tokens, return_embeddings=True)
            if not self.return_sequences:
                out = out[:, 0, :]
            # out = self.proj(out)
            return out


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        emb_res,
        text_dim=512,
        init_gain=1.,
        gain_scale=200.,
        resid=True,
        lr_mult=None,
        needs_tgt_pos_emb=True,
        use_checkpoint=False,
    ):
        super().__init__()
        print(f"xattn: emb_res {emb_res} | dim {dim} | heads {heads}")
        self.dim = dim
        self.heads = heads
        self.text_dim = text_dim
        self.use_checkpoint = use_checkpoint

        self.q = torch.nn.Linear(self.dim, self.dim, bias=False)
        self.kv = torch.nn.Linear(self.text_dim, 2*self.dim, bias=False)
        self.attn = torch.nn.MultiheadAttention(self.dim, self.heads, batch_first=True)

        self.src_ln = torch.nn.LayerNorm(self.text_dim)
        self.tgt_ln = normalization(self.dim)

        self.emb_res = emb_res
        self.tgt_pos_emb = None
        if needs_tgt_pos_emb:
            self.tgt_pos_emb = AxialPositionalEmbedding(
                dim=self.dim,
                axial_shape=(emb_res, emb_res),
                axial_dims=(self.dim // 2, self.dim // 2),
            )

        self.gain_scale = gain_scale
        self.gain = torch.nn.Parameter(torch.as_tensor(np.log(init_gain) / gain_scale))

        self.resid = resid

        # torch.nn.init.orthogonal_(self.q.weight)
        # torch.nn.init.orthogonal_(self.kv.weight)
        # torch.nn.init.orthogonal_(self.attn.out_proj.weight)

        if lr_mult is not None:
            multiply_lr_via_hooks(self, lr_mult)

    def forward(self, src, tgt, tgt_pos_embs=None):
        return checkpoint(self._forward, (src, tgt, tgt_pos_embs,), self.parameters(), self.use_checkpoint)

    def _forward(self, src, tgt, tgt_pos_embs=None):
        b, c, *spatial = tgt.shape
        tgt = tgt.reshape(b, c, -1)
        tgt_in = self.tgt_ln(tgt)
        tgt_in = tgt_in.transpose(1, 2)

        if tgt_pos_embs is None:
            tgt_pos_emb = self.tgt_pos_emb
        else:
            tgt_pos_emb = tgt_pos_embs[str(self.emb_res)]
        if tgt_pos_emb is None:
            raise ValueError('must pass tgt_pos_emb')

        tgt_in = tgt_in + tgt_pos_emb(tgt_in)

        q = self.q(tgt_in)

        src = self.src_ln(src)
        kv = self.kv(src)

        k, v = kv.chunk(2, dim=-1)

        attn_output, attn_output_weights = self.attn(q, k, v)
        attn_output = (self.gain_scale * self.gain).exp() * attn_output
        attn_output = rearrange(attn_output, 'b (h w) c -> b c h w', h=spatial[0])

        if self.resid:
            tgt = tgt.reshape(b, c, *spatial)
            # with torch.no_grad():
            #     norm_in = torch.linalg.norm(tgt).item()
            #     norm_add = torch.linalg.norm(attn_output).item()
            return tgt + attn_output

        return attn_output
