from typing import Optional

import torch as th
from torch.nn.functional import _scaled_dot_product_attention


class BetterMultiheadAttention(th.nn.MultiheadAttention):
    def __init__(self, src_embed_dim, tgt_embed_dim, num_heads, qkv_dim=None, dropout=0., batch_first=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(th.nn.MultiheadAttention, self).__init__()
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

        self.q = th.nn.Linear(tgt_embed_dim, self.qkv_dim, bias=False)
        self.k = th.nn.Linear(src_embed_dim, self.qkv_dim, bias=False)
        self.v = th.nn.Linear(src_embed_dim, self.qkv_dim, bias=False)

        # self.scale = self.num_heads ** 0.5

        self.register_parameter('in_proj_weight', None)
        self.register_parameter('in_proj_bias', None)

        self.out_proj = th.nn.Linear(self.qkv_dim, tgt_embed_dim, bias=False, **factory_kwargs)

        # self.register_buffer("fake_proj_weight", th.eye(self.qkv_dim, **factory_kwargs), persistent=False)

        # self._reset_parameters()

    def _reset_parameters(self):
        th.nn.init.xavier_uniform_(self.q.weight)
        th.nn.init.xavier_uniform_(self.k.weight)
        th.nn.init.xavier_uniform_(self.v.weight)
        th.nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, query, key, value,
                attn_mask=None,
                need_weights: bool = True):
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        query = self.q(query)
        key = self.k(key)
        value = self.v(value)

        in_dtype = query.dtype

        attn_output, attn_output_weights = better_multi_head_attention_forward(
            query, key, value, self.num_heads,
            self.dropout,
            training=self.training,
            need_weights=need_weights,
            attn_mask=attn_mask,
            )

        attn_output = attn_output.to(in_dtype)

        attn_output = self.out_proj(attn_output)

        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

def better_multi_head_attention_forward(
    query: th.Tensor,
    key: th.Tensor,
    value: th.Tensor,
    num_heads: int,
    dropout_p: float,
    out_proj_weight: th.Tensor,
    out_proj_bias: Optional[th.Tensor],
    training: bool = True,
    need_weights: bool = True,
    attn_mask: Optional[th.Tensor] = None,
) -> Tuple[th.Tensor, Optional[th.Tensor]]:
    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape

    if isinstance(embed_dim, th.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads

    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"

    q, k, v = query, key, value

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)

    # update source sequence length after adjustments
    src_len = k.size(1)

    # convert mask to float
    if attn_mask is not None and attn_mask.dtype == th.bool:
        new_attn_mask = th.zeros_like(attn_mask, dtype=q.dtype)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #
    attn_output, attn_output_weights = _scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # optionally average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)

        return attn_output, attn_output_weights
    else:
        return attn_output, None
