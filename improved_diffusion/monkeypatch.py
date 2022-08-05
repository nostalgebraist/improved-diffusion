from x_transformers.x_transformers import *
import x_transformers.x_transformers


def patched_Attention_forward(
    self,
    x,
    context = None,
    mask = None,
    context_mask = None,
    attn_mask = None,
    rel_pos = None,
    sinusoidal_emb = None,
    rotary_pos_emb = None,
    prev_attn = None,
    mem = None
):
    b, n, _, h, talking_heads, head_scale, scale, device, has_context = *x.shape, self.heads, self.talking_heads, self.head_scale, self.scale, x.device, exists(context)
    kv_input = default(context, x)

    q_input = x
    k_input = kv_input
    v_input = kv_input

    if exists(mem):
        k_input = torch.cat((mem, k_input), dim = -2)
        v_input = torch.cat((mem, v_input), dim = -2)

    if exists(sinusoidal_emb):
        # in shortformer, the query would start at a position offset depending on the past cached memory
        offset = k_input.shape[-2] - q_input.shape[-2]
        q_input = q_input + sinusoidal_emb(q_input, offset = offset)
        k_input = k_input + sinusoidal_emb(k_input)

    q = self.to_q(q_input)
    k = self.to_k(k_input)
    v = self.to_v(v_input) if exists(self.to_v) else k

    q = rearrange(q, 'b n (h d) -> b h n d', h = h)

    if not self.one_kv_head:
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (k, v))

    if exists(rotary_pos_emb) and not has_context:
        l = rotary_pos_emb.shape[-1]
        (ql, qr), (kl, kr), (vl, vr) = map(lambda t: (t[..., :l], t[..., l:]), (q, k, v))
        ql, kl, vl = map(lambda t: apply_rotary_pos_emb(t, rotary_pos_emb), (ql, kl, vl))
        q, k, v = map(lambda t: torch.cat(t, dim = -1), ((ql, qr), (kl, kr), (vl, vr)))

    input_mask = None
    if any(map(exists, (mask, context_mask))):
        q_mask = default(mask, lambda: torch.ones((b, n), device = device).bool())
        k_mask = q_mask if not exists(context) else context_mask
        k_mask = default(k_mask, lambda: torch.ones((b, k.shape[-2]), device = device).bool())
        q_mask = rearrange(q_mask, 'b i -> b 1 i 1')
        k_mask = rearrange(k_mask, 'b j -> b 1 1 j')
        input_mask = q_mask * k_mask

    if self.num_mem_kv > 0:
        mem_k, mem_v = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), (self.mem_k, self.mem_v))
        k = torch.cat((mem_k, k), dim = -2)
        v = torch.cat((mem_v, v), dim = -2)
        if exists(input_mask):
            input_mask = F.pad(input_mask, (self.num_mem_kv, 0), value = True)

    if self.qk_norm:
        qk_l2norm = partial(l2norm, groups = self.qk_norm_groups)
        q, k = map(qk_l2norm, (q, k))
        scale = self.qk_norm_scale

    kv_einsum_eq = 'b h j d' if not self.one_kv_head else 'b j d'

    dots = einsum(f'b h i d, {kv_einsum_eq} -> b h i j', q, k) * scale

    mask_value = max_neg_value(dots)

    if exists(prev_attn):
        dots = dots + prev_attn

    # pre_softmax_attn = dots.clone()

    if talking_heads:
        dots = self.pre_softmax_talking_heads(dots)

    if exists(rel_pos):
        dots = rel_pos(dots)

    if exists(input_mask):
        dots.masked_fill_(~input_mask, mask_value)
        del input_mask

    if exists(attn_mask):
        assert 2 <= attn_mask.ndim <= 4, 'attention mask must have greater than 2 dimensions but less than or equal to 4'
        if attn_mask.ndim == 2:
            attn_mask = rearrange(attn_mask, 'i j -> 1 1 i j')
        elif attn_mask.ndim == 3:
            attn_mask = rearrange(attn_mask, 'h i j -> 1 h i j')
        dots.masked_fill_(~attn_mask, mask_value)

    if exists(self.max_attend_past):
        i, j = dots.shape[-2:]
        range_q = torch.arange(j - i, j, device = device)
        range_k = torch.arange(j, device = device)
        dist = rearrange(range_q, 'i -> 1 1 i 1') - rearrange(range_k, 'j -> 1 1 1 j')
        mask = dist > self.max_attend_past
        dots.masked_fill_(mask, mask_value)
        del mask

    if self.causal:
        i, j = dots.shape[-2:]
        r = torch.arange(i, device = device)
        mask = rearrange(r, 'i -> 1 1 i 1') < rearrange(r, 'j -> 1 1 1 j')
        mask = F.pad(mask, (j - i, 0), value = False)
        dots.masked_fill_(mask, mask_value)
        del mask

    if exists(self.sparse_topk) and self.sparse_topk < dots.shape[-1]:
        top, _ = dots.topk(self.sparse_topk, dim = -1)
        vk = top[..., -1].unsqueeze(-1).expand_as(dots)
        mask = dots < vk
        dots.masked_fill_(mask, mask_value)
        del mask

    attn = self.attn_fn(dots, dim = -1)
    # post_softmax_attn = attn.clone()

    attn = self.dropout(attn)

    if talking_heads:
        attn = self.post_softmax_talking_heads(attn)

    out = einsum(f'b h i j, {kv_einsum_eq} -> b h i d', attn, v)

    if head_scale:
        out = out * self.head_scale_params

    out = rearrange(out, 'b h n d -> b n (h d)')

    if exists(self.to_v_gate):
        gates = self.to_v_gate(x)
        out = out * gates.sigmoid()

    intermediates = Intermediates(
        pre_softmax_attn = None,
        post_softmax_attn = None
    )

    return self.to_out(out), intermediates


x_transformers.x_transformers.Attention.forward = patched_Attention_forward
