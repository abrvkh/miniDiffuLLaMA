import torch


def get_anneal_attn_mask(seq_len: int, bsz: int, dtype, device, attn_mask_ratio: float) -> torch.Tensor:
    """
    Builds a causal mask and gradually relaxes it with random extra attention
    according to attn_mask_ratio. Returns a 4D additive mask with 0 for allowed
    positions and -inf for blocked positions.
    """
    causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=dtype))
    random_mask = torch.bernoulli(torch.full((seq_len, seq_len), 0.0, device=device) + attn_mask_ratio)
    anneal_mask = torch.logical_or(causal_mask, random_mask)
    expanded_mask = anneal_mask[None, None, :, :].expand(bsz, 1, seq_len, seq_len)
    inverted_mask = 1.0 - expanded_mask.to(dtype)
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
