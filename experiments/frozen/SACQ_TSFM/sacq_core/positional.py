"""Positional encoding utilities for SACQ (shared timeline, continuation, relative time bias)."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


def compute_out_patch_num(
    target_window: int,
    patch_len: int,
    patch_stride: int,
    overlap_avg: bool = False,
) -> int:
    patch_stride = int(patch_len if patch_stride is None else patch_stride)
    if overlap_avg:
        out_stride = patch_stride
        if target_window <= patch_len:
            return 1
        return int(math.ceil((target_window - patch_len) / out_stride)) + 1
    return int(math.ceil(target_window / patch_len))


def continuation_future_pe(
    hist_pos_emb: torch.Tensor,
    out_patch_num: int,
    pe_delta: torch.Tensor,
) -> torch.Tensor:
    """Future PE: e_{N_p-1} + (k+1) * delta for k = 0 .. N_f-1. Returns [N_f, D]."""
    if out_patch_num < 1:
        raise ValueError(f"out_patch_num must be >= 1, got {out_patch_num}.")
    e_last = hist_pos_emb[-1]
    steps = torch.arange(1, out_patch_num + 1, device=hist_pos_emb.device, dtype=hist_pos_emb.dtype)
    return e_last.unsqueeze(0) + steps.unsqueeze(1) * pe_delta.unsqueeze(0)


class RelativeTimeBias(nn.Module):
    """Learnable bias b(tau_q - tau_k) on cross-attention logits (T5-style buckets)."""

    def __init__(self, n_heads: int, hist_patch_num: int, num_buckets: int = 32):
        super().__init__()
        self.n_heads = int(n_heads)
        self.hist_patch_num = int(hist_patch_num)
        self.num_buckets = int(num_buckets)
        self.bias_emb = nn.Embedding(self.num_buckets, self.n_heads)
        nn.init.zeros_(self.bias_emb.weight)

    def forward(self, fut_len: int, kv_len: int, device, dtype) -> torch.Tensor:
        q_idx = torch.arange(fut_len, device=device).unsqueeze(1)
        kv_idx = torch.arange(kv_len, device=device).unsqueeze(0)
        rel = (self.hist_patch_num + q_idx) - kv_idx
        center = self.num_buckets // 2
        buckets = (rel + center).clamp(0, self.num_buckets - 1).long()
        bias = self.bias_emb(buckets).to(dtype=dtype)
        return bias.permute(2, 0, 1)

    def expanded_attn_mask(self, fut_len: int, kv_len: int, batch_size: int, device, dtype) -> torch.Tensor:
        """Shape [batch_size * n_heads, fut_len, kv_len] for nn.MultiheadAttention."""
        bias_hls = self.forward(fut_len, kv_len, device, dtype)
        return (
            bias_hls.unsqueeze(0)
            .expand(batch_size, -1, -1, -1)
            .reshape(batch_size * self.n_heads, fut_len, kv_len)
        )
