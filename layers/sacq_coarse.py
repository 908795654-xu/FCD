"""Coarse scaffold init and AR query conditioning for SACQ."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_prev_mean(x: torch.Tensor) -> torch.Tensor:
    """Per position i, mean of x[:, :i, :] (i=0 -> zeros). x: [B, O, *]."""
    if x.shape[1] <= 1:
        out = x.new_zeros(x.shape)
        if x.shape[1] == 1:
            return out
        return out
    csum = torch.cumsum(x, dim=1)
    counts = torch.arange(1, x.shape[1] + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
    mean_incl = csum / counts
    prev = F.pad(mean_incl[:, :-1], (0, 0, 0, 1, 0, 0))
    return prev


class HistoryFutureRoutingCoarse(nn.Module):
    """History->future soft correspondence for coarse patch scaffold F.

    Each future patch queries encoder memory with optional local window / top-k sparsity,
    then aggregates history patch values into patch_len coordinates.
    """

    def __init__(
        self,
        d_model: int,
        patch_len: int,
        out_patch_num: int,
        hist_patch_num: int = 0,
        topk: int = 0,
        local_window: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_len = patch_len
        self.out_patch_num = out_patch_num
        self.hist_patch_num = int(hist_patch_num)
        self.topk = max(0, int(topk))
        self.local_window = max(0, int(local_window))
        self.future_queries = nn.Parameter(torch.zeros(out_patch_num, d_model))
        nn.init.normal_(self.future_queries, mean=0.0, std=0.02)
        self.hist_value_proj = nn.Linear(d_model, patch_len)
        self.dropout = nn.Dropout(dropout)
        self.scale = d_model ** -0.5

    def _local_window_mask(self, n_hist: int, device: torch.device) -> Optional[torch.Tensor]:
        if self.local_window <= 0:
            return None
        o_idx = torch.arange(self.out_patch_num, device=device)
        p_idx = torch.arange(n_hist, device=device)
        center = (n_hist - self.out_patch_num + o_idx).clamp(0, n_hist - 1)
        dist = (p_idx.unsqueeze(0) - center.unsqueeze(1)).abs()
        return dist > self.local_window

    def _apply_topk(self, scores: torch.Tensor) -> torch.Tensor:
        k = min(self.topk, scores.shape[-1])
        if k <= 0 or k >= scores.shape[-1]:
            return scores
        topk_vals, topk_idx = scores.topk(k, dim=-1)
        sparse = scores.new_full(scores.shape, float('-inf'))
        sparse.scatter_(-1, topk_idx, topk_vals)
        return sparse

    def forward(self, kv: torch.Tensor) -> torch.Tensor:
        # kv: [B, P_hist, D]
        batch_size, n_hist, _ = kv.shape
        q = self.future_queries.unsqueeze(0).expand(batch_size, -1, -1)
        scores = torch.matmul(q, kv.transpose(-2, -1)) * self.scale

        local_mask = self._local_window_mask(n_hist, kv.device)
        if local_mask is not None:
            scores = scores.masked_fill(local_mask.unsqueeze(0), float('-inf'))

        scores = self._apply_topk(scores)
        weights = torch.softmax(scores, dim=-1)
        hist_vals = self.hist_value_proj(kv)
        coarse = torch.matmul(weights, hist_vals)
        return self.dropout(coarse)


class SACQARVerifyAdapter(nn.Module):
    """Inject causal context from prior fused patch beliefs into later query tokens."""

    def __init__(self, d_model: int, patch_len: int):
        super().__init__()
        self.ctx_proj = nn.Linear(patch_len, d_model)
        self.gate = nn.Parameter(torch.tensor(-2.5))

    def forward(self, q: torch.Tensor, patch_belief: torch.Tensor) -> torch.Tensor:
        ctx = self.ctx_proj(causal_prev_mean(patch_belief))
        return q + torch.sigmoid(self.gate) * ctx
