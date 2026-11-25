"""DSACQ: SACQ head with explicit level / trend / residual coarse branches."""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from layers.PatchTST_backbone import (
    Flatten_Head,
    SACQCrossAttentionHead,
    _merge_patch_predictions,
)
from layers.PatchTST_layers import series_decomp


class DSACQCrossAttentionHead(SACQCrossAttentionHead):
    """SACQ head with optional decomposed coarse scaffold.

    When ``dsacq_coarse_split`` is enabled and ``hist_seq`` is provided:

        scaffold = level_patches + trend_patches + residual_patches
        y = scaffold + alpha * correction   (same gated readout as SACQ)

    Level: last value + slope extrapolation (no sigmoid cap).
    Trend: ``series_decomp`` moving average history -> Linear(seq_len -> future).
    Residual: separate ``resid_flatten_head`` on encoder features (seasonal SACQ path).

    Set ``dsacq_coarse_split=False`` to recover the original single-branch SACQ coarse.
    """

    def __init__(
        self,
        *args,
        context_window: int,
        dsacq_coarse_split: bool = True,
        dsacq_decomp_kernel: int = 25,
        dsacq_level_slope_k: int = 25,
        head_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, head_dropout=head_dropout, **kwargs)
        self.dsacq_coarse_split = bool(dsacq_coarse_split)
        self.context_window = int(context_window)
        self.dsacq_decomp_kernel = int(dsacq_decomp_kernel)
        self.level_slope_k = max(1, int(dsacq_level_slope_k))
        self._hist_seq: Optional[torch.Tensor] = None

        coarse_flat_len = self.out_patch_num * self.patch_len
        if self.dsacq_coarse_split:
            self.decomp_module = series_decomp(self.dsacq_decomp_kernel)
            self.resid_flatten_head = Flatten_Head(
                individual=self.individual,
                n_vars=self.n_vars,
                nf=self.nf,
                target_window=coarse_flat_len,
                head_dropout=head_dropout,
            )
            if self.individual:
                self.trend_proj = nn.ModuleList(
                    [
                        nn.Linear(self.context_window, coarse_flat_len)
                        for _ in range(self.n_vars)
                    ]
                )
            else:
                self.trend_proj = nn.Linear(self.context_window, coarse_flat_len)
        else:
            self.decomp_module = None
            self.resid_flatten_head = None
            self.trend_proj = None

    def _level_future_seq(self, hist_seq: torch.Tensor) -> torch.Tensor:
        """[B, nvars, target_window] level extrapolation."""
        b, c, seq_len = hist_seq.shape
        k = min(self.level_slope_k, max(1, seq_len - 1))
        last = hist_seq[:, :, -1]
        slope = (hist_seq[:, :, -1] - hist_seq[:, :, -1 - k]) / float(k)
        steps = torch.arange(
            1,
            self.target_window + 1,
            device=hist_seq.device,
            dtype=hist_seq.dtype,
        ).view(1, 1, -1)
        return last.unsqueeze(-1) + slope.unsqueeze(-1) * steps

    def _trend_future_flat(self, hist_seq: torch.Tensor) -> torch.Tensor:
        """Moving-average trend history -> coarse flat future [B, nvars, O*p]."""
        x_blc = hist_seq.permute(0, 2, 1)
        _, trend_blc = self.decomp_module(x_blc)
        trend_bcl = trend_blc.permute(0, 2, 1)
        if trend_bcl.shape[-1] != self.context_window:
            trend_bcl = trend_bcl[..., -self.context_window :]
        if self.individual:
            outs = []
            for i in range(self.n_vars):
                outs.append(self.trend_proj[i](trend_bcl[:, i, :]))
            return torch.stack(outs, dim=1)
        return self.trend_proj(trend_bcl)

    def _future_flat_to_patch_bn(
        self, future_flat: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """[B, nvars, O*p] -> [B*nvars, O, p]."""
        nvars = self.n_vars
        patches = future_flat.reshape(batch_size, nvars, self.out_patch_num, self.patch_len)
        return patches.reshape(batch_size * nvars, self.out_patch_num, self.patch_len)

    def _level_patch_bn(self, hist_seq: torch.Tensor, batch_size: int) -> torch.Tensor:
        level_seq = self._level_future_seq(hist_seq)
        need = self.out_patch_num * self.patch_len
        if level_seq.shape[-1] < need:
            pad = level_seq.new_zeros(batch_size, self.n_vars, need - level_seq.shape[-1])
            level_seq = torch.cat([level_seq, pad], dim=-1)
        level_flat = level_seq[..., :need]
        return self._future_flat_to_patch_bn(level_flat, batch_size)

    def _split_coarse_patches(
        self, x: torch.Tensor, kv: torch.Tensor, hist_seq: torch.Tensor
    ) -> torch.Tensor:
        if self.sacq_coarse_init != "flatten":
            raise ValueError(
                "DSACQ coarse split requires sacq_coarse_init='flatten' "
                f"(got {self.sacq_coarse_init!r})."
            )
        bs = x.shape[0]
        level_bn = self._level_patch_bn(hist_seq, bs)
        trend_flat = self._trend_future_flat(hist_seq)
        trend_bn = self._future_flat_to_patch_bn(trend_flat, bs)
        resid_flat = self.resid_flatten_head(x)
        resid_bn = resid_flat.reshape(
            bs * self.n_vars, self.out_patch_num, self.patch_len
        )
        return level_bn + trend_bn + resid_bn

    def _compute_coarse_patches(self, x, kv):
        hist_seq = self._hist_seq
        if self.dsacq_coarse_split and hist_seq is not None:
            return self._split_coarse_patches(x, kv, hist_seq)
        return super()._compute_coarse_patches(x, kv)

    def forward(self, x, hist_seq: Optional[torch.Tensor] = None):
        split = self.dsacq_coarse_split and hist_seq is not None
        if not split:
            return super().forward(x)

        self._hist_seq = hist_seq
        try:
            y, aux = super().forward(x)
        finally:
            self._hist_seq = None

        bs = x.shape[0]
        level_seq = self._level_future_seq(hist_seq)
        trend_flat = self._trend_future_flat(hist_seq)
        trend_seq = trend_flat[..., : self.target_window]
        resid_flat = self.resid_flatten_head(x)
        resid_seq = _merge_patch_predictions(
            resid_flat.reshape(bs, self.n_vars, self.out_patch_num, self.patch_len),
            self.target_window,
            self.patch_len,
            self.out_patch_stride,
            self.overlap_avg,
        )
        aux = dict(aux)
        aux["dsacq_level_seq"] = level_seq
        aux["dsacq_trend_seq"] = trend_seq
        aux["dsacq_resid_flat_seq"] = resid_seq
        aux["dsacq_coarse_split"] = True
        return y, aux
