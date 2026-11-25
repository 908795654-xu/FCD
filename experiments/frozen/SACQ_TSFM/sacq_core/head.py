# FCD head extracted/adapted from the PatchTST-derived implementation; see repository NOTICE.md.
"""Standalone SACQ prediction head extracted from the original PatchTST implementation.

Canonical input shape:
    x: [batch, channels, hidden_dim, history_patches]

Output shape:
    prediction: [batch, channels, prediction_length]

The encoder/backbone is intentionally not included in this module.
For the pure SACQ baseline, callers must explicitly set
``dsacq_enable=False``, ``dsacq_mem_gate_enable=False``, and
``dsacq_drift_gate_enable=False``.
"""

from typing import Optional
import math

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .coarse import HistoryFutureRoutingCoarse, SACQARVerifyAdapter
from .positional import RelativeTimeBias, continuation_future_pe

__all__ = ["SACQCrossAttentionHead"]


def _softplus_inv(y: float, eps: float = 1e-6) -> float:
    """Inverse of softplus for positive gate init (alpha = softplus(x))."""
    y = max(float(y), eps)
    return math.log(math.expm1(y))


def _sigmoid_inv(p: float, eps: float = 1e-6) -> float:
    p = min(max(float(p), eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


class Flatten_Head(nn.Module):
    def __init__(
        self,
        individual,
        n_vars,
        nf,
        target_window,
        head_dropout=0,
    ):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])
                z = self.linears[i](z)
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x


def _merge_patch_predictions(patches, target_window, patch_len, out_patch_stride, overlap_avg):
    bs, nvars, out_patch_num, _ = patches.shape
    if overlap_avg:
        merged_len = (out_patch_num - 1) * out_patch_stride + patch_len
        y_sum = patches.new_zeros(bs, nvars, merged_len)
        y_count = patches.new_zeros(bs, nvars, merged_len)
        for patch_idx in range(out_patch_num):
            start = patch_idx * out_patch_stride
            end = start + patch_len
            y_sum[..., start:end] += patches[..., patch_idx, :]
            y_count[..., start:end] += 1.0
        merged = y_sum / y_count.clamp_min(1.0)
    else:
        merged = patches.reshape(bs, nvars, out_patch_num * patch_len)
    return merged[..., : target_window]


class SACQDecoderStack(nn.Module):
    """Cross-self decoder: cross-attn to encoder memory + self-attn over future queries -> patch residual."""

    def __init__(
        self,
        d_model: int,
        patch_len: int,
        cross_attn_layers: int,
        n_heads: int,
        head_dropout: float = 0.0,
        force_causal: bool = False,
        relative_time_bias: Optional[RelativeTimeBias] = None,
        ar_verify: bool = False,
    ):
        super().__init__()
        self.cross_attn_layers = int(cross_attn_layers)
        self.force_causal = bool(force_causal)
        self.ar_verify = bool(ar_verify)
        self.relative_time_bias = relative_time_bias
        self.out_dropout = nn.Dropout(head_dropout)
        n_inter = max(0, self.cross_attn_layers - 1)
        if self.ar_verify and n_inter > 0:
            self.ar_adapter = SACQARVerifyAdapter(d_model, patch_len)
            self.ar_patch_readout = nn.ModuleList(
                [nn.Linear(d_model, patch_len) for _ in range(n_inter)]
            )
        else:
            self.ar_adapter = None
            self.ar_patch_readout = None

        self.cross_attn_layers_stack = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=head_dropout,
                batch_first=True,
            )
            for _ in range(self.cross_attn_layers)
        ])
        self.cross_attn_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(self.cross_attn_layers)
        ])
        self.self_attn_layers_stack = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=head_dropout,
                batch_first=True,
            )
            for _ in range(max(0, self.cross_attn_layers - 1))
        ])
        self.self_attn_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(max(0, self.cross_attn_layers - 1))
        ])
        ffn_hidden = 4 * d_model
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, ffn_hidden),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(ffn_hidden, d_model),
            )
            for _ in range(max(0, self.cross_attn_layers - 1))
        ])
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(max(0, self.cross_attn_layers - 1))
        ])

        self.final_self_in_proj = nn.Linear(d_model, patch_len)
        final_heads = math.gcd(n_heads, patch_len)
        if final_heads < 1:
            final_heads = 1
        self.final_self_attn = nn.MultiheadAttention(
            embed_dim=patch_len,
            num_heads=final_heads,
            dropout=head_dropout,
            batch_first=True,
        )
        self.final_self_norm = nn.LayerNorm(patch_len)
        final_ffn_hidden = 4 * patch_len
        self.final_ffn = nn.Sequential(
            nn.Linear(patch_len, final_ffn_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(final_ffn_hidden, patch_len),
        )
        self.final_ffn_norm = nn.LayerNorm(patch_len)

    @staticmethod
    def _resolve_self_attn_mask(q_len, device, force_causal, optional_mask):
        if force_causal and q_len > 1:
            return SACQCrossAttentionHead._future_self_attn_mask(q_len, device)
        return optional_mask

    def forward(
        self,
        q,
        kv,
        kv_value=None,
        mem_beta=None,
        self_attn_mask=None,
        return_cross_attn: bool = False,
        scaffold_patch=None,
    ):
        attn_out = q
        cross_attn_maps = []
        if self.cross_attn_layers == 0:
            return attn_out.new_zeros(attn_out.shape[0], attn_out.shape[1], self.final_self_in_proj.out_features), cross_attn_maps

        mask = self._resolve_self_attn_mask(q.shape[1], q.device, self.force_causal, self_attn_mask)
        use_ar = (
            self.ar_verify
            and self.ar_adapter is not None
            and scaffold_patch is not None
        )
        batch_size = q.shape[0]
        rtb_mask = None
        if self.relative_time_bias is not None:
            rtb_mask = self.relative_time_bias.expanded_attn_mask(
                q.shape[1], kv.shape[1], batch_size, q.device, q.dtype
            )
        for layer_idx, cross_layer in enumerate(self.cross_attn_layers_stack):
            cross_in = attn_out
            cross_v = kv if kv_value is None else kv_value
            if return_cross_attn:
                cross_out, cross_attn = cross_layer(
                    cross_in,
                    kv,
                    cross_v,
                    attn_mask=rtb_mask,
                    need_weights=True,
                    average_attn_weights=False,
                )
                cross_attn_maps.append(cross_attn.detach())
            else:
                cross_out, _ = cross_layer(
                    cross_in, kv, cross_v, attn_mask=rtb_mask, need_weights=False
                )
            if mem_beta is not None:
                cross_out = cross_out * mem_beta
            attn_out = self.cross_attn_norms[layer_idx](cross_in + self.out_dropout(cross_out))
            if layer_idx < self.cross_attn_layers - 1:
                self_in = attn_out
                self_out, _ = self.self_attn_layers_stack[layer_idx](
                    self_in, self_in, self_in, attn_mask=mask
                )
                attn_out = self.self_attn_norms[layer_idx](self_in + self.out_dropout(self_out))
                ffn_in = attn_out
                ffn_out = self.ffn_layers[layer_idx](ffn_in)
                attn_out = self.ffn_norms[layer_idx](ffn_in + self.out_dropout(ffn_out))
                if use_ar:
                    partial_patch = scaffold_patch + self.ar_patch_readout[layer_idx](attn_out)
                    attn_out = self.ar_adapter(attn_out, partial_patch)
            else:
                self_in = self.final_self_in_proj(attn_out)
                self_out, _ = self.final_self_attn(self_in, self_in, self_in, attn_mask=mask)
                attn_out = self.final_self_norm(self_in + self.out_dropout(self_out))
                final_ffn_in = attn_out
                final_ffn_out = self.final_ffn(final_ffn_in)
                attn_out = self.final_ffn_norm(final_ffn_in + self.out_dropout(final_ffn_out))
        return attn_out, cross_attn_maps


class SACQCrossAttentionHead(nn.Module):
    """SACQ cross-attention prediction head:
    1) Generate a coarse future sequence using a flatten head.
    2) Reshape the coarse sequence into future patches and project each patch to d_model as dynamic queries.
    3) Add a global history bias from encoder memory.
    4) Refine the query tokens with stacked cross-attention/self-attention blocks.
    5) Add the coarse patch prediction back as a gated residual shortcut in patch space.
    """
    def __init__(
        self,
        n_vars,
        nf,
        d_model,
        target_window,
        patch_len,
        cross_attn_layers=1,
        n_heads=4,
        head_dropout=0.0,
        individual=False,
        query_pos_emb=None,
        return_attention_map: bool = False,
        patch_stride: Optional[int] = None,
        overlap_avg: bool = False,
        sacq_alpha_mode: str = "learnable",
        sacq_alpha_value: float = 0.5,
        sacq_drift_alpha_enable: bool = True,
        sacq_alpha_temperature: float = 0.5,
        sacq_alpha_granularity: str = "patch",
        sacq_alpha_softplus: bool = False,
        sacq_mem_film_enable: bool = False,
        sacq_mem_film_hidden: int = 64,
        sacq_mem_film_scale: float = 0.25,
        sacq_grad_firewall_enable: bool = False,
        sacq_grad_firewall_hidden: int = 64,
        sacq_grad_firewall_temp: float = 1.0,
        sacq_grad_firewall_mode: str = "learnable",
        dsacq_enable: bool = True,
        dsacq_mem_gate_enable: bool = True,
        dsacq_drift_gate_enable: bool = True,
        dsacq_mem_gate_hidden: int = 32,
        dsacq_mem_gate_mode: str = 'stats',
        mem_shift_aug_std: float = 0.15,
        mem_shift_aug_prob: float = 0.5,
        align_ablate_shift: bool = False,
        align_ablate_dual_bank: bool = False,
        align_ablate_token_gate: bool = False,
        align_ablate_shift_drift: bool = False,
        sacq_autoregressive: bool = False,
        sacq_future_causal: bool = False,
        sacq_ar_verify: bool = False,
        sacq_coarse_init: str = 'flatten',
        sacq_routing_topk: int = 0,
        sacq_routing_window: int = 0,
        sacq_scheme: str = 'default',
        sacq_dual_struct_gate: bool = False,
        sacq_dual_res_fuse: bool = False,
        sacq_dual_detach: bool = True,
        sacq_dual_scaffold_ln: bool = False,
        sacq_pe_mode: str = 'default',
        hist_patch_num: int = 0,
        timeline_pos_emb: Optional[nn.Parameter] = None,
        cont_hist_pos_emb: Optional[nn.Parameter] = None,
        cont_pe_delta: Optional[nn.Parameter] = None,
        sacq_pe_rel_buckets: int = 32,
    ):
        super().__init__()
        valid_pe_modes = ('default', 'shared', 'continuation', 'relative_bias')
        if sacq_pe_mode not in valid_pe_modes:
            raise ValueError(f"sacq_pe_mode must be one of {valid_pe_modes}, got {sacq_pe_mode!r}.")
        self.sacq_pe_mode = sacq_pe_mode
        self.hist_patch_num = int(hist_patch_num)
        self.timeline_pos_emb = timeline_pos_emb
        self.cont_hist_pos_emb = cont_hist_pos_emb
        self.cont_pe_delta = cont_pe_delta
        valid_schemes = {'default', 'dual_residual', 'template_ar'}
        if sacq_scheme not in valid_schemes:
            raise ValueError(f"sacq_scheme must be one of {sorted(valid_schemes)}, got {sacq_scheme!r}.")
        self.sacq_scheme = sacq_scheme
        self.sacq_autoregressive = bool(sacq_autoregressive)
        self.sacq_future_causal = bool(sacq_future_causal) or self.sacq_autoregressive
        self.sacq_ar_verify = bool(sacq_ar_verify)
        valid_coarse_inits = ('flatten', 'routing', 'routing_blend')
        if sacq_coarse_init not in valid_coarse_inits:
            raise ValueError(
                f"sacq_coarse_init must be one of {sorted(valid_coarse_inits)}, got {sacq_coarse_init!r}."
            )
        self.sacq_coarse_init = sacq_coarse_init
        self.sacq_routing_topk = max(0, int(sacq_routing_topk))
        self.sacq_routing_window = max(0, int(sacq_routing_window))
        self.sacq_dual_struct_gate = bool(sacq_dual_struct_gate)
        self.sacq_dual_res_fuse = bool(sacq_dual_res_fuse)
        self.sacq_dual_detach = bool(sacq_dual_detach)
        self.sacq_dual_scaffold_ln = bool(sacq_dual_scaffold_ln)
        self.n_vars = n_vars
        self.nf = nf
        self.d_model = d_model
        self.target_window = target_window
        self.patch_len = int(patch_len)
        self.patch_stride = int(self.patch_len if patch_stride is None else patch_stride)
        if self.patch_stride < 1:
            raise ValueError(f"patch_stride must be >= 1, got {self.patch_stride}.")
        self.overlap_avg = bool(overlap_avg)
        if self.overlap_avg:
            self.out_patch_stride = self.patch_stride
            if target_window <= self.patch_len:
                self.out_patch_num = 1
            else:
                self.out_patch_num = int(np.ceil((target_window - self.patch_len) / self.out_patch_stride)) + 1
        else:
            self.out_patch_stride = self.patch_len
            self.out_patch_num = int(np.ceil(target_window / self.patch_len))
        self.cross_attn_layers = int(cross_attn_layers)
        self.individual = individual
        self.return_attention_map = bool(return_attention_map)
        self.query_pos_emb = query_pos_emb
        self.dsacq_enable = bool(dsacq_enable)
        self.dsacq_mem_gate = bool(dsacq_enable) and bool(dsacq_mem_gate_enable)
        self.dsacq_drift_gate = bool(dsacq_enable) and bool(dsacq_drift_gate_enable)
        self.dsacq_mem_gate_hidden = max(8, int(dsacq_mem_gate_hidden))
        if dsacq_mem_gate_mode not in ('stats', 'e2e', 'align'):
            raise ValueError(
                f"dsacq_mem_gate_mode must be 'stats', 'e2e', or 'align', got {dsacq_mem_gate_mode!r}."
            )
        self.dsacq_mem_gate_mode = dsacq_mem_gate_mode if self.dsacq_mem_gate else 'stats'
        self.sacq_grad_firewall_enable = bool(sacq_grad_firewall_enable)
        self.sacq_grad_firewall_temp = max(0.05, float(sacq_grad_firewall_temp))
        self.sacq_grad_firewall_hidden = max(8, int(sacq_grad_firewall_hidden))
        if sacq_grad_firewall_mode not in ('learnable', 'fixed_stat'):
            raise ValueError(
                f"sacq_grad_firewall_mode must be 'learnable' or 'fixed_stat', got {sacq_grad_firewall_mode!r}."
            )
        self.sacq_grad_firewall_mode = sacq_grad_firewall_mode
        self.mem_shift_aug_std = float(mem_shift_aug_std)
        self.mem_shift_aug_prob = float(mem_shift_aug_prob)
        self.mem_recent_frac = 0.25
        self.align_ablate_shift = bool(align_ablate_shift)
        self.align_ablate_dual_bank = bool(align_ablate_dual_bank)
        self.align_ablate_token_gate = bool(align_ablate_token_gate)
        self.align_ablate_shift_drift = bool(align_ablate_shift_drift)
        if self.cross_attn_layers < 0:
            raise ValueError(f"cross_attn_layers must be >= 0, got {self.cross_attn_layers}.")

        self.query_window = (self.out_patch_num - 1) * self.out_patch_stride + self.patch_len
        coarse_flat_len = self.out_patch_num * self.patch_len
        self.query_flatten_head = Flatten_Head(
            individual=individual,
            n_vars=n_vars,
            nf=nf,
            target_window=coarse_flat_len,
            head_dropout=head_dropout
        )
        self.coarse_routing = None
        self.coarse_blend_logit = None
        if self.sacq_coarse_init in ('routing', 'routing_blend'):
            self.coarse_routing = HistoryFutureRoutingCoarse(
                d_model=d_model,
                patch_len=self.patch_len,
                out_patch_num=self.out_patch_num,
                hist_patch_num=self.hist_patch_num,
                topk=self.sacq_routing_topk,
                local_window=self.sacq_routing_window,
                dropout=head_dropout,
            )
            if self.sacq_coarse_init == 'routing_blend':
                self.coarse_blend_logit = nn.Parameter(torch.tensor(0.0))
        self.query_patch_proj = nn.Linear(self.patch_len, d_model)
        self.base_q_out_proj = nn.Linear(d_model, self.patch_len)
        self.global_proj = nn.Linear(d_model, d_model)
        self.global_gate = nn.Parameter(torch.tensor(0.0))

        self.out_dropout = nn.Dropout(head_dropout)
        relative_time_bias = None
        self.relative_time_bias = None
        if self.sacq_pe_mode == 'relative_bias':
            self.relative_time_bias = RelativeTimeBias(
                n_heads=n_heads,
                hist_patch_num=self.hist_patch_num,
                num_buckets=sacq_pe_rel_buckets,
            )
        dec_kw = dict(
            d_model=d_model,
            patch_len=patch_len,
            cross_attn_layers=self.cross_attn_layers,
            n_heads=n_heads,
            head_dropout=head_dropout,
            relative_time_bias=self.relative_time_bias,
            ar_verify=self.sacq_ar_verify,
        )
        dec_causal = self.sacq_future_causal
        self.dual_struct_logit = None
        self.scaffold_ln = None
        self.res_fuse_logit = None
        self.res_q_norm = None
        if self.sacq_scheme == 'default':
            self.decoder = SACQDecoderStack(**dec_kw, force_causal=dec_causal)
            self.struct_decoder = None
            self.res_decoder = None
            self.template_causal_self = None
            self.template_cross = None
            self.hist_patch_proj = None
            self.template_q_norm = None
        elif self.sacq_scheme == 'dual_residual':
            self.decoder = None
            self.struct_decoder = SACQDecoderStack(**dec_kw, force_causal=True)
            self.res_decoder = SACQDecoderStack(**dec_kw, force_causal=dec_causal)
            self.template_causal_self = None
            self.template_cross = None
            self.hist_patch_proj = None
            self.template_q_norm = None
            self.dual_struct_logit = (
                nn.Parameter(torch.tensor(1.5)) if self.sacq_dual_struct_gate else None
            )
            self.scaffold_ln = (
                nn.LayerNorm(patch_len) if self.sacq_dual_scaffold_ln else None
            )
            self.res_fuse_logit = (
                nn.Parameter(torch.tensor(-3.0)) if self.sacq_dual_res_fuse else None
            )
            self.res_q_norm = (
                nn.LayerNorm(d_model) if self.sacq_dual_res_fuse else None
            )
        else:
            self.decoder = None
            self.struct_decoder = None
            self.res_decoder = SACQDecoderStack(**dec_kw, force_causal=dec_causal)
            self.template_q_norm = nn.LayerNorm(d_model)
            tmpl_heads = max(1, min(n_heads, d_model))
            self.template_causal_self = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=tmpl_heads,
                dropout=head_dropout,
                batch_first=True,
            )
            self.template_cross = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=head_dropout,
                batch_first=True,
            )
            self.hist_patch_proj = nn.Linear(d_model, patch_len)
        if self.dsacq_mem_gate and self.dsacq_mem_gate_mode == 'stats':
            self.mem_gate_mlp = nn.Sequential(
                nn.Linear(8, self.dsacq_mem_gate_hidden),
                nn.GELU(),
                nn.Linear(self.dsacq_mem_gate_hidden, self.out_patch_num),
            )
        else:
            self.mem_gate_mlp = None
        if self.dsacq_mem_gate and self.dsacq_mem_gate_mode == 'e2e':
            gate_heads = max(1, min(2, n_heads))
            self.mem_gate_cross = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=gate_heads,
                dropout=head_dropout,
                batch_first=True,
            )
            self.mem_gate_patch_mlp = nn.Sequential(
                nn.Linear(3 * d_model, self.dsacq_mem_gate_hidden),
                nn.GELU(),
                nn.Linear(self.dsacq_mem_gate_hidden, 1),
            )
            self.mem_gate_token_mlp = nn.Sequential(
                nn.Linear(2 * d_model, self.dsacq_mem_gate_hidden),
                nn.GELU(),
                nn.Linear(self.dsacq_mem_gate_hidden, 1),
            )
            self.shift_encoder = None
            self.mem_bank_mixer = None
            self.mem_gate_token_simple = None
        elif self.dsacq_mem_gate and self.dsacq_mem_gate_mode == 'align':
            gate_heads = max(1, min(2, n_heads))
            self.mem_gate_cross = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=gate_heads,
                dropout=head_dropout,
                batch_first=True,
            )
            shift_in_dim = d_model * 4 + 1
            self.shift_encoder = nn.Sequential(
                nn.Linear(shift_in_dim, self.dsacq_mem_gate_hidden),
                nn.GELU(),
                nn.Linear(self.dsacq_mem_gate_hidden, d_model),
            )
            self.shift_scalar_proj = nn.Linear(d_model, 1)
            self.mem_gate_patch_mlp = nn.Sequential(
                nn.Linear(3 * d_model, self.dsacq_mem_gate_hidden),
                nn.GELU(),
                nn.Linear(self.dsacq_mem_gate_hidden, 1),
            )
            self.mem_gate_token_mlp = nn.Sequential(
                nn.Linear(2 * d_model + 2, self.dsacq_mem_gate_hidden),
                nn.GELU(),
                nn.Linear(self.dsacq_mem_gate_hidden, 1),
            )
            self.mem_bank_mixer = nn.Sequential(
                nn.Linear(2 * d_model + 1, self.dsacq_mem_gate_hidden),
                nn.GELU(),
                nn.Linear(self.dsacq_mem_gate_hidden, 1),
            )
            self.mem_gate_token_simple = nn.Sequential(
                nn.Linear(2 * d_model, self.dsacq_mem_gate_hidden),
                nn.GELU(),
                nn.Linear(self.dsacq_mem_gate_hidden, 1),
            )
        else:
            self.mem_gate_cross = None
            self.mem_gate_patch_mlp = None
            self.mem_gate_token_mlp = None
            self.shift_encoder = None
            self.mem_bank_mixer = None
            self.shift_scalar_proj = None
            self.mem_gate_token_simple = None
        if not (self.dsacq_mem_gate and self.dsacq_mem_gate_mode == 'align'):
            self.shift_scalar_proj = None
            self.mem_gate_token_simple = None

        valid_alpha_modes = {"learnable", "fixed"}
        if sacq_alpha_mode not in valid_alpha_modes:
            raise ValueError(
                f"sacq_alpha_mode must be one of {sorted(valid_alpha_modes)}, got {sacq_alpha_mode!r}."
            )
        self.sacq_alpha_mode = sacq_alpha_mode
        self.sacq_alpha_value = float(sacq_alpha_value)
        self.sacq_drift_alpha_enable = bool(sacq_drift_alpha_enable)
        self.sacq_alpha_temperature = max(0.05, float(sacq_alpha_temperature))
        self.sacq_mem_film_enable = bool(sacq_mem_film_enable)
        self.sacq_mem_film_hidden = max(8, int(sacq_mem_film_hidden))
        self.sacq_mem_film_scale = max(0.0, float(sacq_mem_film_scale))
        if sacq_alpha_granularity not in ("patch", "step", "global"):
            raise ValueError(
                f"sacq_alpha_granularity must be 'patch', 'step', or 'global', "
                f"got {sacq_alpha_granularity!r}."
            )
        self.sacq_alpha_granularity = sacq_alpha_granularity
        self.sacq_alpha_softplus = bool(sacq_alpha_softplus)
        self._alpha_global = sacq_alpha_granularity == "global"
        self._alpha_per_step = sacq_alpha_granularity == "step"
        self._alpha_out_dim = patch_len if self._alpha_per_step else 1

        self.coarse_gate = nn.Parameter(torch.tensor(1.0))
        if self.sacq_mem_film_enable:
            self.mem_film_token_cond = nn.Sequential(
                nn.LayerNorm(2 * d_model),
                nn.Linear(2 * d_model, self.sacq_mem_film_hidden),
                nn.GELU(),
                nn.Linear(self.sacq_mem_film_hidden, 2 * d_model),
            )
            # Keep initial behavior near identity: small FiLM branch + small residual mix.
            nn.init.zeros_(self.mem_film_token_cond[-1].weight)
            nn.init.zeros_(self.mem_film_token_cond[-1].bias)
            self.mem_film_mix_logit = nn.Parameter(torch.tensor(-4.0))
        else:
            self.mem_film_token_cond = None
            self.mem_film_mix_logit = None
        if self.sacq_grad_firewall_enable and self.sacq_grad_firewall_mode == 'learnable':
            fw_in_dim = 3 * d_model + 1
            self.grad_firewall_mlp = nn.Sequential(
                nn.Linear(fw_in_dim, self.sacq_grad_firewall_hidden),
                nn.GELU(),
                nn.Linear(self.sacq_grad_firewall_hidden, 1),
            )
        else:
            self.grad_firewall_mlp = None

        if self.sacq_alpha_mode == "learnable":
            self.fusion_logit = nn.Parameter(torch.tensor(0.0))
            self.fusion_gain = nn.Parameter(torch.tensor(2.0))
            self.fusion_tau = 0.5
            if self._alpha_global:
                if self.sacq_alpha_softplus:
                    if self.sacq_alpha_value < 0.0:
                        raise ValueError(
                            f"sacq_alpha_value must be >= 0 when sacq_alpha_softplus=1, "
                            f"got {self.sacq_alpha_value}."
                        )
                    init_logit = _softplus_inv(self.sacq_alpha_value)
                else:
                    if not (0.0 < self.sacq_alpha_value <= 1.0):
                        raise ValueError(
                            f"sacq_alpha_value must be in (0, 1] when sacq_alpha_mode='learnable' "
                            f"and sacq_alpha_granularity='global', got {self.sacq_alpha_value}."
                        )
                    if self.sacq_alpha_value >= 1.0 - 1e-6:
                        init_logit = 15.0
                    else:
                        init_logit = _sigmoid_inv(self.sacq_alpha_value)
                self.global_alpha_logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))
                self.alpha_feat_norm = None
                self.alpha_proj = None
            else:
                if self._alpha_per_step:
                    alpha_in_dim = (
                        5 * patch_len + 1 if self.sacq_drift_alpha_enable else 2 * patch_len + 1
                    )
                else:
                    alpha_in_dim = 2 * patch_len + (4 if self.sacq_drift_alpha_enable else 1)
                alpha_hidden = max(16, patch_len)
                self.alpha_feat_norm = nn.LayerNorm(alpha_in_dim)
                self.alpha_proj = nn.Sequential(
                    nn.Linear(alpha_in_dim, alpha_hidden),
                    nn.GELU(),
                    nn.Linear(alpha_hidden, self._alpha_out_dim),
                )
                if self.sacq_alpha_softplus:
                    if self.sacq_alpha_value < 0.0:
                        raise ValueError(
                            f"sacq_alpha_value must be >= 0 when sacq_alpha_softplus=1, "
                            f"got {self.sacq_alpha_value}."
                        )
                    init_logit = _softplus_inv(self.sacq_alpha_value)
                else:
                    if not (0.0 <= self.sacq_alpha_value < 1.0):
                        raise ValueError(
                            f"sacq_alpha_value must be in [0, 1) when sacq_alpha_mode='learnable' "
                            f"(alpha init, sigmoid gate), got {self.sacq_alpha_value}."
                        )
                    if self.sacq_alpha_value < 1e-5:
                        init_logit = _sigmoid_inv(1e-4)
                    else:
                        init_logit = _sigmoid_inv(self.sacq_alpha_value)
                nn.init.xavier_uniform_(self.alpha_proj[0].weight)
                nn.init.zeros_(self.alpha_proj[0].bias)
                if self._alpha_out_dim > 1:
                    nn.init.xavier_uniform_(self.alpha_proj[2].weight, gain=0.1)
                else:
                    nn.init.xavier_uniform_(self.alpha_proj[2].weight)
                nn.init.constant_(self.alpha_proj[2].bias, init_logit)
                self.global_alpha_logit = None
            self.register_buffer("_fixed_alpha_buf", torch.empty(0))
        else:
            if self.sacq_alpha_softplus:
                if self.sacq_alpha_value < 0.0:
                    raise ValueError(
                        f"sacq_alpha_value must be >= 0 when sacq_alpha_softplus=1, "
                        f"got {self.sacq_alpha_value}."
                    )
            elif not (0.0 <= self.sacq_alpha_value <= 1.0):
                raise ValueError(f"sacq_alpha_value must be in [0, 1], got {self.sacq_alpha_value}.")
            self.fusion_logit = None
            self.fusion_gain = None
            self.fusion_tau = 0.5
            self.alpha_feat_norm = None
            self.alpha_proj = None
            self.global_alpha_logit = None
            self.register_buffer("_fixed_alpha_buf", torch.tensor([self.sacq_alpha_value]))

    def _alpha_gate(self, z: Tensor) -> Tensor:
        """Map logits to gate α: sigmoid (default, α∈(0,1)) or softplus (α>0, unbounded)."""
        z = z / self.sacq_alpha_temperature
        if self.sacq_alpha_softplus:
            return F.softplus(z)
        return torch.sigmoid(z)

    def _series_stats(self, series_2d):
        # series_2d: [B, T]
        mu = series_2d.mean(dim=1)
        var = series_2d.var(dim=1, unbiased=False)
        if series_2d.shape[1] > 1:
            slope = (series_2d[:, 1:] - series_2d[:, :-1]).abs().mean(dim=1)
        else:
            slope = torch.zeros_like(mu)

        spec = torch.fft.rfft(series_2d, dim=1).abs().pow(2)
        if spec.shape[1] > 1:
            spec = spec[:, 1:]
        spec = spec + 1e-8
        p = spec / spec.sum(dim=1, keepdim=True).clamp_min(1e-8)
        entropy = -(p * torch.log(p)).sum(dim=1)
        entropy = entropy / np.log(max(2, p.shape[1] + 1))
        return torch.stack([mu, var, slope, entropy], dim=1)

    def _compute_memory_gate_stats(self, kv, coarse_patch):
        # kv: [B, P, D], coarse_patch: [B, O, patch_len]
        hist_series = kv.mean(dim=-1)
        coarse_series = coarse_patch.mean(dim=-1)
        stats_h = self._series_stats(hist_series)
        stats_f = self._series_stats(coarse_series)
        stats = torch.cat([stats_h, stats_f], dim=1)
        beta = torch.sigmoid(self.mem_gate_mlp(stats)).unsqueeze(-1)  # [B, O, 1]
        return beta, None

    def _compute_memory_gate_e2e(self, kv, base_q):
        # kv: [B, P, D], base_q: [B, O, D]
        cross_out, _ = self.mem_gate_cross(base_q, kv, kv, need_weights=False)
        hist_global = kv.mean(dim=1, keepdim=True).expand(-1, base_q.shape[1], -1)
        patch_feat = torch.cat([base_q, cross_out, hist_global], dim=-1)
        mem_beta = torch.sigmoid(self.mem_gate_patch_mlp(patch_feat))

        q_global = base_q.mean(dim=1, keepdim=True)
        token_feat = torch.cat([kv, q_global.expand(-1, kv.shape[1], -1)], dim=-1)
        token_beta = torch.sigmoid(self.mem_gate_token_mlp(token_feat))
        return mem_beta, token_beta, None

    def _compute_memory_gate_align(self, kv, base_q):
        # kv: [B, P, D], base_q: [B, O, D]
        batch_size, patch_num, dim = kv.shape
        out_patch_num = base_q.shape[1]
        k_recent = max(2, int(patch_num * self.mem_recent_frac))

        cross_full, attn_full = self.mem_gate_cross(
            base_q, kv, kv, need_weights=True, average_attn_weights=True
        )
        kv_recent = kv[:, -k_recent:, :]
        cross_recent, attn_recent = self.mem_gate_cross(
            base_q, kv_recent, kv_recent, need_weights=True, average_attn_weights=True
        )

        hist_global = kv.mean(dim=1)
        query_global = base_q.mean(dim=1)
        hist_recent = kv[:, -k_recent:, :].mean(dim=1)
        if self.align_ablate_shift:
            shift_embed = hist_global
            shift_scalar = kv.new_full((batch_size, 1), 0.5)
        else:
            cos_sim = F.cosine_similarity(hist_global, query_global, dim=-1, eps=1e-6).unsqueeze(-1)
            shift_raw = torch.cat(
                [
                    hist_global,
                    query_global,
                    (hist_global - query_global).abs(),
                    (hist_recent - hist_global).abs(),
                    cos_sim,
                ],
                dim=-1,
            )
            shift_embed = self.shift_encoder(shift_raw)
            shift_scalar = torch.sigmoid(self.shift_scalar_proj(shift_embed))

        if self.align_ablate_dual_bank:
            cross_blended = cross_full
            attn_combined = attn_full.mean(dim=1)
            attn_combined = attn_combined / attn_combined.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        else:
            bank_in = torch.cat([cross_full.mean(dim=1), cross_recent.mean(dim=1), shift_scalar], dim=-1)
            recent_weight = torch.sigmoid(self.mem_bank_mixer(bank_in))
            cross_blended = recent_weight.unsqueeze(1) * cross_recent + (1.0 - recent_weight.unsqueeze(1)) * cross_full

            attn_full_mean = attn_full.mean(dim=1)
            attn_recent_padded = kv.new_zeros(batch_size, patch_num)
            attn_recent_padded[:, -k_recent:] = attn_recent.mean(dim=1)
            attn_combined = (1.0 - recent_weight) * attn_full_mean + recent_weight * attn_recent_padded
            attn_combined = attn_combined / attn_combined.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        if self.align_ablate_token_gate:
            q_global = base_q.mean(dim=1, keepdim=True)
            token_feat = torch.cat([kv, q_global.expand(-1, patch_num, -1)], dim=-1)
            token_beta = torch.sigmoid(self.mem_gate_token_simple(token_feat))
        else:
            pos = torch.linspace(0.0, 1.0, patch_num, device=kv.device, dtype=kv.dtype)
            pos = pos.unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)
            token_feat = torch.cat(
                [
                    kv,
                    shift_embed.unsqueeze(1).expand(-1, patch_num, -1),
                    attn_combined.unsqueeze(-1),
                    pos,
                ],
                dim=-1,
            )
            token_beta = torch.sigmoid(self.mem_gate_token_mlp(token_feat))

        patch_feat = torch.cat(
            [base_q, cross_blended, shift_embed.unsqueeze(1).expand(-1, out_patch_num, -1)],
            dim=-1,
        )
        mem_beta = torch.sigmoid(self.mem_gate_patch_mlp(patch_feat))
        return mem_beta, token_beta, shift_scalar

    def _estimate_global_drift(self, coarse_patch, kv):
        coarse_center = coarse_patch.mean(dim=-1)
        hist_anchor = kv[:, -1:, :].mean(dim=-1)
        drift = (coarse_center - hist_anchor).abs().amax(dim=1, keepdim=True)
        return drift

    def _apply_coarse_conditioned_mem_film(self, kv: Tensor, base_q: Tensor):
        """Query-aware token FiLM on V only (keep K geometry unchanged)."""
        if not self.sacq_mem_film_enable or self.mem_film_token_cond is None:
            return kv, kv, None, None, None
        q_global = base_q.mean(dim=1).unsqueeze(1).expand(-1, kv.shape[1], -1)  # [B, P, D]
        tok_feat = torch.cat([kv, q_global], dim=-1)  # [B, P, 2D]
        film_raw = self.mem_film_token_cond(tok_feat)  # [B, P, 2D]
        gamma_raw, beta_raw = torch.chunk(film_raw, 2, dim=-1)
        scale = self.sacq_mem_film_scale
        gamma = 1.0 + scale * torch.tanh(gamma_raw)
        beta = scale * torch.tanh(beta_raw)
        v_mod = kv * gamma + beta
        mix = torch.sigmoid(self.mem_film_mix_logit).view(1, 1, 1)
        k_out = kv
        v_out = (1.0 - mix) * kv + mix * v_mod
        gamma_mean = gamma.mean(dim=1)
        beta_mean = beta.mean(dim=1)
        return k_out, v_out, mix.squeeze(), gamma_mean, beta_mean

    @staticmethod
    def _future_self_attn_mask(seq_len, device):
        if seq_len <= 1:
            return None
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def _apply_gradient_firewall(self, kv, base_q):
        if not self.sacq_grad_firewall_enable:
            return kv, None, None
        patch_num = kv.shape[1]
        q_global = base_q.mean(dim=1, keepdim=True).expand(-1, patch_num, -1)
        if self.sacq_grad_firewall_mode == 'fixed_stat':
            # Fixed-function gate: g = sigmoid((|K-Qbar|-mu)/tau), no trainable parameters.
            drift_mag = (kv - q_global).detach().abs().mean(dim=-1, keepdim=True)
            mu = drift_mag.mean(dim=1, keepdim=True)
            gate = torch.sigmoid((drift_mag - mu) / self.sacq_grad_firewall_temp)
        else:
            if self.grad_firewall_mlp is None:
                return kv, None, None
            drift = (kv - q_global).abs()
            mag = kv.abs().mean(dim=-1, keepdim=True)
            feat = torch.cat([kv, q_global, drift, mag], dim=-1)
            gate = torch.sigmoid(self.grad_firewall_mlp(feat) / self.sacq_grad_firewall_temp)
        kv_mix = (1.0 - gate) * kv + gate * kv.detach()
        return kv_mix, gate, gate.mean(dim=1)

    def _build_pos_q(self, patch_num: int, batch_n: int, device, dtype):
        if self.sacq_pe_mode == 'shared':
            if self.timeline_pos_emb is None:
                raise RuntimeError("timeline_pos_emb is required for sacq_pe_mode='shared'.")
            pos_end = patch_num + self.out_patch_num
            pos = self.timeline_pos_emb[patch_num:pos_end]
            return pos.unsqueeze(0).expand(batch_n, -1, -1).to(device=device, dtype=dtype)
        if self.sacq_pe_mode == 'continuation':
            if self.cont_hist_pos_emb is None or self.cont_pe_delta is None:
                raise RuntimeError("continuation PE parameters are missing.")
            pos = continuation_future_pe(
                self.cont_hist_pos_emb, self.out_patch_num, self.cont_pe_delta
            )
            return pos.unsqueeze(0).expand(batch_n, -1, -1).to(device=device, dtype=dtype)
        if self.query_pos_emb is not None:
            pos_start = patch_num
            pos_end = pos_start + self.out_patch_num
            if self.query_pos_emb.shape[0] < pos_end:
                raise ValueError(
                    f"query_pos_emb length ({self.query_pos_emb.shape[0]}) is smaller than required "
                    f"continuation end index ({pos_end})."
                )
            pos = self.query_pos_emb[pos_start:pos_end]
            return pos.unsqueeze(0).expand(batch_n, -1, -1).to(device=device, dtype=dtype)
        return None

    def _apply_shared_kv_pe(self, kv, patch_num: int):
        if self.sacq_pe_mode != 'shared' or self.timeline_pos_emb is None:
            return kv
        hist_pe = self.timeline_pos_emb[:patch_num]
        return kv + hist_pe.unsqueeze(0)

    def _compute_coarse_patches(self, x, kv):
        """Build coarse scaffold F in patch space: [B*nvars, O, patch_len]."""
        flat_q = self.query_flatten_head(x)
        flat_q_patch = flat_q.reshape(
            x.shape[0], self.n_vars, self.out_patch_num, self.patch_len
        )
        flat_bn = flat_q_patch.reshape(x.shape[0] * self.n_vars, self.out_patch_num, self.patch_len)
        if self.coarse_routing is None:
            return flat_bn
        route_bn = self.coarse_routing(kv)
        if self.coarse_blend_logit is None:
            return route_bn
        blend = torch.sigmoid(self.coarse_blend_logit)
        return (1.0 - blend) * flat_bn + blend * route_bn

    def _forward_dual_residual(self, q, kv, kv_value, flat_q_patch_bn, pos_q, mem_beta):
        struct_out, struct_maps = self.struct_decoder(
            q,
            kv,
            kv_value=kv_value,
            mem_beta=mem_beta,
            return_cross_attn=self.return_attention_map,
            scaffold_patch=flat_q_patch_bn,
        )
        if self.sacq_dual_struct_gate and self.dual_struct_logit is not None:
            struct_scale = torch.sigmoid(self.dual_struct_logit)
            scaffold_patch_bn = flat_q_patch_bn + struct_scale * struct_out
        else:
            scaffold_patch_bn = flat_q_patch_bn + struct_out
        if self.scaffold_ln is not None:
            scaffold_patch_bn = self.scaffold_ln(scaffold_patch_bn)
        scaffold_for_res = (
            scaffold_patch_bn.detach() if self.sacq_dual_detach else scaffold_patch_bn
        )
        scaffold_emb = self.query_patch_proj(scaffold_for_res)
        if self.sacq_dual_res_fuse:
            legacy_res_q = scaffold_emb + pos_q
            blend = torch.sigmoid(self.res_fuse_logit)
            res_q = (1.0 - blend) * legacy_res_q + blend * q
            if self.res_q_norm is not None:
                res_q = self.res_q_norm(res_q)
        else:
            res_q = scaffold_emb + pos_q
        attn_out, res_maps = self.res_decoder(
            res_q,
            kv,
            kv_value=kv_value,
            mem_beta=mem_beta,
            return_cross_attn=self.return_attention_map,
            scaffold_patch=scaffold_patch_bn,
        )
        cross_attn_maps = list(struct_maps) + list(res_maps)
        return attn_out, scaffold_patch_bn, cross_attn_maps

    def forward(self, x, query_time_emb: Optional[Tensor] = None):
        bs, nvars, d_model, patch_num = x.shape
        assert nvars == self.n_vars and d_model == self.d_model

        kv = x.permute(0, 1, 3, 2).contiguous().reshape(bs * nvars, patch_num, d_model)
        kv = self._apply_shared_kv_pe(kv, patch_num)

        v_global = kv.mean(dim=1)
        global_bias = self.global_proj(v_global).unsqueeze(1)

        flat_q_patch_bn = self._compute_coarse_patches(x, kv)
        flat_q_patch = flat_q_patch_bn.reshape(bs, nvars, self.out_patch_num, self.patch_len)
        base_q = self.query_patch_proj(flat_q_patch)
        base_q = base_q.reshape(bs * nvars, self.out_patch_num, d_model)
        if query_time_emb is not None:
            if query_time_emb.shape != base_q.shape:
                raise ValueError(
                    f"query_time_emb shape {tuple(query_time_emb.shape)} must match base_q shape {tuple(base_q.shape)}"
                )
            base_q = base_q + query_time_emb

        batch_n = bs * nvars
        pos_q = self._build_pos_q(patch_num, batch_n, base_q.device, base_q.dtype)
        if pos_q is None:
            q = base_q
        else:
            q = base_q + pos_q
        mem_film_mix = None
        mem_film_gamma = None
        mem_film_beta = None
        kv_value = kv
        mem_token_beta = None
        shift_scalar = None
        kv_for_aux = kv
        fw_gate_token = None
        fw_gate_patch = None
        if self.dsacq_mem_gate:
            kv_gate_in = kv
            if (
                self.dsacq_mem_gate_mode == 'align'
                and self.training
                and self.mem_shift_aug_std > 0.0
                and torch.rand((), device=kv.device) < self.mem_shift_aug_prob
            ):
                scale = 1.0 + torch.randn(bs * nvars, 1, 1, device=kv.device, dtype=kv.dtype) * 0.1
                shift = torch.randn(bs * nvars, 1, d_model, device=kv.device, dtype=kv.dtype) * self.mem_shift_aug_std
                kv_gate_in = kv * scale + shift
            if self.dsacq_mem_gate_mode == 'e2e':
                mem_beta, mem_token_beta, _ = self._compute_memory_gate_e2e(kv_gate_in, base_q)
                kv = kv * mem_token_beta
            elif self.dsacq_mem_gate_mode == 'align':
                mem_beta, mem_token_beta, shift_scalar = self._compute_memory_gate_align(kv_gate_in, base_q)
                kv = kv * mem_token_beta
            else:
                mem_beta, _ = self._compute_memory_gate_stats(kv, flat_q_patch_bn)
                mem_beta_global = mem_beta.mean(dim=1, keepdim=True)
                kv = kv * mem_beta_global
        else:
            mem_beta = flat_q_patch_bn.new_ones((bs * nvars, self.out_patch_num, 1))
        kv, fw_gate_token, fw_gate_patch = self._apply_gradient_firewall(kv, base_q)
        kv, kv_value, mem_film_mix, mem_film_gamma, mem_film_beta = (
            self._apply_coarse_conditioned_mem_film(kv, base_q)
        )
        self_attn_mask = None
        if self.sacq_future_causal and self.out_patch_num > 1:
            self_attn_mask = self._future_self_attn_mask(self.out_patch_num, kv.device)

        cross_attn_maps = []
        scaffold_patch_bn = flat_q_patch_bn
        if self.sacq_scheme == 'dual_residual':
            mb = mem_beta if self.dsacq_mem_gate else None
            attn_out, scaffold_patch_bn, cross_attn_maps = self._forward_dual_residual(
                q, kv, kv_value, flat_q_patch_bn, pos_q, mb
            )
        elif self.sacq_scheme == 'template_ar':
            tmpl_mask = self._future_self_attn_mask(self.out_patch_num, kv.device)
            q_norm = self.template_q_norm(q)
            q_causal, _ = self.template_causal_self(
                q_norm, q_norm, q_norm, attn_mask=tmpl_mask
            )
            _, tmpl_weights = self.template_cross(
                q_causal, kv, kv_value, need_weights=True, average_attn_weights=True
            )
            hist_patches = self.hist_patch_proj(kv_value)
            template_patch = torch.bmm(tmpl_weights, hist_patches)
            if self.return_attention_map:
                cross_attn_maps.append(tmpl_weights.detach())
            scaffold_patch_bn = template_patch
            res_q = self.query_patch_proj(template_patch.detach()) + pos_q
            attn_out, res_maps = self.res_decoder(
                res_q,
                kv,
                kv_value=kv_value,
                mem_beta=mem_beta if self.dsacq_mem_gate else None,
                return_cross_attn=self.return_attention_map,
                scaffold_patch=scaffold_patch_bn,
            )
            cross_attn_maps.extend(res_maps)
        elif self.cross_attn_layers == 0:
            attn_out = torch.zeros_like(flat_q_patch_bn)
        else:
            attn_out, cross_attn_maps = self.decoder(
                q,
                kv,
                kv_value=kv_value,
                mem_beta=mem_beta if self.dsacq_mem_gate else None,
                self_attn_mask=self_attn_mask,
                return_cross_attn=self.return_attention_map,
                scaffold_patch=flat_q_patch_bn,
            )
        drift_global = self._estimate_global_drift(scaffold_patch_bn, kv)
        if self.sacq_alpha_mode == "learnable":
            if self._alpha_global:
                a = self._alpha_gate(self.global_alpha_logit)
                alpha = a.view(1, 1, 1).expand(bs * nvars, self.out_patch_num, 1)
            else:
                if self.dsacq_drift_gate:
                    use_shift_drift = (
                        shift_scalar is not None
                        and self.dsacq_mem_gate_mode == 'align'
                        and not self.align_ablate_shift_drift
                    )
                    if use_shift_drift:
                        drift_feat = shift_scalar.unsqueeze(-1).expand(-1, self.out_patch_num, 1)
                    else:
                        drift_feat = drift_global.unsqueeze(-1).expand(-1, self.out_patch_num, 1)
                elif self.sacq_drift_alpha_enable:
                    drift_feat = drift_global.unsqueeze(-1).expand(-1, self.out_patch_num, 1)
                else:
                    drift_feat = flat_q_patch_bn.new_zeros((bs * nvars, self.out_patch_num, 1))
                if self.sacq_drift_alpha_enable:
                    if self._alpha_per_step:
                        corr_mag = attn_out.abs()
                        flat_mag = scaffold_patch_bn.abs().clamp_min(1e-6)
                        rel_energy = corr_mag / flat_mag
                        misalign = F.relu(-(scaffold_patch_bn * attn_out)) / flat_mag
                    else:
                        corr_mag = attn_out.abs().mean(dim=-1, keepdim=True)
                        flat_mag = scaffold_patch_bn.abs().mean(dim=-1, keepdim=True).clamp_min(1e-6)
                        rel_energy = corr_mag / flat_mag
                        misalign = F.relu(-(scaffold_patch_bn * attn_out).sum(dim=-1, keepdim=True))
                        misalign = misalign / scaffold_patch_bn.abs().sum(dim=-1, keepdim=True).clamp_min(1e-6)
                    alpha_feat = torch.cat(
                        [scaffold_patch_bn, attn_out, drift_feat, corr_mag, rel_energy, misalign],
                        dim=-1,
                    )
                else:
                    alpha_feat = torch.cat([scaffold_patch_bn, attn_out, drift_feat], dim=-1)
                alpha_feat = self.alpha_feat_norm(alpha_feat)
                alpha = self._alpha_gate(self.alpha_proj(alpha_feat))
        else:
            a = float(self._fixed_alpha_buf.item())
            alpha = flat_q_patch_bn.new_full(
                (bs * nvars, self.out_patch_num, self._alpha_out_dim), a
            )
        y_patch_bn = scaffold_patch_bn + alpha * attn_out

        flat_4d = scaffold_patch_bn.reshape(bs, nvars, self.out_patch_num, self.patch_len)
        corr_4d = (alpha * attn_out).reshape(bs, nvars, self.out_patch_num, self.patch_len)
        y_patch = y_patch_bn.reshape(bs, nvars, self.out_patch_num, self.patch_len)
        alpha_rs = alpha.reshape(bs, nvars, self.out_patch_num, -1)
        if alpha_rs.shape[-1] == 1:
            alpha_4d = alpha_rs.expand(-1, -1, -1, self.patch_len)
        else:
            alpha_4d = alpha_rs

        y = _merge_patch_predictions(y_patch, self.target_window, self.patch_len, self.out_patch_stride, self.overlap_avg)
        sacq_flat_seq = _merge_patch_predictions(flat_4d, self.target_window, self.patch_len, self.out_patch_stride, self.overlap_avg)
        sacq_corr_seq = _merge_patch_predictions(corr_4d, self.target_window, self.patch_len, self.out_patch_stride, self.overlap_avg)
        sacq_alpha_seq = _merge_patch_predictions(alpha_4d, self.target_window, self.patch_len, self.out_patch_stride, self.overlap_avg)

        base_q_patch = self.base_q_out_proj(base_q).reshape(bs, nvars, self.out_patch_num, self.patch_len)
        base_q_seq = _merge_patch_predictions(
            base_q_patch, self.target_window, self.patch_len, self.out_patch_stride, self.overlap_avg
        )

        ap = alpha.reshape(bs, nvars, self.out_patch_num, -1)
        sacq_alpha_patch = ap.mean(dim=-1) if ap.shape[-1] > 1 else ap.squeeze(-1)
        aux = {
            "base_q_seq": base_q_seq,
            "sacq_flat_seq": sacq_flat_seq,
            "sacq_corr_seq": sacq_corr_seq,
            "sacq_alpha_seq": sacq_alpha_seq,
            "sacq_alpha_patch": sacq_alpha_patch,
            "sacq_drift_scalar": drift_global.reshape(bs, nvars),
            "dsacq_mem_beta_mean": mem_beta.mean(dim=(1, 2)).reshape(bs, nvars),
        }
        if mem_film_mix is not None:
            aux["sacq_mem_film_mix"] = mem_film_mix.expand(bs, nvars)
            aux["sacq_mem_film_gamma_mean"] = mem_film_gamma.mean(dim=-1).reshape(bs, nvars)
            aux["sacq_mem_film_beta_mean"] = mem_film_beta.mean(dim=-1).reshape(bs, nvars)
        if fw_gate_token is not None and fw_gate_patch is not None:
            aux["sacq_fw_gate_token"] = fw_gate_token.reshape(bs, nvars, patch_num)
            aux["sacq_fw_gate_patch"] = fw_gate_patch.reshape(bs, nvars)
            aux["sacq_fw_gate_sample"] = fw_gate_patch.reshape(bs, nvars).mean(dim=1)
        if self.dsacq_mem_gate and self.dsacq_mem_gate_mode == 'align':
            aux["dsacq_kv_tokens"] = kv_for_aux.reshape(bs, nvars, patch_num, d_model)
            aux["dsacq_query_tokens"] = base_q.reshape(bs, nvars, self.out_patch_num, d_model)
            if shift_scalar is not None:
                aux["dsacq_shift_scalar"] = shift_scalar.reshape(bs, nvars, 1)
        if self.return_attention_map and cross_attn_maps:
            stacked_attn = torch.stack(cross_attn_maps, dim=1)
            stacked_attn = stacked_attn.view(
                bs, nvars, self.cross_attn_layers, stacked_attn.shape[2], stacked_attn.shape[3], stacked_attn.shape[4]
            )
            aux["sacq_cross_attn"] = stacked_attn
        return y, aux
