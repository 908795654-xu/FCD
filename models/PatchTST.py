# Adapted PatchTST wrapper for FCD heads; see NOTICE.md and LICENSES/PatchTST-Apache-2.0.txt.
__all__ = ['PatchTST']

from typing import Optional
import torch
from torch import nn
from torch import Tensor

from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp


class Model(nn.Module):
    def __init__(
        self,
        configs,
        max_seq_len: Optional[int] = 1024,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        norm: str = 'BatchNorm',
        attn_dropout: float = 0.0,
        act: str = "gelu",
        key_padding_mask: bool = 'auto',
        padding_var: Optional[int] = None,
        attn_mask: Optional[Tensor] = None,
        res_attention: bool = True,
        pre_norm: bool = False,
        store_attn: bool = False,
        pe: str = 'zeros',
        learn_pe: bool = True,
        head_type='flatten',
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__()

        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len

        n_layers = configs.e_layers
        cross_attn_layers = getattr(configs, 'cross_attn_layers', 1)
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        individual = configs.individual
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        affine_group_size = int(getattr(configs, 'affine_group_size', 0))
        sacq_alpha_mode = getattr(configs, 'sacq_alpha_mode', 'fixed')
        sacq_alpha_value = float(getattr(configs, 'sacq_alpha_value', 0.5))
        sacq_drift_alpha_enable = bool(int(getattr(configs, 'sacq_drift_alpha_enable', 1)))
        sacq_alpha_temperature = float(getattr(configs, 'sacq_alpha_temperature', 0.5))
        sacq_alpha_granularity = getattr(configs, 'sacq_alpha_granularity', 'patch')
        sacq_alpha_softplus = bool(int(getattr(configs, 'sacq_alpha_softplus', 0)))
        sacq_mem_film_enable = bool(int(getattr(configs, 'sacq_mem_film_enable', 0)))
        sacq_mem_film_hidden = int(getattr(configs, 'sacq_mem_film_hidden', 64))
        sacq_mem_film_scale = float(getattr(configs, 'sacq_mem_film_scale', 0.25))
        sacq_grad_firewall_enable = bool(int(getattr(configs, 'sacq_grad_firewall_enable', 0)))
        sacq_grad_firewall_hidden = int(getattr(configs, 'sacq_grad_firewall_hidden', 64))
        sacq_grad_firewall_temp = float(getattr(configs, 'sacq_grad_firewall_temp', 1.0))
        sacq_grad_firewall_mode = getattr(configs, 'sacq_grad_firewall_mode', 'learnable')
        sacq_overlap_avg = bool(getattr(configs, 'sacq_overlap_avg', 0))
        dsacq_enable = bool(getattr(configs, 'dsacq_enable', 1))
        dsacq_mem_gate_enable = bool(getattr(configs, 'dsacq_mem_gate_enable', 1))
        dsacq_drift_gate_enable = bool(getattr(configs, 'dsacq_drift_gate_enable', 1))
        dsacq_mem_gate_hidden = int(getattr(configs, 'dsacq_mem_gate_hidden', 32))
        dsacq_mem_gate_mode = getattr(configs, 'dsacq_mem_gate_mode', 'stats')
        mem_shift_aug_std = float(getattr(configs, 'mem_shift_aug_std', 0.15))
        mem_shift_aug_prob = float(getattr(configs, 'mem_shift_aug_prob', 0.5))
        align_ablate_shift = bool(int(getattr(configs, 'align_ablate_shift', 0)))
        align_ablate_dual_bank = bool(int(getattr(configs, 'align_ablate_dual_bank', 0)))
        align_ablate_token_gate = bool(int(getattr(configs, 'align_ablate_token_gate', 0)))
        align_ablate_shift_drift = bool(int(getattr(configs, 'align_ablate_shift_drift', 0)))
        save_attention_map = bool(getattr(configs, 'save_attention_map', 0))
        sacq_autoregressive = bool(int(getattr(configs, 'sacq_autoregressive', 0)))
        sacq_future_causal = bool(int(getattr(configs, 'sacq_future_causal', 0)))
        sacq_ar_verify = bool(int(getattr(configs, 'sacq_ar_verify', 0)))
        sacq_coarse_init = getattr(configs, 'sacq_coarse_init', 'flatten')
        sacq_routing_topk = int(getattr(configs, 'sacq_routing_topk', 0))
        sacq_routing_window = int(getattr(configs, 'sacq_routing_window', 0))
        sacq_scheme = getattr(configs, 'sacq_scheme', 'default')
        sacq_dual_struct_gate = bool(int(getattr(configs, 'sacq_dual_struct_gate', 0)))
        sacq_dual_res_fuse = bool(int(getattr(configs, 'sacq_dual_res_fuse', 0)))
        sacq_dual_detach = bool(int(getattr(configs, 'sacq_dual_detach', 1)))
        sacq_dual_scaffold_ln = bool(int(getattr(configs, 'sacq_dual_scaffold_ln', 0)))
        sacq_pe_mode = getattr(configs, 'sacq_pe_mode', 'default')
        sacq_pe_rel_buckets = int(getattr(configs, 'sacq_pe_rel_buckets', 32))
        # Temporarily hard-disable time-token injection for stable baseline runs.
        sacq_time_token_enable = False
        sacq_time_token_hidden = int(getattr(configs, 'sacq_time_token_hidden', 64))
        sacq_time_token_dropout = float(getattr(configs, 'sacq_time_token_dropout', 0.0))
        sacq_time_token_pool = getattr(configs, 'sacq_time_token_pool', 'last')
        kernel_size = configs.kernel_size
        dsacq_coarse_split = bool(int(getattr(configs, 'dsacq_coarse_split', 1)))
        dsacq_decomp_kernel = int(getattr(configs, 'dsacq_decomp_kernel', kernel_size))
        dsacq_level_slope_k = int(getattr(configs, 'dsacq_level_slope_k', kernel_size))
        decomposition = configs.decomposition
        head_type = kwargs.pop('head_type', getattr(configs, 'head_type', head_type))

        backbone_kw = dict(
            c_in=c_in,
            context_window=context_window,
            target_window=target_window,
            patch_len=patch_len,
            stride=stride,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            cross_attn_layers=cross_attn_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            act=act,
            key_padding_mask=key_padding_mask,
            padding_var=padding_var,
            attn_mask=attn_mask,
            res_attention=res_attention,
            pre_norm=pre_norm,
            store_attn=store_attn,
            pe=pe,
            learn_pe=learn_pe,
            fc_dropout=fc_dropout,
            head_dropout=head_dropout,
            padding_patch=padding_patch,
            head_type=head_type,
            individual=individual,
            revin=revin,
            affine=affine,
            subtract_last=subtract_last,
            affine_group_size=affine_group_size,
            sacq_alpha_mode=sacq_alpha_mode,
            sacq_alpha_value=sacq_alpha_value,
            sacq_drift_alpha_enable=sacq_drift_alpha_enable,
            sacq_alpha_temperature=sacq_alpha_temperature,
            sacq_alpha_granularity=sacq_alpha_granularity,
            sacq_alpha_softplus=sacq_alpha_softplus,
            sacq_mem_film_enable=sacq_mem_film_enable,
            sacq_mem_film_hidden=sacq_mem_film_hidden,
            sacq_mem_film_scale=sacq_mem_film_scale,
            sacq_grad_firewall_enable=sacq_grad_firewall_enable,
            sacq_grad_firewall_hidden=sacq_grad_firewall_hidden,
            sacq_grad_firewall_temp=sacq_grad_firewall_temp,
            sacq_grad_firewall_mode=sacq_grad_firewall_mode,
            sacq_overlap_avg=sacq_overlap_avg,
            dsacq_enable=dsacq_enable,
            dsacq_mem_gate_enable=dsacq_mem_gate_enable,
            dsacq_drift_gate_enable=dsacq_drift_gate_enable,
            dsacq_mem_gate_hidden=dsacq_mem_gate_hidden,
            dsacq_mem_gate_mode=dsacq_mem_gate_mode,
            mem_shift_aug_std=mem_shift_aug_std,
            mem_shift_aug_prob=mem_shift_aug_prob,
            align_ablate_shift=align_ablate_shift,
            align_ablate_dual_bank=align_ablate_dual_bank,
            align_ablate_token_gate=align_ablate_token_gate,
            align_ablate_shift_drift=align_ablate_shift_drift,
            save_attention_map=save_attention_map,
            sacq_autoregressive=sacq_autoregressive,
            sacq_future_causal=sacq_future_causal,
            sacq_ar_verify=sacq_ar_verify,
            sacq_coarse_init=sacq_coarse_init,
            sacq_routing_topk=sacq_routing_topk,
            sacq_routing_window=sacq_routing_window,
            sacq_scheme=sacq_scheme,
            sacq_dual_struct_gate=sacq_dual_struct_gate,
            sacq_dual_res_fuse=sacq_dual_res_fuse,
            sacq_dual_detach=sacq_dual_detach,
            sacq_dual_scaffold_ln=sacq_dual_scaffold_ln,
            sacq_pe_mode=sacq_pe_mode,
            sacq_pe_rel_buckets=sacq_pe_rel_buckets,
            sacq_time_token_enable=sacq_time_token_enable,
            sacq_time_token_hidden=sacq_time_token_hidden,
            sacq_time_token_dropout=sacq_time_token_dropout,
            sacq_time_token_pool=sacq_time_token_pool,
            dsacq_coarse_split=dsacq_coarse_split,
            dsacq_decomp_kernel=dsacq_decomp_kernel,
            dsacq_level_slope_k=dsacq_level_slope_k,
            verbose=verbose,
            **kwargs,
        )

        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(**backbone_kw)
            self.model_res = PatchTST_backbone(**backbone_kw)
        else:
            self.model = PatchTST_backbone(**backbone_kw)

    def forward(self, x, batch_x_mark=None, batch_y_mark=None):
        x_hist = x
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
            res = self.model_res(res_init, x_mark=batch_x_mark)
            trend = self.model_trend(trend_init, x_mark=batch_x_mark)
            if isinstance(res, tuple):
                res = res[0]
            if isinstance(trend, tuple):
                trend = trend[0]
            out = (res + trend).permute(0, 2, 1)
            return out

        x = x.permute(0, 2, 1)
        out = self.model(x, x_mark=batch_x_mark)
        if isinstance(out, tuple):
            pred, aux = out
            return pred.permute(0, 2, 1), aux
        return out.permute(0, 2, 1)

    def get_aux_loss(self):
        return torch.tensor(0.0, device=next(self.parameters()).device)
