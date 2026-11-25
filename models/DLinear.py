# DLinear-derived model with FCD-head integration; DLinear Authors (2022), Apache-2.0; see NOTICE.md.
import torch
import torch.nn as nn
from layers.PatchTST_backbone import SACQCrossAttentionHead
from layers.RevIN import RevIN


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        raw_ht = getattr(configs, 'head_type', 'flatten')
        self.head_type = 'flatten' if raw_ht in ('linear',) else raw_ht
        if self.head_type not in ('flatten', 'sacq_cross_attention'):
            raise ValueError(
                f"DLinear unsupported head_type={self.head_type!r}; use 'flatten' or 'sacq_cross_attention'."
            )

        kernel_size = int(getattr(configs, 'kernel_size', 25))
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in
        self.use_sacq_head = self.head_type == 'sacq_cross_attention'

        self._revin = bool(getattr(configs, "revin", 0))
        self._affine = bool(getattr(configs, "affine", 0))
        self._subtract_last = bool(getattr(configs, "subtract_last", 0))
        self._stride = int(getattr(configs, "stride", 8))
        self._padding_patch = getattr(configs, "padding_patch", None)

        self.revin_layer = None
        if self._revin:
            self.revin_layer = RevIN(
                self.channels,
                affine=self._affine,
                subtract_last=self._subtract_last,
                max_steps=max(self.seq_len, self.pred_len),
            )

        if self.use_sacq_head:
            self.patch_len = int(getattr(configs, 'patch_len', min(16, self.seq_len)))
            self.patch_len = max(1, min(self.patch_len, self.seq_len))
            self.stride = int(getattr(configs, 'stride', self.patch_len))
            self.stride = max(1, self.stride)
            self.padding_patch = getattr(configs, 'padding_patch', None)
            self.d_model = int(getattr(configs, 'd_model', 128))
            self.n_heads = int(getattr(configs, 'n_heads', 4))
            self.cross_attn_layers = int(getattr(configs, 'cross_attn_layers', 1))
            self.head_dropout = float(getattr(configs, 'head_dropout', 0.0))
            self.save_attention_map = bool(getattr(configs, 'save_attention_map', False))
            self.sacq_overlap_avg = bool(getattr(configs, 'sacq_overlap_avg', 0))
            self.sacq_alpha_mode = getattr(configs, 'sacq_alpha_mode', 'learnable')
            self.sacq_alpha_value = float(getattr(configs, 'sacq_alpha_value', 0.5))
            self.dsacq_enable = bool(getattr(configs, 'dsacq_enable', 1))
            self.dsacq_mem_gate_enable = bool(getattr(configs, 'dsacq_mem_gate_enable', 1))
            self.dsacq_drift_gate_enable = bool(getattr(configs, 'dsacq_drift_gate_enable', 1))
            self.dsacq_mem_gate_hidden = int(getattr(configs, 'dsacq_mem_gate_hidden', 32))
            self.dsacq_mem_gate_mode = getattr(configs, 'dsacq_mem_gate_mode', 'stats')
            self.mem_shift_aug_std = float(getattr(configs, 'mem_shift_aug_std', 0.15))
            self.mem_shift_aug_prob = float(getattr(configs, 'mem_shift_aug_prob', 0.5))

            if self.padding_patch == 'end':
                self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))

            seq_for_patch = self.seq_len + (self.stride if self.padding_patch == 'end' else 0)
            self.in_patch_num = int((seq_for_patch - self.patch_len) / self.stride + 1)
            self.in_patch_num = max(1, self.in_patch_num)
            head_nf = self.d_model * self.in_patch_num

            self.token_proj_seasonal = nn.Linear(self.patch_len, self.d_model)
            self.token_proj_trend = nn.Linear(self.patch_len, self.d_model)

            sacq_kw = dict(
                n_vars=self.channels,
                nf=head_nf,
                d_model=self.d_model,
                target_window=self.pred_len,
                patch_len=self.patch_len,
                cross_attn_layers=self.cross_attn_layers,
                n_heads=self.n_heads,
                head_dropout=self.head_dropout,
                individual=self.individual,
                query_pos_emb=None,
                return_attention_map=self.save_attention_map,
                patch_stride=self.stride,
                overlap_avg=self.sacq_overlap_avg,
                sacq_alpha_mode=self.sacq_alpha_mode,
                sacq_alpha_value=self.sacq_alpha_value,
                dsacq_enable=self.dsacq_enable,
                dsacq_mem_gate_enable=self.dsacq_mem_gate_enable,
                dsacq_drift_gate_enable=self.dsacq_drift_gate_enable,
                dsacq_mem_gate_hidden=self.dsacq_mem_gate_hidden,
                dsacq_mem_gate_mode=self.dsacq_mem_gate_mode,
                mem_shift_aug_std=self.mem_shift_aug_std,
                mem_shift_aug_prob=self.mem_shift_aug_prob,
            )
            self.SCAQ_Seasonal = SACQCrossAttentionHead(**sacq_kw)
            self.SCAQ_Trend = SACQCrossAttentionHead(**sacq_kw)
        else:
            if self.individual:
                self.Linear_Seasonal = nn.ModuleList()
                self.Linear_Trend = nn.ModuleList()
                for _ in range(self.channels):
                    self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                    self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
            else:
                self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
                self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def _build_sacq_tokens(self, x, token_proj):
        if self.padding_patch == 'end':
            x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = token_proj(x)
        return x.permute(0, 1, 3, 2).contiguous()

    def forward(self, x):
        if self.revin_layer is not None:
            z = x.permute(0, 2, 1)
            z = self.revin_layer(z.permute(0, 2, 1), "norm").permute(0, 2, 1)

        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)

        if self.use_sacq_head:
            seasonal_output, seasonal_aux = self.SCAQ_Seasonal(
                self._build_sacq_tokens(seasonal_init, self.token_proj_seasonal)
            )
            trend_output, trend_aux = self.SCAQ_Trend(
                self._build_sacq_tokens(trend_init, self.token_proj_trend)
            )
            out = seasonal_output + trend_output
            aux = {}
            if isinstance(seasonal_aux, dict) and 'sacq_cross_attn' in seasonal_aux:
                aux['sacq_cross_attn'] = seasonal_aux['sacq_cross_attn']
            elif isinstance(trend_aux, dict) and 'sacq_cross_attn' in trend_aux:
                aux['sacq_cross_attn'] = trend_aux['sacq_cross_attn']
            out = out.permute(0, 2, 1)
            if self.revin_layer is not None:
                out = self.revin_layer(out, "denorm")
            return out, aux

        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                dtype=seasonal_init.dtype,
                device=seasonal_init.device,
            )
            trend_output = torch.zeros_like(seasonal_output)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        out = (seasonal_output + trend_output).permute(0, 2, 1)
        if self.revin_layer is not None:
            out = self.revin_layer(out, "denorm")
        return out
