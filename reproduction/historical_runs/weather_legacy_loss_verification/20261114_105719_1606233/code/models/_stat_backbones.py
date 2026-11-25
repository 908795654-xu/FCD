import math

import torch
import torch.nn as nn


def _safe_len_norm(x):
    return max(int(x), 1)


class _BasePatchStatEncoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.patch_len = int(getattr(configs, "patch_len", 16))
        self.d_model = int(getattr(configs, "d_model", 128))
        self.dropout = nn.Dropout(float(getattr(configs, "dropout", 0.05)))
        self.out_proj = nn.Linear(self.patch_len * 3, self.d_model)
        self.norm = nn.LayerNorm(self.d_model)

    def _stack_feats(self, x, a, b):
        feat = torch.cat([x, a, b], dim=-1)
        feat = self.out_proj(feat)
        feat = self.dropout(feat)
        return self.norm(feat)


class ARIMAPatchEncoder(_BasePatchStatEncoder):
    """
    ARIMA-like tokenizer:
    - AR proxy: original patch.
    - I proxy: first-order differencing.
    - MA proxy: short moving average on differenced signal.
    """

    def __init__(self, configs):
        super().__init__(configs)
        self.ma_kernel = max(2, int(getattr(configs, "arima_ma_kernel", 3)))

    def forward(self, x):
        # x: [B, C, patch_len, patch_num]
        x = x.permute(0, 1, 3, 2)
        diff = torch.diff(x, dim=-1, prepend=x[..., :1])
        ma = diff.cumsum(dim=-1) / torch.arange(
            1, diff.shape[-1] + 1, device=diff.device, dtype=diff.dtype
        )
        return self._stack_feats(x, diff, ma).permute(0, 1, 3, 2)


class ETSPatchEncoder(_BasePatchStatEncoder):
    """
    ETS-like tokenizer:
    - level via exponential smoothing.
    - trend as level difference.
    - seasonal proxy via residuals against smoothed level.
    """

    def __init__(self, configs):
        super().__init__(configs)
        alpha = float(getattr(configs, "ets_alpha", 0.3))
        self.alpha = min(max(alpha, 1e-3), 0.999)

    def forward(self, x):
        x = x.permute(0, 1, 3, 2)
        level = torch.zeros_like(x)
        level[..., :1] = x[..., :1]
        for t in range(1, x.shape[-1]):
            level[..., t : t + 1] = self.alpha * x[..., t : t + 1] + (1 - self.alpha) * level[
                ..., t - 1 : t
            ]
        trend = torch.diff(level, dim=-1, prepend=level[..., :1])
        seasonal = x - level
        return self._stack_feats(level, trend, seasonal).permute(0, 1, 3, 2)


class ThetaPatchEncoder(_BasePatchStatEncoder):
    """
    Theta-like tokenizer:
    - theta0: linear trend line within patch.
    - theta2: doubled curvature term around trend.
    """

    def __init__(self, configs):
        super().__init__(configs)
        t = torch.linspace(0.0, 1.0, _safe_len_norm(self.patch_len))
        self.register_buffer("t", t.view(1, 1, 1, -1), persistent=False)

    def forward(self, x):
        x = x.permute(0, 1, 3, 2)
        t = self.t.to(device=x.device, dtype=x.dtype)
        x0 = x[..., :1]
        slope = x[..., -1:] - x0
        trend = x0 + slope * t
        theta2 = 2.0 * x - trend
        return self._stack_feats(x, trend, theta2).permute(0, 1, 3, 2)


class ProphetPatchEncoder(_BasePatchStatEncoder):
    """
    Prophet-like tokenizer:
    - piecewise-linear trend proxy.
    - Fourier seasonality basis.
    """

    def __init__(self, configs):
        super().__init__(configs)
        self.k = max(1, int(getattr(configs, "prophet_fourier_k", 3)))
        t = torch.linspace(0.0, 1.0, _safe_len_norm(self.patch_len))
        self.register_buffer("t", t.view(1, 1, 1, -1), persistent=False)
        basis = []
        for i in range(1, self.k + 1):
            basis.append(torch.sin(2 * math.pi * i * t))
            basis.append(torch.cos(2 * math.pi * i * t))
        self.register_buffer("fourier", torch.stack(basis, dim=0), persistent=False)
        self.season_proj = nn.Linear(self.k * 2, 1)

    def forward(self, x):
        x = x.permute(0, 1, 3, 2)
        t = self.t.to(device=x.device, dtype=x.dtype)
        fourier = self.fourier.to(device=x.device, dtype=x.dtype)

        x0 = x[..., :1]
        slope = x[..., -1:] - x0
        trend = x0 + slope * t

        season_feat = fourier.permute(1, 0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        season_feat = season_feat.expand(x.shape[0], x.shape[1], x.shape[2], -1, -1)
        season = self.season_proj(season_feat).squeeze(-1)
        season = season * x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        resid = x - trend - season
        return self._stack_feats(trend, season, resid).permute(0, 1, 3, 2)


class NBEATSPatchEncoder(nn.Module):
    """
    Lightweight N-BEATS style tokenizer:
    repeated fully-connected blocks build latent patch representation.
    """

    def __init__(self, configs):
        super().__init__()
        self.patch_len = int(getattr(configs, "patch_len", 16))
        self.d_model = int(getattr(configs, "d_model", 128))
        hidden = int(getattr(configs, "nbeats_hidden", 256))
        depth = max(2, int(getattr(configs, "nbeats_depth", 4)))
        dropout = float(getattr(configs, "dropout", 0.05))
        layers = [nn.Linear(self.patch_len, hidden), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout)]
        self.backbone = nn.Sequential(*layers)
        self.proj = nn.Linear(hidden, self.d_model)
        self.norm = nn.LayerNorm(self.d_model)

    def forward(self, x):
        # x: [B, C, patch_len, patch_num]
        x = x.permute(0, 1, 3, 2)
        z = self.backbone(x)
        z = self.proj(z)
        z = self.norm(z)
        return z.permute(0, 1, 3, 2)

