"""Training losses for long-horizon forecasting."""

from __future__ import annotations

import torch
import torch.nn as nn


class LogCoshLoss(nn.Module):
    """Mean log-cosh loss with a stable formulation (constant -log 2 omitted)."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        err = pred - target
        # |e| + log(1 + exp(-2|e|))  ==  log(cosh(e))  up to an additive constant
        return torch.mean(err.abs() + torch.log1p(torch.exp(-2.0 * err.abs())))


def _irse_robust_weight(abs_err: torch.Tensor, beta: torch.Tensor, eps: float) -> torch.Tensor:
    """Down-weight large residuals smoothly: w ~ 1 when |e| << beta, w ~ 0 when |e| >> beta."""
    ratio = abs_err / beta.clamp_min(eps)
    return 1.0 / (1.0 + ratio * ratio)


def estimate_beta_irse(
    abs_err: torch.Tensor,
    k: float = 1.0,
    n_iters: int = 3,
    eps: float = 1e-8,
    beta_min: float = 1e-6,
    beta_max: float = 1e6,
) -> torch.Tensor:
    """
    Iterative Robust Scale Estimation (IRSE).

    1) beta_0 = k * (median(|err|) + eps)
    2) repeat n_iters: w = 1/(1+(|err|/beta)^2), beta = sqrt(sum(w*err^2)/sum(w))
    """
    flat = abs_err.reshape(-1)
    beta = flat.median() + flat.new_tensor(eps)
    beta = flat.new_tensor(float(k)) * beta

    for _ in range(n_iters):
        w = _irse_robust_weight(flat, beta, eps)
        w_sum = w.sum().clamp_min(eps)
        beta = torch.sqrt((w * flat * flat).sum() / w_sum)
        beta = beta.clamp(beta_min, beta_max)

    return beta.detach()


class ScaledLogCoshLoss(nn.Module):
    """
    Scaled log-cosh loss: log(cosh((pred-target)/beta)) with a small-residual normalization.

    Motivation:
      - vanilla logcosh has no scale/transition parameter, so it may over/under-robust depending
        on the residual magnitude distribution of each dataset.
      - with the beta^2 factor, the local curvature around 0 matches MSE:
            logcosh(x) ~= x^2/2  ->  beta^2*logcosh(err/beta) ~= err^2/2

    Beta estimation:
      - fixed: use a constant beta (no extra overhead at inference).
      - auto_mad / irse: IRSE on batch residuals during training; EMA beta at eval/inference.
    """

    _AUTO_MODES = frozenset({"auto_mad", "irse"})

    def __init__(
        self,
        beta: float = 1.0,
        beta_mode: str = "auto_mad",
        beta_min: float = 1e-6,
        beta_max: float = 1e6,
        irse_iters: int = 3,
        irse_eps: float = 1e-8,
        beta_ema_momentum: float = 0.01,
    ):
        super().__init__()
        if beta <= 0:
            raise ValueError(f"`beta` must be > 0, got {beta}.")
        key = str(beta_mode).lower()
        if key not in ("auto_mad", "irse", "fixed"):
            raise ValueError(
                f"`beta_mode` must be one of ['auto_mad','irse','fixed'], got {beta_mode!r}."
            )
        if irse_iters < 1:
            raise ValueError(f"`irse_iters` must be >= 1, got {irse_iters}.")
        if not (0.0 < beta_ema_momentum <= 1.0):
            raise ValueError(
                f"`beta_ema_momentum` must be in (0, 1], got {beta_ema_momentum}."
            )

        self.beta = float(beta)  # In auto mode: multiplier k; in fixed mode: absolute beta.
        self.beta_mode = key
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.irse_iters = int(irse_iters)
        self.irse_eps = float(irse_eps)
        self.beta_ema_momentum = float(beta_ema_momentum)

        if self.beta_mode in self._AUTO_MODES:
            # Warm-start EMA; updated online during training, reused at eval/inference.
            self.register_buffer("beta_ema", torch.tensor(float(beta)))

    def _resolve_beta(self, abs_err_raw: torch.Tensor) -> torch.Tensor:
        if self.beta_mode == "fixed":
            return abs_err_raw.new_tensor(self.beta)

        if self.training:
            beta = estimate_beta_irse(
                abs_err_raw,
                k=self.beta,
                n_iters=self.irse_iters,
                eps=self.irse_eps,
                beta_min=self.beta_min,
                beta_max=self.beta_max,
            )
            with torch.no_grad():
                m = self.beta_ema_momentum
                self.beta_ema.mul_(1.0 - m).add_(beta.to(self.beta_ema.dtype) * m)
            return beta

        beta = self.beta_ema.to(device=abs_err_raw.device, dtype=abs_err_raw.dtype)
        return beta.clamp(self.beta_min, self.beta_max)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        err_raw = pred - target
        abs_err_raw = err_raw.abs()
        beta = self._resolve_beta(abs_err_raw)

        err = err_raw / beta
        abs_err = err.abs()
        # log(cosh(err)) up to a constant: |err| + log(1+exp(-2|err|))
        per_elem = abs_err + torch.log1p(torch.exp(-2.0 * abs_err))
        return torch.mean((beta**2) * per_elem)


def build_training_loss(
    name: str,
    beta: float = 1.0,
    beta_mode: str = "auto_mad",
    irse_iters: int = 3,
    beta_ema_momentum: float = 0.01,
) -> nn.Module:
    key = str(name).lower()
    if key == "mse":
        return nn.MSELoss()
    if key in ("logcosh", "log_cosh"):
        return LogCoshLoss()
    if key in ("logcosh_scaled", "scaled_logcosh", "log_cosh_scaled"):
        return ScaledLogCoshLoss(
            beta=beta,
            beta_mode=beta_mode,
            irse_iters=irse_iters,
            beta_ema_momentum=beta_ema_momentum,
        )
    raise ValueError(
        f"Unsupported training loss: {name!r}. Choose from ['mse', 'logcosh', 'logcosh_scaled']."
    )
