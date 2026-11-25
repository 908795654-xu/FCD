"""Training losses for long-horizon forecasting."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogCoshLoss(nn.Module):
    """Mean log-cosh loss with a stable formulation (constant -log 2 omitted)."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        err = pred - target
        # |e| + log(1 + exp(-2|e|))  ==  log(cosh(e))  up to an additive constant
        return torch.mean(err.abs() + torch.log1p(torch.exp(-2.0 * err.abs())))


class ScaledLogCoshLoss(nn.Module):
    """
    Scaled log-cosh loss: log(cosh((pred-target)/beta)) with a small-residual normalization.

    Motivation:
      - vanilla logcosh has no scale/transition parameter, so it may over/under-robust depending
        on the residual magnitude distribution of each dataset.
      - with the beta^2 factor, the local curvature around 0 matches MSE:
            logcosh(x) ~= x^2/2  ->  beta^2*logcosh(err/beta) ~= err^2/2
    """

    def __init__(
        self,
        beta: float = 1.0,
        beta_mode: str = "auto_mad",
        beta_min: float = 1e-6,
        beta_max: float = 1e6,
    ):
        super().__init__()
        if beta <= 0:
            raise ValueError(f"`beta` must be > 0, got {beta}.")
        key = str(beta_mode).lower()
        if key not in ("auto_mad", "fixed"):
            raise ValueError(f"`beta_mode` must be one of ['auto_mad','fixed'], got {beta_mode!r}.")
        self.beta = float(beta)  # In auto mode: multiplier k; in fixed mode: absolute beta.
        self.beta_mode = key
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        err_raw = pred - target

        # B: robust automatic scale estimation from current batch residuals.
        if self.beta_mode == "fixed":
            beta = err_raw.new_tensor(self.beta)
        else:
            # Use median(|err|) as a robust scale estimator.
            # For normal residuals: median(|Z|) ~= 0.6745 * sigma  ->  sigma ~= median(|Z|)/0.6745
            abs_err_raw = err_raw.abs()
            med = abs_err_raw.reshape(-1).median()  # robust central scale
            sigma = med / 0.6745
            beta = err_raw.new_tensor(self.beta) * sigma
            beta = beta.clamp(self.beta_min, self.beta_max)
            beta = beta.detach()  # do not backprop through median/scale estimation

        err = err_raw / beta
        abs_err = err.abs()
        # log(cosh(err)) up to a constant: |err| + log(1+exp(-2|err|))
        per_elem = abs_err + torch.log1p(torch.exp(-2.0 * abs_err))
        return torch.mean((beta**2) * per_elem)


def build_training_loss(name: str, beta: float = 1.0, beta_mode: str = "auto_mad") -> nn.Module:
    key = str(name).lower()
    if key == "mse":
        return nn.MSELoss()
    if key in ("logcosh", "log_cosh"):
        return LogCoshLoss()
    if key in ("logcosh_scaled", "scaled_logcosh", "log_cosh_scaled"):
        return ScaledLogCoshLoss(beta=beta, beta_mode=beta_mode)
    raise ValueError(
        f"Unsupported training loss: {name!r}. Choose from ['mse', 'logcosh', 'logcosh_scaled']."
    )
