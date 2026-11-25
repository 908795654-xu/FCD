"""Lightweight fallback implementations for missing channel group mix module.

These classes preserve interface compatibility for configurations that do not
actively enable group/channel mix features.
"""

import torch
from torch import nn


class ChannelDenseMix(nn.Module):
    def __init__(self, c_in: int, identity_init: bool = True):
        super().__init__()
        self.c_in = c_in
        self.identity_init = identity_init

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z


class ChannelGroupLinearMix(nn.Module):
    def __init__(
        self,
        c_in: int,
        groups=None,
        trend_only: bool = False,
        ma_kernel: int = 7,
        identity_init: bool = True,
        feature_mode: str = "mean",
        mixer_type: str = "linear",
        mlp_hidden=None,
    ):
        super().__init__()
        self.c_in = c_in

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z


class ChannelGroupLinearMixDModel(nn.Module):
    def __init__(
        self,
        c_in: int,
        d_model: int,
        groups=None,
        identity_init: bool = True,
        feature_mode: str = "mean",
    ):
        super().__init__()
        self.c_in = c_in
        self.d_model = d_model

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z
