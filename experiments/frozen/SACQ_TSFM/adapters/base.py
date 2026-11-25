"""Common interface for frozen time-series backbone adapters."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
from torch import Tensor, nn


@dataclass
class BackboneFeatures:
    """Normalized backbone memory and the statistics needed to restore scale.

    Attributes
    ----------
    memory:
        Historical token memory in the unified shape [B, C, D, P].
    loc:
        Per-series location in shape [B, C, 1], or None.
    scale:
        Per-series scale in shape [B, C, 1], or None.
    metadata:
        Model-specific diagnostic values.
    """

    memory: Tensor
    loc: Optional[Tensor] = None
    scale: Optional[Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def denormalize(self, prediction: Tensor) -> Tensor:
        """Convert a normalized [B,C,H] prediction to data scale."""
        if prediction.ndim != 3:
            raise ValueError(
                f"prediction must have shape [B,C,H], got {tuple(prediction.shape)}"
            )

        if self.loc is None and self.scale is None:
            return prediction

        if self.loc is None or self.scale is None:
            raise RuntimeError("loc and scale must either both be set or both be None")

        return prediction * self.scale + self.loc


class FrozenBackboneAdapter(nn.Module):
    """Base class for adapters whose pretrained parameters remain frozen."""

    def freeze_backbone(self, backbone: nn.Module) -> None:
        backbone.requires_grad_(False)
        backbone.eval()

    @staticmethod
    def validate_bct(x: Tensor) -> None:
        if x.ndim != 3:
            raise ValueError(
                f"input must have shape [B,C,T], got {tuple(x.shape)}"
            )
        if not torch.is_floating_point(x):
            raise TypeError("input time series must be floating point")
