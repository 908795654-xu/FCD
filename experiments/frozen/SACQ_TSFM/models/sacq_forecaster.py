"""Unified frozen-backbone forecasting wrapper."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from torch import Tensor, nn

from adapters.base import BackboneFeatures


@dataclass
class ForecastOutput:
    """Forecasts and diagnostics returned by HeadForecaster."""

    prediction: Tensor
    normalized_prediction: Tensor
    features: BackboneFeatures
    head_aux: Dict[str, Tensor] = field(default_factory=dict)


class HeadForecaster(nn.Module):
    """Attach a newly initialized Linear or SACQ head to a frozen adapter."""

    def __init__(self, adapter: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.adapter = adapter
        self.head = head

    def train(self, mode: bool = True):
        super().train(mode)
        # Adapter overrides train() and keeps its backbone in eval mode.
        self.adapter.train(False)
        self.head.train(mode)
        return self

    def forward(
        self,
        values: Tensor,
        observed_mask: Optional[Tensor] = None,
        adapter_kwargs: Optional[Dict[str, Any]] = None,
    ) -> ForecastOutput:
        kwargs = dict(adapter_kwargs or {})
        features = self.adapter(
            values,
            observed_mask=observed_mask,
            **kwargs,
        )

        head_output = self.head(features.memory)

        if isinstance(head_output, tuple):
            if len(head_output) != 2:
                raise RuntimeError(
                    "tuple head output must be (prediction, aux)"
                )
            normalized_prediction, head_aux = head_output
        else:
            normalized_prediction = head_output
            head_aux = {}

        prediction = features.denormalize(normalized_prediction)

        return ForecastOutput(
            prediction=prediction,
            normalized_prediction=normalized_prediction,
            features=features,
            head_aux=head_aux,
        )
