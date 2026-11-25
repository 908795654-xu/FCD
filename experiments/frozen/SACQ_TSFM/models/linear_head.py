"""Direct linear forecasting head."""

import torch
from torch import Tensor, nn


class LinearForecastHead(nn.Module):
    """Map frozen memory [B,C,D,P] directly to [B,C,H].

    The same linear projection is shared across channels, matching the
    ``individual=False`` coarse head used by SACQ.
    """

    def __init__(
        self,
        hidden_dim: int,
        patch_count: int,
        prediction_length: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.patch_count = int(patch_count)
        self.prediction_length = int(prediction_length)

        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(float(dropout))
        self.projection = nn.Linear(
            self.hidden_dim * self.patch_count,
            self.prediction_length,
        )

    def forward(self, memory: Tensor) -> Tensor:
        if memory.ndim != 4:
            raise ValueError(
                f"memory must have shape [B,C,D,P], got {tuple(memory.shape)}"
            )

        if memory.shape[-2:] != (
            self.hidden_dim,
            self.patch_count,
        ):
            raise ValueError(
                f"expected memory tail shape "
                f"{(self.hidden_dim, self.patch_count)}, "
                f"got {tuple(memory.shape[-2:])}"
            )

        return self.projection(
            self.dropout(self.flatten(memory))
        )
