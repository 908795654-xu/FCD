"""Frozen Chronos-Bolt encoder adapter."""

from pathlib import Path
from typing import Optional, Union

import torch
from torch import Tensor

from chronos import ChronosBoltPipeline

from .base import BackboneFeatures, FrozenBackboneAdapter


class ChronosBoltAdapter(FrozenBackboneAdapter):
    """Expose Chronos-Bolt encoder patch memory as [B,C,D,P].

    Multivariate inputs are handled channel-independently:
    [B,C,T] -> [B*C,T] -> encoder -> [B,C,D,P].
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        context_length: int = 512,
    ) -> None:
        super().__init__()
        self.model_path = str(Path(model_path).resolve())

        pipeline = ChronosBoltPipeline.from_pretrained(
            self.model_path,
            device_map="cpu",
        )
        self.model = pipeline.model
        self.freeze_backbone(self.model)

        self.max_context_length = int(
            self.model.chronos_config.context_length
        )
        self.context_length = int(context_length)
        if not 1 <= self.context_length <= self.max_context_length:
            raise ValueError(
                f"context_length must be in [1,{self.max_context_length}], "
                f"got {self.context_length}"
            )
        self.hidden_dim = int(self.model.config.d_model)
        self.uses_reg_token = bool(
            self.model.chronos_config.use_reg_token
        )

    def train(self, mode: bool = True):
        """The wrapper may train, while Chronos always stays in eval mode."""
        super().train(mode)
        self.model.eval()
        return self

    def forward(
        self,
        values: Tensor,
        observed_mask: Optional[Tensor] = None,
    ) -> BackboneFeatures:
        self.validate_bct(values)

        batch_size, channels, length = values.shape
        if length != self.context_length:
            raise ValueError(
                f"Chronos-Bolt expects exactly {self.context_length} "
                f"context points, got {length}"
            )

        if observed_mask is None:
            observed_mask = torch.isfinite(values)
        elif observed_mask.shape != values.shape:
            raise ValueError(
                "observed_mask must have the same [B,C,T] shape as values"
            )

        flat_values = values.reshape(batch_size * channels, length)
        flat_mask = observed_mask.reshape(
            batch_size * channels,
            length,
        )

        self.model.eval()
        with torch.no_grad():
            hidden, loc_scale, _, attention_mask = self.model.encode(
                context=flat_values,
                mask=flat_mask,
            )

        encoded_token_count = hidden.shape[1]

        # Chronos appends [REG] after all historical patch tokens.
        if self.uses_reg_token:
            if encoded_token_count < 2:
                raise RuntimeError(
                    "Chronos encoder output is too short to contain [REG]"
                )
            hidden = hidden[:, :-1, :]
            attention_mask = attention_mask[:, :-1]

        patch_count = hidden.shape[1]
        memory = (
            hidden.reshape(
                batch_size,
                channels,
                patch_count,
                self.hidden_dim,
            )
            .permute(0, 1, 3, 2)
            .contiguous()
        )

        loc, scale = loc_scale
        loc = loc.reshape(batch_size, channels, 1)
        scale = scale.reshape(batch_size, channels, 1)

        expected = (
            batch_size,
            channels,
            self.hidden_dim,
            patch_count,
        )
        if memory.shape != expected:
            raise RuntimeError(
                f"unexpected Chronos memory shape {tuple(memory.shape)}; "
                f"expected {expected}"
            )

        return BackboneFeatures(
            memory=memory,
            loc=loc,
            scale=scale,
            metadata={
                "backbone": "chronos_bolt",
                "context_length": self.context_length,
                "max_context_length": self.max_context_length,
                "hidden_dim": self.hidden_dim,
                "patch_count": patch_count,
                "encoded_token_count": encoded_token_count,
                "removed_reg_token": self.uses_reg_token,
                "valid_patch_tokens": int(attention_mask.sum().item()),
            },
        )
