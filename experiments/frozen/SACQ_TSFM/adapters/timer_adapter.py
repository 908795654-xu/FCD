"""Frozen Timer backbone adapter."""

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModelForCausalLM

from .base import BackboneFeatures, FrozenBackboneAdapter


class TimerAdapter(FrozenBackboneAdapter):
    """Expose Timer decoder tokens as [B,C,D,P].

    Timer uses non-overlapping 96-point tokens. For protocol context 512,
    normalized zeros are padded on the left to length 576. This preserves
    every observed context point and avoids silently dropping the newest
    32 points.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        context_length: int = 512,
        scale_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        self.model_path = str(Path(model_path).resolve())
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.freeze_backbone(self.model)

        self.context_length = int(context_length)
        self.token_length = int(self.model.config.input_token_len)
        self.hidden_dim = int(self.model.config.hidden_size)
        self.scale_epsilon = float(scale_epsilon)

        if self.context_length < 2:
            raise ValueError("context_length must be at least 2")
        if self.token_length < 1:
            raise ValueError("Timer input_token_len must be positive")

        remainder = self.context_length % self.token_length
        self.left_padding = (
            0 if remainder == 0 else self.token_length - remainder
        )
        self.padded_length = self.context_length + self.left_padding
        self.patch_count = self.padded_length // self.token_length

    def train(self, mode: bool = True):
        """The wrapper may train, while Timer always stays in eval mode."""
        super().train(mode)
        self.model.eval()
        return self

    def _masked_revin(
        self,
        values: Tensor,
        observed_mask: Tensor,
    ):
        """Normalize [N,T] using only observed values.

        The variance denominator is count-1, matching torch.std's default
        behavior used by Timer's official ``revin=True`` path.
        """
        mask = observed_mask.to(dtype=values.dtype)
        count = mask.sum(dim=-1, keepdim=True)

        if (count < 2).any():
            raise ValueError(
                "Timer normalization needs at least two observed values "
                "per series"
            )

        clean = torch.where(observed_mask, values, torch.zeros_like(values))
        loc = clean.sum(dim=-1, keepdim=True) / count
        centered = torch.where(
            observed_mask,
            values - loc,
            torch.zeros_like(values),
        )
        variance = centered.square().sum(dim=-1, keepdim=True) / (count - 1)
        scale = variance.clamp_min(self.scale_epsilon ** 2).sqrt()
        normalized = torch.where(
            observed_mask,
            centered / scale,
            torch.zeros_like(centered),
        )
        return normalized, loc, scale

    def forward(
        self,
        values: Tensor,
        observed_mask: Optional[Tensor] = None,
    ) -> BackboneFeatures:
        self.validate_bct(values)

        batch_size, channels, length = values.shape
        if length != self.context_length:
            raise ValueError(
                f"Timer adapter expects exactly {self.context_length} "
                f"context points, got {length}"
            )

        if observed_mask is None:
            observed_mask = torch.isfinite(values)
        elif observed_mask.shape != values.shape:
            raise ValueError(
                "observed_mask must have the same [B,C,T] shape as values"
            )

        flat_values = values.reshape(batch_size * channels, length)
        flat_mask = observed_mask.reshape(batch_size * channels, length)
        normalized, loc, scale = self._masked_revin(
            flat_values,
            flat_mask,
        )

        if self.left_padding:
            normalized = F.pad(
                normalized,
                (self.left_padding, 0),
                mode="constant",
                value=0.0,
            )

        if normalized.shape[-1] != self.padded_length:
            raise RuntimeError("Timer padded input length is inconsistent")

        self.model.eval()
        with torch.no_grad():
            output = self.model.model(
                input_ids=normalized,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )

        hidden = output.last_hidden_state
        if hidden.shape != (
            batch_size * channels,
            self.patch_count,
            self.hidden_dim,
        ):
            raise RuntimeError(
                f"unexpected Timer hidden shape {tuple(hidden.shape)}; "
                f"expected {(batch_size * channels, self.patch_count, self.hidden_dim)}"
            )

        memory = (
            hidden.reshape(
                batch_size,
                channels,
                self.patch_count,
                self.hidden_dim,
            )
            .permute(0, 1, 3, 2)
            .contiguous()
        )
        loc = loc.reshape(batch_size, channels, 1)
        scale = scale.reshape(batch_size, channels, 1)

        return BackboneFeatures(
            memory=memory,
            loc=loc,
            scale=scale,
            metadata={
                "backbone": "timer",
                "context_length": self.context_length,
                "token_length": self.token_length,
                "left_padding": self.left_padding,
                "padded_length": self.padded_length,
                "patch_count": self.patch_count,
                "hidden_dim": self.hidden_dim,
            },
        )
