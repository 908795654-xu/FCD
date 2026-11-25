"""Frozen TinyTimeMixer adapter."""

from pathlib import Path
from typing import Optional, Union

import torch
from torch import Tensor

from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

from .base import BackboneFeatures, FrozenBackboneAdapter


class TTMAdapter(FrozenBackboneAdapter):
    """Expose TTM decoder tokens as [B,C,D,P] memory.

    Input
    -----
    values:
        Time series in shape [B,C,T].

    Output
    ------
    BackboneFeatures.memory:
        Decoder representation in shape [B,C,decoder_d_model,P].
    """

    def __init__(self, model_path: Union[str, Path]) -> None:
        super().__init__()
        self.model_path = str(Path(model_path).resolve())
        self.model = TinyTimeMixerForPrediction.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        self.freeze_backbone(self.model)

        self.context_length = int(self.model.config.context_length)
        self.hidden_dim = int(self.model.config.decoder_d_model)
        self.patch_count = int(self.model.config.num_patches)
        self.requires_frequency_token = bool(
            getattr(self.model.config, "resolution_prefix_tuning", False)
        )

    def train(self, mode: bool = True):
        """The adapter may be put in train mode, but TTM stays in eval mode."""
        super().train(mode)
        self.model.eval()
        return self

    def _prepare_frequency_token(
        self,
        freq_token: Optional[Union[int, Tensor]],
        batch_size: int,
        device: torch.device,
    ) -> Optional[Tensor]:
        if not self.requires_frequency_token:
            return None

        if freq_token is None:
            raise ValueError(
                "This TTM checkpoint requires freq_token. "
                "Pass the official frequency-token integer for the dataset."
            )

        token = torch.as_tensor(freq_token, dtype=torch.long, device=device)

        if token.ndim == 0:
            token = token.expand(batch_size)
        elif token.ndim == 1 and token.numel() == 1:
            token = token.expand(batch_size)
        elif token.shape != (batch_size,):
            raise ValueError(
                f"freq_token must be scalar or shape [{batch_size}], "
                f"got {tuple(token.shape)}"
            )

        return token

    def forward(
        self,
        values: Tensor,
        observed_mask: Optional[Tensor] = None,
        freq_token: Optional[Union[int, Tensor]] = None,
    ) -> BackboneFeatures:
        self.validate_bct(values)

        batch_size, channels, length = values.shape
        if length != self.context_length:
            raise ValueError(
                f"TTM expects exactly {self.context_length} context points, "
                f"got {length}"
            )

        if observed_mask is None:
            observed_mask = torch.isfinite(values)
        elif observed_mask.shape != values.shape:
            raise ValueError(
                "observed_mask must have the same [B,C,T] shape as values"
            )

        clean_values = torch.nan_to_num(values)
        past_values = clean_values.transpose(1, 2).contiguous()
        past_mask = observed_mask.transpose(1, 2).contiguous()
        token = self._prepare_frequency_token(
            freq_token,
            batch_size,
            values.device,
        )

        self.model.eval()
        with torch.no_grad():
            output = self.model(
                past_values=past_values,
                past_observed_mask=past_mask,
                freq_token=token,
                output_hidden_states=False,
                return_loss=False,
                return_dict=True,
            )

        # Official decoder hidden: [B,C,P,D].
        memory = output.decoder_hidden_state.permute(0, 1, 3, 2).contiguous()
        loc = output.loc.transpose(1, 2).contiguous()
        scale = output.scale.transpose(1, 2).contiguous()

        expected = (
            batch_size,
            channels,
            self.hidden_dim,
            self.patch_count,
        )
        if memory.shape != expected:
            raise RuntimeError(
                f"unexpected TTM memory shape {tuple(memory.shape)}; "
                f"expected {expected}"
            )

        return BackboneFeatures(
            memory=memory,
            loc=loc,
            scale=scale,
            metadata={
                "backbone": "ttm",
                "context_length": self.context_length,
                "hidden_dim": self.hidden_dim,
                "patch_count": self.patch_count,
            },
        )
