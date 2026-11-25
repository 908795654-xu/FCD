"""Factory for frozen backbones and newly initialized prediction heads."""

from pathlib import Path
from typing import Tuple, Union

import torch
from torch import nn
from sacq_core.pe_initializer import positional_encoding

from adapters import (
    ChronosBoltAdapter,
    TTMAdapter,
    TimerAdapter,
)
from sacq_core import SACQCrossAttentionHead

from .linear_head import LinearForecastHead
from .sacq_forecaster import HeadForecaster


def build_adapter(
    backbone_name: str,
    model_path: Union[str, Path],
    context_length: int,
):
    name = backbone_name.lower()

    if name == "ttm":
        return TTMAdapter(model_path)

    if name == "chronos":
        adapter = ChronosBoltAdapter(
            model_path,
            context_length=context_length,
        )
        # Chronos patching pads on the left when necessary.
        patch_size = int(
            adapter.model.chronos_config.input_patch_size
        )
        adapter.patch_count = (
            context_length + patch_size - 1
        ) // patch_size
        return adapter

    if name == "timer":
        return TimerAdapter(
            model_path,
            context_length=context_length,
        )

    raise ValueError(
        f"unsupported backbone {backbone_name!r}; "
        "expected ttm, chronos, or timer"
    )


def build_head(
    head_name: str,
    n_vars: int,
    hidden_dim: int,
    patch_count: int,
    prediction_length: int,
    future_patch_length: int = 16,
    cross_attn_layers: int = 2,
    n_heads: int = 16,
    head_dropout: float = 0.0,
    future_query_pe: bool = True,
) -> nn.Module:
    name = head_name.lower()

    if name == "linear":
        return LinearForecastHead(
            hidden_dim=hidden_dim,
            patch_count=patch_count,
            prediction_length=prediction_length,
            dropout=head_dropout,
        )

    if name == "sacq":
        head = SACQCrossAttentionHead(
            n_vars=n_vars,
            nf=hidden_dim * patch_count,
            d_model=hidden_dim,
            target_window=prediction_length,
            patch_len=future_patch_length,
            cross_attn_layers=cross_attn_layers,
            n_heads=n_heads,
            head_dropout=head_dropout,
            individual=False,
            sacq_alpha_mode="learnable",
            sacq_alpha_value=0.5,
            sacq_autoregressive=False,
            sacq_pe_mode="default",
            hist_patch_num=patch_count,

            # Pure SACQ baseline: always explicit.
            dsacq_enable=False,
            dsacq_mem_gate_enable=False,
            dsacq_drift_gate_enable=False,
        )
        # Same initializer, learnability and table length as the original wrapper.
        # Create AFTER shared head weights, isolating RNG to allow controlled PE-on/off checks.
        if future_query_pe:
            with torch.random.fork_rng(devices=[]):
                head.query_pos_emb = positional_encoding(
                    "zeros", True, patch_count + prediction_length, hidden_dim
                )
        return head

    raise ValueError(
        f"unsupported head {head_name!r}; expected linear or sacq"
    )


def build_forecaster(
    backbone_name: str,
    head_name: str,
    model_path: Union[str, Path],
    context_length: int,
    prediction_length: int,
    n_vars: int,
    future_patch_length: int = 16,
    cross_attn_layers: int = 2,
    n_heads: int = 16,
    head_dropout: float = 0.0,
    future_query_pe: bool = True,
) -> Tuple[HeadForecaster, dict]:
    adapter = build_adapter(
        backbone_name=backbone_name,
        model_path=model_path,
        context_length=context_length,
    )

    hidden_dim = int(adapter.hidden_dim)
    patch_count = int(adapter.patch_count)

    if hidden_dim % n_heads != 0:
        raise ValueError(
            f"hidden_dim={hidden_dim} must be divisible by "
            f"n_heads={n_heads}"
        )

    head = build_head(
        head_name=head_name,
        n_vars=n_vars,
        hidden_dim=hidden_dim,
        patch_count=patch_count,
        prediction_length=prediction_length,
        future_patch_length=future_patch_length,
        cross_attn_layers=cross_attn_layers,
        n_heads=n_heads,
        head_dropout=head_dropout,
        future_query_pe=future_query_pe,
    )

    forecaster = HeadForecaster(
        adapter=adapter,
        head=head,
    )

    metadata = {
        "implementation_profile": "readme_sacq_default_pe_v2",
        "future_query_pe": bool(future_query_pe and head_name.lower() == "sacq"),
        "pe_initializer": "zeros_uniform_minus_0.02_to_0.02",
        "pe_learnable": bool(future_query_pe and head_name.lower() == "sacq"),
        "backbone": backbone_name.lower(),
        "head": head_name.lower(),
        "context_length": context_length,
        "prediction_length": prediction_length,
        "n_vars": n_vars,
        "hidden_dim": hidden_dim,
        "patch_count": patch_count,
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in forecaster.parameters()
            if parameter.requires_grad
        ),
        "frozen_parameters": sum(
            parameter.numel()
            for parameter in forecaster.parameters()
            if not parameter.requires_grad
        ),
    }

    return forecaster, metadata
