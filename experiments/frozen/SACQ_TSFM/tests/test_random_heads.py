"""Random-tensor tests for Linear and pure SACQ prediction heads."""

import gc

import torch

from models import LinearForecastHead
from sacq_core import SACQCrossAttentionHead


BACKBONE_PATCH_COUNTS = {
    "ttm": 9,
    "chronos_bolt": 32,
    "timer": 6,
}
PREDICTION_LENGTHS = (96, 720)


def assert_has_finite_gradient(module):
    gradients = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients, "head did not receive any gradients"
    assert all(torch.isfinite(grad).all() for grad in gradients)


def run_linear_case(name, patch_count, prediction_length):
    batch_size = 2
    channels = 3
    hidden_dim = 16

    torch.manual_seed(2026)
    memory = torch.randn(
        batch_size,
        channels,
        hidden_dim,
        patch_count,
    )
    target = torch.randn(
        batch_size,
        channels,
        prediction_length,
    )

    head = LinearForecastHead(
        hidden_dim=hidden_dim,
        patch_count=patch_count,
        prediction_length=prediction_length,
    )

    prediction = head(memory)
    assert prediction.shape == target.shape
    assert torch.isfinite(prediction).all()

    loss = torch.nn.functional.mse_loss(prediction, target)
    loss.backward()
    assert_has_finite_gradient(head)

    print(
        f"LINEAR_OK backbone={name} horizon={prediction_length} "
        f"shape={tuple(prediction.shape)} "
        f"params={sum(p.numel() for p in head.parameters())}"
    )


def run_sacq_case(name, patch_count, prediction_length):
    batch_size = 2
    channels = 3
    hidden_dim = 16

    torch.manual_seed(2026)
    memory = torch.randn(
        batch_size,
        channels,
        hidden_dim,
        patch_count,
    )
    target = torch.randn(
        batch_size,
        channels,
        prediction_length,
    )

    head = SACQCrossAttentionHead(
        n_vars=channels,
        nf=hidden_dim * patch_count,
        d_model=hidden_dim,
        target_window=prediction_length,
        patch_len=16,
        cross_attn_layers=2,
        n_heads=4,
        head_dropout=0.0,
        individual=False,
        sacq_alpha_mode="learnable",
        sacq_alpha_value=0.5,
        sacq_autoregressive=False,
        sacq_pe_mode="default",

        # Pure SACQ must always be explicit.
        dsacq_enable=False,
        dsacq_mem_gate_enable=False,
        dsacq_drift_gate_enable=False,
    )

    assert head.dsacq_enable is False
    assert head.dsacq_mem_gate is False
    assert head.dsacq_drift_gate is False
    assert head.cross_attn_layers == 2
    assert head.sacq_alpha_mode == "learnable"

    prediction, aux = head(memory)

    assert prediction.shape == target.shape
    assert torch.isfinite(prediction).all()
    assert "sacq_alpha_seq" in aux
    assert aux["sacq_alpha_seq"].shape == target.shape
    assert torch.isfinite(aux["sacq_alpha_seq"]).all()

    loss = torch.nn.functional.mse_loss(prediction, target)
    loss.backward()
    assert_has_finite_gradient(head)

    print(
        f"SACQ_OK backbone={name} horizon={prediction_length} "
        f"shape={tuple(prediction.shape)} "
        f"params={sum(p.numel() for p in head.parameters())}"
    )


def main():
    torch.set_num_threads(2)

    for name, patch_count in BACKBONE_PATCH_COUNTS.items():
        for prediction_length in PREDICTION_LENGTHS:
            run_linear_case(name, patch_count, prediction_length)
            gc.collect()

            run_sacq_case(name, patch_count, prediction_length)
            gc.collect()

    print("ALL_RANDOM_HEAD_TESTS_OK")


if __name__ == "__main__":
    main()
