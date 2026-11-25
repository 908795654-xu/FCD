"""Verify that optimization updates only a newly initialized head."""

import gc
from pathlib import Path

import torch

from adapters import TTMAdapter
from models import HeadForecaster, LinearForecastHead
from sacq_core import SACQCrossAttentionHead


MODEL_PATH = Path(
    "/home/bkai/CG2/SACQ_TSFM/models_cache/ttm-512-96-ft-r2.1"
)
HOURLY_TOKEN = 7


def run_one(head_name, head):
    adapter = TTMAdapter(MODEL_PATH)
    model = HeadForecaster(adapter, head)
    model.train()

    assert model.head.training is True
    assert model.adapter.model.training is False
    assert all(
        not parameter.requires_grad
        for parameter in model.adapter.model.parameters()
    )

    backbone_parameter = next(model.adapter.model.parameters())
    backbone_before = backbone_parameter.detach().clone()

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    assert trainable_parameters
    head_parameter_ids = {
        id(parameter)
        for parameter in model.head.parameters()
    }
    assert all(
        id(parameter) in head_parameter_ids
        for parameter in trainable_parameters
    )

    head_before = [
        parameter.detach().clone()
        for parameter in trainable_parameters
    ]

    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=1e-3,
    )

    torch.manual_seed(2026)
    values = torch.randn(1, 2, 512)
    target = torch.randn(1, 2, 96)

    output = model(
        values,
        adapter_kwargs={"freq_token": HOURLY_TOKEN},
    )

    assert output.prediction.shape == target.shape
    assert output.normalized_prediction.shape == target.shape
    assert output.features.memory.shape == (1, 2, 128, 9)
    assert output.features.memory.requires_grad is False

    loss = torch.nn.functional.mse_loss(
        output.prediction,
        target,
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    assert all(
        parameter.grad is None
        for parameter in model.adapter.model.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in trainable_parameters
    )

    optimizer.step()

    torch.testing.assert_close(
        backbone_parameter,
        backbone_before,
        rtol=0.0,
        atol=0.0,
    )

    head_changed = any(
        not torch.equal(before, after.detach())
        for before, after in zip(
            head_before,
            trainable_parameters,
        )
    )
    assert head_changed

    print(
        f"{head_name}_FROZEN_OK "
        f"loss={loss.item():.6f} "
        f"trainable={sum(p.numel() for p in trainable_parameters)}"
    )

    del model, adapter, optimizer, output
    gc.collect()


def main():
    torch.set_num_threads(2)

    linear = LinearForecastHead(
        hidden_dim=128,
        patch_count=9,
        prediction_length=96,
    )
    run_one("LINEAR", linear)

    torch.manual_seed(2026)
    sacq = SACQCrossAttentionHead(
        n_vars=2,
        nf=128 * 9,
        d_model=128,
        target_window=96,
        patch_len=16,
        cross_attn_layers=2,
        n_heads=8,
        head_dropout=0.0,
        individual=False,
        sacq_alpha_mode="learnable",
        sacq_alpha_value=0.5,
        sacq_autoregressive=False,
        sacq_pe_mode="default",
        hist_patch_num=9,

        dsacq_enable=False,
        dsacq_mem_gate_enable=False,
        dsacq_drift_gate_enable=False,
    )
    run_one("SACQ", sacq)

    print("FROZEN_BACKBONE_TESTS_OK")


if __name__ == "__main__":
    main()
