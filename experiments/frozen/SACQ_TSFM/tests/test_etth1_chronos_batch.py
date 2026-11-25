"""One real ETTh1 batch through frozen Chronos-Bolt and both heads."""

import gc
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from adapters import ChronosBoltAdapter
from data_provider import ETTh1Dataset
from models import LinearForecastHead
from sacq_core import SACQCrossAttentionHead


ROOT = Path("/home/bkai/CG2/SACQ_TSFM")
MODEL_PATH = ROOT / "models_cache/chronos-bolt-mini"
DATA_PATH = ROOT / "datasets/ETTh1.csv"


def optimize_one_step(name, head, adapter, features, target):
    head.train()
    trainable = [
        parameter
        for parameter in head.parameters()
        if parameter.requires_grad
    ]
    before = [
        parameter.detach().clone()
        for parameter in trainable
    ]

    optimizer = torch.optim.Adam(trainable, lr=1e-3)

    head_output = head(features.memory)
    if isinstance(head_output, tuple):
        normalized_prediction, aux = head_output
    else:
        normalized_prediction = head_output
        aux = {}

    prediction = features.denormalize(normalized_prediction)
    assert prediction.shape == target.shape
    assert torch.isfinite(prediction).all()

    loss = torch.nn.functional.mse_loss(prediction, target)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    assert any(
        parameter.grad is not None
        for parameter in trainable
    )
    assert all(
        parameter.grad is None
        for parameter in adapter.model.parameters()
    )

    optimizer.step()

    changed = any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, trainable)
    )
    assert changed

    if name == "SACQ":
        assert "sacq_alpha_seq" in aux
        assert aux["sacq_alpha_seq"].shape == target.shape

    print(
        f"{name}_REAL_BATCH_OK "
        f"loss={loss.item():.6f} "
        f"prediction={tuple(prediction.shape)} "
        f"trainable={sum(p.numel() for p in trainable)}"
    )


torch.set_num_threads(2)
torch.manual_seed(2026)

dataset = ETTh1Dataset(
    DATA_PATH,
    split="train",
    context_length=512,
    prediction_length=96,
)
loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0,
)
batch = next(iter(loader))
values = batch["values"]
target = batch["target"]

adapter = ChronosBoltAdapter(
    MODEL_PATH,
    context_length=512,
)
adapter.eval()
features = adapter(values)

assert features.memory.shape == (2, 7, 384, 32)
assert features.memory.requires_grad is False
assert features.metadata["removed_reg_token"] is True
assert all(
    parameter.grad is None
    for parameter in adapter.model.parameters()
)

print("values:", tuple(values.shape))
print("target:", tuple(target.shape))
print("memory:", tuple(features.memory.shape))

linear = LinearForecastHead(
    hidden_dim=384,
    patch_count=32,
    prediction_length=96,
)
optimize_one_step(
    "LINEAR",
    linear,
    adapter,
    features,
    target,
)
del linear
gc.collect()

torch.manual_seed(2026)
sacq = SACQCrossAttentionHead(
    n_vars=7,
    nf=384 * 32,
    d_model=384,
    target_window=96,
    patch_len=16,
    cross_attn_layers=2,
    n_heads=16,
    head_dropout=0.0,
    individual=False,
    sacq_alpha_mode="learnable",
    sacq_alpha_value=0.5,
    sacq_autoregressive=False,
    sacq_pe_mode="default",
    hist_patch_num=32,

    dsacq_enable=False,
    dsacq_mem_gate_enable=False,
    dsacq_drift_gate_enable=False,
)
optimize_one_step(
    "SACQ",
    sacq,
    adapter,
    features,
    target,
)

assert all(
    parameter.grad is None
    for parameter in adapter.model.parameters()
)

print("ETTH1_CHRONOS_TWO_HEADS_OK")
