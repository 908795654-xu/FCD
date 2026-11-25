"""Bounded ETTh1/96 smoke training for one backbone/head pair."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_provider import ETTh1Dataset
from models import build_forecaster


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATHS = {
    "ttm": ROOT / "models_cache/ttm-512-96-ft-r2.1",
    "chronos": ROOT / "models_cache/chronos-bolt-mini",
    "timer": ROOT / "models_cache/timer-base-84m",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone",
        choices=sorted(MODEL_PATHS),
        required=True,
    )
    parser.add_argument(
        "--head",
        choices=("linear", "sacq"),
        required=True,
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=2,
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def validate_smoke_limits(args):
    if not 1 <= args.epochs <= 3:
        raise ValueError("smoke epochs must be in [1,3]")
    if not 1 <= args.max_train_batches <= 50:
        raise ValueError("max_train_batches must be in [1,50]")
    if not 1 <= args.max_val_batches <= 20:
        raise ValueError("max_val_batches must be in [1,20]")
    if not 1 <= args.batch_size <= 32:
        raise ValueError("batch_size must be in [1,32]")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def adapter_kwargs(backbone):
    if backbone == "ttm":
        # ETTh1 is hourly.
        return {"freq_token": 7}
    return {}


def train_epoch(
    model,
    loader,
    optimizer,
    device,
    kwargs,
    max_batches,
):
    model.train()
    loss_sum = 0.0
    element_count = 0

    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break

        values = batch["values"].to(device)
        target = batch["target"].to(device)

        output = model(
            values,
            adapter_kwargs=kwargs,
        )
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

        optimizer.step()

        elements = target.numel()
        loss_sum += loss.item() * elements
        element_count += elements

    return loss_sum / element_count


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    kwargs,
    max_batches,
):
    model.eval()
    squared_error = 0.0
    absolute_error = 0.0
    element_count = 0

    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break

        values = batch["values"].to(device)
        target = batch["target"].to(device)
        prediction = model(
            values,
            adapter_kwargs=kwargs,
        ).prediction

        error = prediction - target
        squared_error += error.square().sum().item()
        absolute_error += error.abs().sum().item()
        element_count += target.numel()

    return {
        "mse": squared_error / element_count,
        "mae": absolute_error / element_count,
    }


def main():
    args = parse_args()
    validate_smoke_limits(args)
    set_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")

    train_dataset = ETTh1Dataset(
        ROOT / "datasets/ETTh1.csv",
        split="train",
        context_length=512,
        prediction_length=96,
    )
    val_dataset = ETTh1Dataset(
        ROOT / "datasets/ETTh1.csv",
        split="val",
        context_length=512,
        prediction_length=96,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model, metadata = build_forecaster(
        backbone_name=args.backbone,
        head_name=args.head,
        model_path=MODEL_PATHS[args.backbone],
        context_length=512,
        prediction_length=96,
        n_vars=7,
        future_patch_length=16,
        cross_attn_layers=2,
        n_heads=16,
        head_dropout=0.0,
    )
    model.to(device)

    assert all(
        not parameter.requires_grad
        for parameter in model.adapter.model.parameters()
    )

    trainable = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    head_parameter_ids = {
        id(parameter)
        for parameter in model.head.parameters()
    }
    assert trainable
    assert all(
        id(parameter) in head_parameter_ids
        for parameter in trainable
    )

    backbone_parameter = next(
        model.adapter.model.parameters()
    )
    backbone_before = (
        backbone_parameter.detach().cpu().clone()
    )

    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start_time = time.perf_counter()
    train_losses = []

    kwargs = adapter_kwargs(args.backbone)

    print("metadata:", metadata)
    print("device:", device)
    print(
        "limits:",
        {
            "epochs": args.epochs,
            "max_train_batches": args.max_train_batches,
            "max_val_batches": args.max_val_batches,
        },
    )

    initial_val = evaluate(
        model,
        val_loader,
        device,
        kwargs,
        args.max_val_batches,
    )
    print("initial_val:", initial_val)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            kwargs,
            args.max_train_batches,
        )
        train_losses.append(train_loss)

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            kwargs,
            args.max_val_batches,
        )
        print(
            f"epoch={epoch} "
            f"train_mse={train_loss:.6f} "
            f"val_mse={val_metrics['mse']:.6f} "
            f"val_mae={val_metrics['mae']:.6f}"
        )

    elapsed = time.perf_counter() - start_time

    torch.testing.assert_close(
        backbone_parameter.detach().cpu(),
        backbone_before,
        rtol=0.0,
        atol=0.0,
    )
    assert all(
        parameter.grad is None
        for parameter in model.adapter.model.parameters()
    )

    peak_memory_mb = None
    if device.type == "cuda":
        peak_memory_mb = (
            torch.cuda.max_memory_allocated(device)
            / 1024 ** 2
        )

    result = {
        **metadata,
        "dataset": "ETTh1",
        "seed": args.seed,
        "device": str(device),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "max_train_batches": args.max_train_batches,
        "max_val_batches": args.max_val_batches,
        "learning_rate": args.learning_rate,
        "loss": "mse",
        "train_losses": train_losses,
        "initial_val": initial_val,
        "final_val": val_metrics,
        "elapsed_seconds": elapsed,
        "peak_memory_mb": peak_memory_mb,
        "backbone_unchanged": True,
    }

    output_dir = ROOT / "results/smoke_etth1_96"
    output_dir.mkdir(parents=True, exist_ok=True)
    device_tag = str(device).replace(":", "")
    run_tag = (
        f"e{args.epochs}_"
        f"tb{args.max_train_batches}_"
        f"vb{args.max_val_batches}"
    )
    output_path = output_dir / (
        f"{args.backbone}_{args.head}_seed{args.seed}_"
        f"{device_tag}_{run_tag}.json"
    )
    output_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("result_path:", output_path)
    print("result:", json.dumps(result, indent=2))
    print("SMOKE_TEST_OK")


if __name__ == "__main__":
    main()
