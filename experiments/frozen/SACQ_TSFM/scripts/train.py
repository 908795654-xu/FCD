"""Formal frozen-backbone benchmark training."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_provider import BenchmarkDataset
from models import build_forecaster


ROOT = Path(__file__).resolve().parents[1]

DATASET_CONFIGS = {
    "etth1": {
        "filename": "ETTh1.csv",
        "n_vars": 7,
        "freq_token": 7,
    },
    "weather": {
        "filename": "weather.csv",
        "n_vars": 21,
        "freq_token": 4,
    },
    "electricity": {
        "filename": "electricity.csv",
        "n_vars": 321,
        "freq_token": 7,
    },
}

MODEL_PATHS = {
    ("ttm", 96): ROOT / "models_cache/ttm-512-96-ft-r2.1",
    ("ttm", 720): ROOT / "models_cache/ttm-512-720-r2",
    ("chronos", 96): ROOT / "models_cache/chronos-bolt-mini",
    ("chronos", 720): ROOT / "models_cache/chronos-bolt-mini",
    ("timer", 96): ROOT / "models_cache/timer-base-84m",
    ("timer", 720): ROOT / "models_cache/timer-base-84m",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_CONFIGS),
        default="etth1",
    )
    parser.add_argument(
        "--backbone",
        choices=("ttm", "chronos", "timer"),
        required=True,
    )
    parser.add_argument(
        "--head",
        choices=("linear", "sacq"),
        required=True,
    )
    parser.add_argument(
        "--prediction-length",
        type=int,
        choices=(96, 720),
        default=96,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--training-protocol",
        choices=("frozen_v1", "paper"),
        default="frozen_v1",
    )
    return parser.parse_args()


def validate_args(args):
    if not 1 <= args.epochs <= 100:
        raise ValueError("epochs must be in [1,100]")
    if not 1 <= args.patience <= args.epochs:
        raise ValueError("patience must be in [1,epochs]")
    if not 1 <= args.batch_size <= 128:
        raise ValueError("batch_size must be in [1,128]")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_adapter_kwargs(backbone, dataset_name):
    if backbone == "ttm":
        return {
            "freq_token": DATASET_CONFIGS[
                dataset_name
            ]["freq_token"]
        }
    return {}


def make_loaders(args, device):
    config = DATASET_CONFIGS[args.dataset]
    datasets = {
        split: BenchmarkDataset(
            csv_path=ROOT / "datasets" / config["filename"],
            dataset_name=args.dataset,
            split=split,
            context_length=512,
            prediction_length=args.prediction_length,
        )
        for split in ("train", "val", "test")
    }

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }

    loaders = {
        "train": DataLoader(
            datasets["train"],
            shuffle=True,
            generator=generator,
            **common,
        ),
        "val": DataLoader(
            datasets["val"],
            shuffle=False,
            **common,
        ),
        "test": DataLoader(
            datasets["test"],
            shuffle=False,
            **common,
        ),
    }
    return datasets, loaders


def train_epoch(model, loader, optimizer, device, adapter_kwargs):
    model.train()
    squared_error = 0.0
    element_count = 0

    for batch in loader:
        values = batch["values"].to(
            device,
            non_blocking=True,
        )
        target = batch["target"].to(
            device,
            non_blocking=True,
        )

        prediction = model(
            values,
            adapter_kwargs=adapter_kwargs,
        ).prediction
        loss = torch.nn.functional.mse_loss(
            prediction,
            target,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if any(
            parameter.grad is not None
            for parameter in model.adapter.model.parameters()
        ):
            raise RuntimeError("frozen backbone received gradients")

        optimizer.step()

        squared_error += loss.item() * target.numel()
        element_count += target.numel()

    return squared_error / element_count


@torch.no_grad()
def evaluate(model, loader, device, adapter_kwargs):
    model.eval()
    squared_error = 0.0
    absolute_error = 0.0
    element_count = 0

    for batch in loader:
        values = batch["values"].to(
            device,
            non_blocking=True,
        )
        target = batch["target"].to(
            device,
            non_blocking=True,
        )
        prediction = model(
            values,
            adapter_kwargs=adapter_kwargs,
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
    validate_args(args)
    set_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    datasets, loaders = make_loaders(args, device)

    model, metadata = build_forecaster(
        backbone_name=args.backbone,
        head_name=args.head,
        model_path=MODEL_PATHS[
            (args.backbone, args.prediction_length)
        ],
        context_length=512,
        prediction_length=args.prediction_length,
        n_vars=DATASET_CONFIGS[args.dataset]["n_vars"],
        future_patch_length=16,
        cross_attn_layers=2,
        n_heads=16,
        head_dropout=0.0,
    )
    model.to(device)

    if not all(
        not parameter.requires_grad
        for parameter in model.adapter.model.parameters()
    ):
        raise RuntimeError("backbone is not fully frozen")

    trainable = [
        parameter
        for parameter in model.head.parameters()
        if parameter.requires_grad
    ]
    if args.training_protocol == "paper":
        optimizer = torch.optim.Adam(
            trainable,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        scheduler = None
    else:
        optimizer = torch.optim.AdamW(
            trainable,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=1,
            min_lr=1e-6,
        )

    run_name = (
        f"{args.dataset}_p{args.prediction_length}_"
        f"{args.backbone}_{args.head}_seed{args.seed}_"
        f"{args.training_protocol}"
    )
    checkpoint_dir = ROOT / "checkpoints/formal"
    result_dir = ROOT / "results/formal"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"{run_name}.pt"
    result_path = result_dir / f"{run_name}.json"

    backbone_parameter = next(
        model.adapter.model.parameters()
    )
    backbone_before = (
        backbone_parameter.detach().cpu().clone()
    )

    adapter_kwargs = get_adapter_kwargs(
        args.backbone,
        args.dataset,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    print("run_name:", run_name)
    print("metadata:", metadata)
    print(
        "dataset_lengths:",
        {key: len(value) for key, value in datasets.items()},
    )
    print(
        "training:",
        {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "training_protocol": args.training_protocol,
            "optimizer": type(optimizer).__name__,
        },
    )

    start_time = time.perf_counter()
    history = []
    epochs_without_improvement = 0

    initial_val = evaluate(
        model,
        loaders["val"],
        device,
        adapter_kwargs,
    )
    best_val_mse = initial_val["mse"]
    best_epoch = 0

    history.append(
        {
            "epoch": 0,
            "train_mse": None,
            "val_mse": initial_val["mse"],
            "val_mae": initial_val["mae"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": 0.0,
        }
    )

    torch.save(
        {
            "head_state_dict": model.head.state_dict(),
            "epoch": 0,
            "val_metrics": initial_val,
            "metadata": metadata,
            "args": vars(args),
        },
        checkpoint_path,
    )

    print(
        f"epoch=00 train_mse=NA "
        f"val_mse={initial_val['mse']:.6f} "
        f"val_mae={initial_val['mae']:.6f} "
        f"lr={optimizer.param_groups[0]['lr']:.2e}"
    )

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()

        train_mse = train_epoch(
            model,
            loaders["train"],
            optimizer,
            device,
            adapter_kwargs,
        )
        val_metrics = evaluate(
            model,
            loaders["val"],
            device,
            adapter_kwargs,
        )

        if args.training_protocol == "paper":
            # Paper schedule: epochs 1-3 use the initial LR.
            # Starting with epoch 4, multiply LR by 0.9 each epoch.
            if epoch >= 3:
                for group in optimizer.param_groups:
                    group["lr"] *= 0.9
        else:
            scheduler.step(val_metrics["mse"])

        learning_rate = optimizer.param_groups[0]["lr"]
        epoch_seconds = time.perf_counter() - epoch_start

        row = {
            "epoch": epoch,
            "train_mse": train_mse,
            "val_mse": val_metrics["mse"],
            "val_mae": val_metrics["mae"],
            "learning_rate": learning_rate,
            "epoch_seconds": epoch_seconds,
        }
        history.append(row)

        print(
            f"epoch={epoch:02d} "
            f"train_mse={train_mse:.6f} "
            f"val_mse={val_metrics['mse']:.6f} "
            f"val_mae={val_metrics['mae']:.6f} "
            f"lr={learning_rate:.2e} "
            f"seconds={epoch_seconds:.2f}"
        )

        if val_metrics["mse"] < best_val_mse:
            best_val_mse = val_metrics["mse"]
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "head_state_dict": model.head.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "metadata": metadata,
                    "args": vars(args),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(
                f"early_stop epoch={epoch} "
                f"best_epoch={best_epoch}"
            )
            break

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.head.load_state_dict(
        checkpoint["head_state_dict"],
        strict=True,
    )

    best_val = evaluate(
        model,
        loaders["val"],
        device,
        adapter_kwargs,
    )
    test_metrics = evaluate(
        model,
        loaders["test"],
        device,
        adapter_kwargs,
    )

    elapsed_seconds = time.perf_counter() - start_time

    torch.testing.assert_close(
        backbone_parameter.detach().cpu(),
        backbone_before,
        rtol=0.0,
        atol=0.0,
    )

    peak_memory_mb = None
    if device.type == "cuda":
        peak_memory_mb = (
            torch.cuda.max_memory_allocated(device)
            / 1024 ** 2
        )

    result = {
        **metadata,
        "run_name": run_name,
        "dataset": args.dataset,
        "seed": args.seed,
        "device": str(device),
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "completed_epochs": len(history),
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "loss": "mse",
        "best_epoch": best_epoch,
        "initial_val": initial_val,
        "best_val": best_val,
        "test": test_metrics,
        "history": history,
        "elapsed_seconds": elapsed_seconds,
        "peak_memory_mb": peak_memory_mb,
        "backbone_unchanged": True,
        "checkpoint_path": str(checkpoint_path),
    }
    result_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("best_epoch:", best_epoch)
    print("best_val:", best_val)
    print("test:", test_metrics)
    print("checkpoint_path:", checkpoint_path)
    print("result_path:", result_path)
    print("FORMAL_TRAINING_OK")


if __name__ == "__main__":
    main()
