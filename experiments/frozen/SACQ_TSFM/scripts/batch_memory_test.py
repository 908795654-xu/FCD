"""One-batch forward/backward memory test."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data_provider import BenchmarkDataset
from models import build_forecaster
from scripts.train import (
    DATASET_CONFIGS,
    MODEL_PATHS,
    get_adapter_kwargs,
    set_seed,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=("weather", "electricity"),
        required=True,
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
        required=True,
    )
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this memory test requires CUDA")

    config = DATASET_CONFIGS[args.dataset]
    dataset = BenchmarkDataset(
        csv_path=ROOT / "datasets" / config["filename"],
        dataset_name=args.dataset,
        split="train",
        context_length=512,
        prediction_length=args.prediction_length,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    batch = next(iter(loader))

    model, metadata = build_forecaster(
        backbone_name=args.backbone,
        head_name=args.head,
        model_path=MODEL_PATHS[
            (args.backbone, args.prediction_length)
        ],
        context_length=512,
        prediction_length=args.prediction_length,
        n_vars=config["n_vars"],
        future_patch_length=16,
        cross_attn_layers=2,
        n_heads=16,
        head_dropout=0.0,
    )
    model.to(device)
    model.train()

    trainable = [
        parameter
        for parameter in model.head.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=1e-4,
        weight_decay=1e-2,
    )

    values = batch["values"].to(
        device,
        non_blocking=True,
    )
    target = batch["target"].to(
        device,
        non_blocking=True,
    )

    adapter_kwargs = get_adapter_kwargs(
        args.backbone,
        args.dataset,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    output = model(
        values,
        adapter_kwargs=adapter_kwargs,
    )
    prediction = output.prediction
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
    torch.cuda.synchronize(device)

    result = {
        "dataset": args.dataset,
        "backbone": args.backbone,
        "head": args.head,
        "prediction_length": args.prediction_length,
        "batch_size": args.batch_size,
        "values_shape": list(values.shape),
        "prediction_shape": list(prediction.shape),
        "loss": float(loss.detach().cpu()),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in trainable
        ),
        "peak_memory_mb": (
            torch.cuda.max_memory_allocated(device)
            / 1024 ** 2
        ),
        "backbone_frozen": True,
        "metadata": metadata,
    }

    print(json.dumps(result, indent=2))
    print("BATCH_MEMORY_TEST_OK")


if __name__ == "__main__":
    main()
