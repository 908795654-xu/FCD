#!/usr/bin/env python3
"""Benchmark SACQ inference latency vs d_model; plot like sacq_total_vs_sacqonly_dmodel."""
from __future__ import annotations

import argparse
import csv
import time
from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import PatchTST

SACQ_COMMON = dict(
    seq_len=512,
    pred_len=336,
    patch_len=16,
    stride=8,
    enc_in=7,
    dec_in=7,
    c_out=7,
    e_layers=3,
    n_heads=4,
    cross_attn_layers=2,
    sacq_autoregressive=1,
    sacq_alpha_mode="learnable",
    sacq_alpha_value=0.5,
    sacq_drift_alpha_enable=1,
    sacq_alpha_temperature=0.5,
    sacq_alpha_granularity="patch",
    sacq_alpha_softplus=0,
    sacq_overlap_avg=0,
    sacq_future_causal=0,
    sacq_ar_verify=0,
    sacq_coarse_init="flatten",
    sacq_routing_topk=0,
    sacq_routing_window=0,
    sacq_scheme="default",
    sacq_dual_struct_gate=0,
    sacq_dual_res_fuse=0,
    sacq_dual_detach=1,
    sacq_dual_scaffold_ln=0,
    sacq_pe_mode="default",
    sacq_pe_rel_buckets=32,
    dsacq_enable=0,
    dsacq_mem_gate_enable=0,
    dsacq_drift_gate_enable=0,
    dsacq_mem_gate_hidden=32,
    dsacq_mem_gate_mode="stats",
    mem_shift_aug_std=0.15,
    mem_shift_aug_prob=0.5,
    align_ablate_shift=0,
    align_ablate_dual_bank=0,
    align_ablate_token_gate=0,
    align_ablate_shift_drift=0,
    padding_patch="end",
    revin=1,
    affine=0,
    subtract_last=0,
    individual=0,
    head_dropout=0.0,
    dropout=0.2,
    fc_dropout=0.2,
    decomposition=0,
    kernel_size=25,
    d_layers=1,
    label_len=48,
)


def build_model(d_model: int, head_type: str) -> torch.nn.Module:
    d_ff = 128 if d_model <= 32 else 256
    cfg = Namespace(**SACQ_COMMON, d_model=d_model, d_ff=d_ff, head_type=head_type)
    m = PatchTST.Model(cfg).eval().cuda()
    return m


@torch.no_grad()
def bench_forward(model: torch.nn.Module, x: torch.Tensor, warmup: int, repeats: int) -> float:
    """Return mean latency in ms per forward (batch)."""
    for _ in range(warmup):
        model(x)
    if x.is_cuda:
        torch.cuda.synchronize()

    timings = []
    for _ in range(repeats):
        if x.is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x)
        if x.is_cuda:
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - t0) * 1000.0)

    timings.sort()
    # median for stability
    return timings[len(timings) // 2]


@torch.no_grad()
def _encode(bb, x_blc: torch.Tensor) -> torch.Tensor:
    """x: [B, seq_len, n_vars] -> encoder tokens [B, nvars, d_model, patch_num]."""
    x_bcl = x_blc.permute(0, 2, 1)
    z_norm = bb._norm_series(x_bcl)
    return bb._encode(z_norm, already_normed=True)


@torch.no_grad()
def _bench_coarse_head(head, z: torch.Tensor) -> None:
    bs, nvars, d_model, patch_num = z.shape
    kv = z.permute(0, 1, 3, 2).contiguous().reshape(bs * nvars, patch_num, d_model)
    kv = head._apply_shared_kv_pe(kv, patch_num)
    head._compute_coarse_patches(z, kv)


@torch.no_grad()
def bench_sacq_breakdown(
    model: torch.nn.Module, x: torch.Tensor, warmup: int, repeats: int
) -> tuple[float, float]:
    """total E2E ms; sacq_only = (encode+full head) - (encode+coarse) ms."""
    total_ms = bench_forward(model, x, warmup, repeats)

    bb = model.model
    head = bb.head
    for _ in range(warmup):
        z = _encode(bb, x)
        _bench_coarse_head(head, z)
    torch.cuda.synchronize()

    coarse_times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        z = _encode(bb, x)
        _bench_coarse_head(head, z)
        torch.cuda.synchronize()
        coarse_times.append((time.perf_counter() - t0) * 1000.0)

    coarse_times.sort()
    coarse_ms = coarse_times[len(coarse_times) // 2]
    sacq_only_ms = max(total_ms - coarse_ms, 0.0)
    return total_ms, sacq_only_ms


def measure_latencies(
    d_model: int,
    batch_size: int,
    warmup: int,
    repeats: int,
) -> tuple[float, float]:
    x = torch.randn(batch_size, SACQ_COMMON["seq_len"], SACQ_COMMON["enc_in"], device="cuda")
    m_sacq = build_model(d_model, "sacq_cross_attention")
    total_ms, sacq_only_ms = bench_sacq_breakdown(m_sacq, x, warmup, repeats)
    del m_sacq, x
    torch.cuda.empty_cache()
    return total_ms, sacq_only_ms


def plot_latency(out: Path, rows: list[dict]) -> None:
    xs = [r["d_model"] for r in rows]
    yt = [r["total_ms"] for r in rows]
    yo = [r["sacq_only_ms"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(
        xs,
        yt,
        "o-",
        color="#1f77b4",
        label="Total inference",
        linewidth=3.0,
        markersize=7,
    )
    ax.plot(
        xs,
        yo,
        "s--",
        color="#d62728",
        label="SACQ-only overhead (excl. coarse flatten)",
        linewidth=3.0,
        markersize=7,
    )
    ax.set_xlabel("d_model")
    ax.set_ylabel("Latency (ms / forward)")
    ax.set_title(
        "SACQ inference latency vs d_model\n"
        f"(enc_in=7, pred_len=336, batch={rows[0]['batch_size']}, cross_attn_layers=2)"
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT
        / "logs/sacq_param_sweep/figures/sacq_total_vs_sacqonly_dmodel_latency.png",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_ROOT / "logs/sacq_param_sweep/sacq_latency_sweep.csv",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=80)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for latency benchmark")

    d_models = [8, 16, 32, 64, 96, 128, 192, 256]
    n_heads = SACQ_COMMON["n_heads"]
    rows = []
    for d in d_models:
        if d % n_heads != 0:
            continue
        print(f"benchmark d_model={d} ...", flush=True)
        total_ms, sacq_only_ms = measure_latencies(
            d, args.batch_size, args.warmup, args.repeats
        )
        rows.append(
            {
                "d_model": d,
                "n_heads": n_heads,
                "batch_size": args.batch_size,
                "total_ms": round(total_ms, 4),
                "sacq_only_ms": round(sacq_only_ms, 4),
            }
        )
        print(f"  total={total_ms:.2f} ms  sacq_only={sacq_only_ms:.2f} ms", flush=True)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot_latency(args.out, rows)
    print(f"Saved {args.out}")
    print(f"Saved {args.csv}")


if __name__ == "__main__":
    main()
