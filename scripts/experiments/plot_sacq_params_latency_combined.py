#!/usr/bin/env python3
"""Combine params + latency SACQ vs d_model panels (legend below, no titles)."""
from __future__ import annotations

import argparse
import csv
from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import PatchTST

# shared style with single-panel scripts
BLUE = "#1f77b4"
RED = "#d62728"
LW = 3.0
MS = 7
LABEL_TOTAL = "Total"
LABEL_SACQ = "SACQ-only overhead (excl. coarse flatten)"

SACQ_COMMON = dict(
    seq_len=512,
    pred_len=336,
    patch_len=16,
    stride=8,
    enc_in=7,
    dec_in=7,
    c_out=7,
    e_layers=3,
    cross_attn_layers=2,
    sacq_autoregressive=1,
    head_type="sacq_cross_attention",
    dsacq_enable=0,
    dsacq_mem_gate_enable=0,
    dsacq_drift_gate_enable=0,
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

D_MODELS = [8, 16, 32, 64, 96, 128, 192, 256]
N_HEADS = 4


def count_sacq(d_model: int) -> tuple[list[int], list[float], list[float]]:
    d_ff = 128 if d_model <= 32 else 256
    cfg = Namespace(**SACQ_COMMON, d_model=d_model, n_heads=N_HEADS, d_ff=d_ff)
    m = PatchTST.Model(cfg)
    total = sum(p.numel() for p in m.parameters())
    coarse = sum(p.numel() for p in m.model.head.query_flatten_head.parameters())
    return total, total - coarse


def load_latency_csv(path: Path) -> tuple[list[int], list[float], list[float]]:
    rows = list(csv.DictReader(open(path)))
    xs = [int(r["d_model"]) for r in rows]
    yt = [float(r["total_ms"]) for r in rows]
    yo = [float(r["sacq_only_ms"]) for r in rows]
    return xs, yt, yo


def _plot_panel(ax, xs, yt, yo, ylabel: str) -> None:
    ax.plot(xs, yt, "o-", color=BLUE, label=LABEL_TOTAL, linewidth=LW, markersize=MS)
    ax.plot(xs, yo, "s--", color=RED, label=LABEL_SACQ, linewidth=LW, markersize=MS)
    ax.set_xlabel("d_model")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def plot_combined(out: Path, latency_csv: Path) -> None:
    px, py, po = [], [], []
    for d in D_MODELS:
        if d % N_HEADS != 0:
            continue
        t, s = count_sacq(d)
        px.append(d)
        py.append(t / 1e6)
        po.append(s / 1e6)

    lx, ly, lo = load_latency_csv(latency_csv)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    _plot_panel(axes[0], px, py, po, "Parameters (M)")
    _plot_panel(axes[1], lx, ly, lo, "Latency (ms / forward)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.16)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        fontsize=10,
        frameon=True,
        bbox_to_anchor=(0.5, 0.02),
        borderaxespad=0.0,
    )
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT
        / "logs/sacq_param_sweep/figures/sacq_total_vs_sacqonly_dmodel_combined.png",
    )
    parser.add_argument(
        "--latency_csv",
        type=Path,
        default=PROJECT_ROOT / "logs/sacq_param_sweep/sacq_latency_sweep.csv",
    )
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot_combined(args.out, args.latency_csv)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
