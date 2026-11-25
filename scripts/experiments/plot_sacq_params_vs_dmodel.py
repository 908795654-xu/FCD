#!/usr/bin/env python3
"""Plot SACQ parameter count vs d_model (n_heads does not change param count)."""
from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt

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


def count_sacq(d_model: int, n_heads: int = 4):
    d_ff = 128 if d_model <= 32 else 256
    cfg = Namespace(**SACQ_COMMON, d_model=d_model, n_heads=n_heads, d_ff=d_ff)
    m = PatchTST.Model(cfg)
    total = sum(p.numel() for p in m.parameters())
    coarse = sum(p.numel() for p in m.model.head.query_flatten_head.parameters())
    return total, total - coarse


def plot_total_vs_sacqonly(out: Path, d_models: list[int], n_heads: int):
    xs, yt, yo = [], [], []
    for d in d_models:
        if d % n_heads != 0:
            continue
        total, sacq_only = count_sacq(d, n_heads)
        xs.append(d)
        yt.append(total / 1e6)
        yo.append(sacq_only / 1e6)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(
        xs,
        yt,
        "o-",
        color="#1f77b4",
        label="Total params",
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
    ax.set_ylabel("Parameters (M)")
    ax.set_title(
        "SACQ params vs d_model\n(enc_in=7, pred_len=336, cross_attn_layers=2)"
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
        default=PROJECT_ROOT / "logs/sacq_param_sweep/figures/sacq_total_vs_sacqonly_dmodel.png",
    )
    parser.add_argument("--n_heads", type=int, default=4)
    args = parser.parse_args()
    d_models = [8, 16, 32, 64, 96, 128, 192, 256]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot_total_vs_sacqonly(args.out, d_models, args.n_heads)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
