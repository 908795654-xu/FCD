"""Plot test MSE vs seq_len — 2-way or 4-way (style aligned with corruption mask MSE plots)."""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Match eval_etth1_train_corruption_sl512.MODEL_COLORS
VARIANT_2WAY = ("flatten_mse", "sar_logcosh_scaled")

VARIANT_STYLES = {
    "flatten_mse": {
        "label": "Flatten+MSE",
        "color": "#ff7f0e",
        "marker": "o",
        "ls": "-",
    },
    "flatten_logcosh_scaled": {
        "label": "Flatten+logcosh",
        "color": "#d62728",
        "marker": "o",
        "ls": "-",
    },
    "sar_logcosh_scaled": {
        "label": "SAR+logcosh",
        "color": "#2ca02c",
        "marker": "o",
        "ls": "-",
    },
    "sar_mse": {
        "label": "SAR+MSE",
        "color": "#1f77b4",
        "marker": "o",
        "ls": "-",
    },
}

FIGSIZE = (7.5, 5)  # 3:2, same as test_corruption_mask_mse.png
LINE_KW = dict(lw=2, ms=8)


def load_summary(
    csv_path: str, pred_len: int | None = None
) -> tuple[list[int], dict[str, dict[int, dict[str, float]]]]:
    """variant -> seq_len -> {mse, mae} (last ok row per key)"""
    data: dict[str, dict[int, dict[str, float]]] = {v: {} for v in VARIANT_STYLES}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            if pred_len is not None and int(row.get("pred_len", 0)) != pred_len:
                continue
            variant = row.get("variant", "")
            if variant not in data:
                continue
            if not row.get("mse") or not row.get("mae"):
                continue
            sl = int(row["seq_len"])
            data[variant][sl] = {"mse": float(row["mse"]), "mae": float(row["mae"])}
    seq_lens = sorted({sl for v in data.values() for sl in v})
    return seq_lens, data


def plot_mse_vs_seqlen(
    seq_lens: list[int],
    data: dict,
    out_path: str,
    pred_len: int,
    variants: tuple[str, ...] | None = None,
    split: str = "test",
):
    """Single-panel MSE curve; layout matches plot_mask_mse_only."""
    use = variants or tuple(VARIANT_STYLES.keys())
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for variant in use:
        style = VARIANT_STYLES.get(variant)
        if not style:
            continue
        series = data.get(variant, {})
        x = [sl for sl in seq_lens if sl in series]
        if not x:
            continue
        y = [series[sl]["mse"] for sl in x]
        ax.plot(
            x,
            y,
            f"{style['marker']}{style['ls']}",
            color=style["color"],
            label=style["label"],
            **LINE_KW,
        )

    ax.set_xticks(seq_lens)
    ax.set_xticklabels([str(sl) for sl in seq_lens], fontsize=10)
    ax.set_ylabel(f"{split.capitalize()} MSE")
    ax.set_title(rf"ETTh1 $T_f$={pred_len}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=str,
        default="logs/etth1_pl720_seqlen_sweep/summary.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="logs/etth1_pl720_seqlen_sweep/figures",
    )
    parser.add_argument("--pred_len", type=int, default=720)
    parser.add_argument(
        "--plot_2way",
        action="store_true",
        help="also save Flatten+MSE vs SAR+logcosh 2-way figure",
    )
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    csv_path = os.path.join(PROJECT_ROOT, args.summary)
    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    pl = args.pred_len
    seq_lens, data = load_summary(csv_path, pred_len=pl)
    out_4way = os.path.join(out_dir, f"etth1_pl{pl}_mse_mae_vs_seqlen_4way.png")
    plot_mse_vs_seqlen(seq_lens, data, out_4way, pl, split=args.split)
    print(f"seq_lens={seq_lens}", flush=True)
    for v in VARIANT_STYLES:
        print(f"  {v}: {sorted(data[v].keys())}", flush=True)
    print(f"saved -> {out_4way}", flush=True)

    if args.plot_2way:
        out_2way = os.path.join(out_dir, f"etth1_pl{pl}_mse_mae_vs_seqlen.png")
        plot_mse_vs_seqlen(
            seq_lens,
            data,
            out_2way,
            pl,
            variants=VARIANT_2WAY,
            split=args.split,
        )
        print(f"saved -> {out_2way}", flush=True)


if __name__ == "__main__":
    main()
