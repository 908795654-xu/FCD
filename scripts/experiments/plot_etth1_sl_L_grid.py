#!/usr/bin/env python3
"""Heatmap + best-L curves from etth1_sl_L_grid_pl720/summary.csv."""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_grid(csv_path: str) -> tuple[list[int], list[int], np.ndarray]:
    rows: dict[tuple[int, int], float] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") not in ("ok", "skipped", "reused"):
                continue
            if not row.get("mse"):
                continue
            sl = int(row["seq_len"])
            L = int(row["cross_attn_layers"])
            rows[(sl, L)] = float(row["mse"])
    seq_lens = sorted({k[0] for k in rows})
    l_vals = sorted({k[1] for k in rows})
    mat = np.full((len(l_vals), len(seq_lens)), np.nan)
    for i, L in enumerate(l_vals):
        for j, sl in enumerate(seq_lens):
            mat[i, j] = rows.get((sl, L), np.nan)
    return seq_lens, l_vals, mat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="logs/etth1_sl_L_grid_pl720/summary.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="logs/etth1_sl_L_grid_pl720/figures",
    )
    args = parser.parse_args()

    csv_path = os.path.join(PROJECT_ROOT, args.csv)
    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    seq_lens, l_vals, mat = load_grid(csv_path)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd_r")
    ax.set_xticks(range(len(seq_lens)))
    ax.set_xticklabels(seq_lens)
    ax.set_yticks(range(len(l_vals)))
    ax.set_yticklabels([f"L={L}" for L in l_vals])
    ax.set_xlabel("seq_len (sl)")
    ax.set_ylabel("cross_attn_layers")
    ax.set_title("ETTh1 pl=720 test MSE — sl x L grid (SAR+logcosh_scaled)")
    for i in range(len(l_vals)):
        for j in range(len(seq_lens)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9, color="black")
    fig.colorbar(im, ax=ax, label="MSE")
    fig.tight_layout()
    heat_path = os.path.join(out_dir, "sl_L_mse_heatmap.png")
    fig.savefig(heat_path, dpi=150)
    plt.close(fig)

    best_L = []
    best_mse = []
    for j, sl in enumerate(seq_lens):
        col = mat[:, j]
        idx = int(np.nanargmin(col))
        best_L.append(l_vals[idx])
        best_mse.append(col[idx])

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(seq_lens, best_mse, "o-", lw=2, ms=8, color="#2ca02c", label="best L per sl")
    for sl, L, m in zip(seq_lens, best_L, best_mse):
        ax.annotate(f"L={L}", (sl, m), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xlabel("seq_len")
    ax.set_ylabel("test MSE")
    ax.set_title("ETTh1 pl=720 — oracle best SACQ depth per context length")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    best_path = os.path.join(out_dir, "sl_best_L_mse.png")
    fig.savefig(best_path, dpi=150)
    plt.close(fig)

    print(f"saved {heat_path}")
    print(f"saved {best_path}")


if __name__ == "__main__":
    main()
