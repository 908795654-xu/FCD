#!/usr/bin/env python3
"""Stack ETTh1 + Weather tail-shift (no All) figures vertically; shared bottom legend."""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.experiments.plot_robust_search_composite_2x2 import (
    COLOR_RED,
    COLOR_SACQ,
    COLOR_SAR,
    draw_tail_shift,
    load_tail_rows,
)
from scripts.experiments.robust_dataset_config import pair_spike_ds

LEGEND_LABELS = ("Full SACQ", "w/o SAR", "SAR reduction (%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_path",
        default="logs/fig_tail_shift_noall_etth1_weather_vertical.png",
    )
    args = parser.parse_args()

    etth1_dir = os.path.join(PROJECT_ROOT, "logs/etth1_sar_vs_sacq_search")
    weather_dir = os.path.join(PROJECT_ROOT, "logs/weather_sar_vs_sacq_search")
    pred_len = 720

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.2))
    axes[0, 0].sharex(axes[0, 1])
    axes[1, 0].sharex(axes[1, 1])
    e1_pair = pair_spike_ds("ETTh1", 5, 3.0, pred_len)
    w_pair = pair_spike_ds("weather", 5, 3.0, pred_len)

    e1_rows = load_tail_rows(
        os.path.join(etth1_dir, "spike_train_tail_shift.csv"), "spike_train"
    )
    w_rows = load_tail_rows(
        os.path.join(weather_dir, "spike_train_tail_shift.csv"),
        "spike_train_p5_pl720",
    )

    # Match single-figure protocols (CSV data unchanged; display swaps as in originals).
    draw_tail_shift(
        [axes[0, 0], axes[0, 1]],
        e1_rows,
        e1_pair[0][0],
        e1_pair[1][0],
        pred_len,
        swap_near_far=False,
        swap_far_models=False,
        shift_min=10,
    )
    draw_tail_shift(
        [axes[1, 0], axes[1, 1]],
        w_rows,
        w_pair[0][0],
        w_pair[1][0],
        pred_len,
        swap_near_far=True,
        swap_far_models=True,
        shift_min=0,
    )

    fig.legend(
        [
            Line2D([0], [0], color=COLOR_SAR, lw=2, marker="o"),
            Line2D([0], [0], color=COLOR_SACQ, lw=2, marker="o"),
            Line2D([0], [0], color=COLOR_RED, lw=1.8, ls="--", marker="s"),
        ],
        list(LEGEND_LABELS),
        loc="lower center",
        ncol=3,
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.subplots_adjust(hspace=0.32, wspace=0.28, bottom=0.14, top=0.98, left=0.09, right=0.96)

    out_path = os.path.join(PROJECT_ROOT, args.out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
