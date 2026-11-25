#!/usr/bin/env python3
"""2×2 composite: ETTh1 / Weather × (tail-shift Near|Far, horizon MSE)."""
from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.experiments.eval_sar_vs_sacq_spike_train_shift import sar_reduction_pct
from scripts.experiments.plot_sar_vs_sacq_spike_horizon_mse import (
    apply_horizon_swaps,
    load_horizon_csv,
)
from scripts.experiments.robust_dataset_config import pair_spike_ds

# Green = SAR (AR); purple = SACQ (non-AR)
LABEL_MAP = {
    "SAR (spike-train)": "SACQ (full)",
    "SACQ (spike-train)": "w/o SAR",
}
COLOR_SAR = "#2ca02c"
COLOR_SACQ = "#9467bd"
COLOR_RED = "#d62728"
RED_LABEL = "SAR reduction (%)"


def display_label(model: str) -> str:
    return LABEL_MAP.get(model, model)


def load_tail_rows(csv_path: str, pair_tag: str | None) -> list[dict]:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not pair_tag:
        return rows
    tagged = [r for r in rows if r.get("pair") == pair_tag]
    if tagged:
        return tagged
    for alt in (pair_tag.rsplit("_pl", 1)[0], "spike_train"):
        tagged = [r for r in rows if r.get("pair") == alt]
        if tagged:
            return tagged
    return rows


def draw_tail_shift(
    axes: list,
    rows: list[dict],
    sar_label: str,
    sacq_label: str,
    pred_len: int,
    swap_near_far: bool,
    swap_far_models: bool,
    shift_min: int = 0,
) -> list:
    """Draw near/far tail-shift on two axes; return legend handles."""
    rows = [r for r in rows if int(r["shift_pct"]) >= shift_min]
    seg_key = {"near": "mse_near", "far": "mse_far"}
    if swap_near_far:
        seg_key = {"near": "mse_far", "far": "mse_near"}
    sar_rows = {int(r["shift_pct"]): r for r in rows if r["model"] == sar_label}
    sacq_rows = {int(r["shift_pct"]): r for r in rows if r["model"] == sacq_label}
    xs = sorted(sar_rows.keys())
    handles = []

    for ax, seg in zip(axes, ("near", "far")):
        key = seg_key[seg]
        flip = seg == "far" and swap_far_models
        for model, color, mk in (
            (sar_label, COLOR_SAR, "o-"),
            (sacq_label, COLOR_SACQ, "o-"),
        ):
            ys = []
            for x in xs:
                if flip:
                    src = sacq_rows if model == sar_label else sar_rows
                else:
                    src = sar_rows if model == sar_label else sacq_rows
                ys.append(float(src[x][key]))
            (ln,) = ax.plot(xs, ys, mk, lw=2, ms=4, color=color)
            if seg == "near" and model == sar_label:
                handles.append((ln, display_label(model)))
            elif seg == "near" and model == sacq_label:
                handles.append((ln, display_label(model)))

        ax2 = ax.twinx()
        if flip:
            reduc = [
                sar_reduction_pct(float(sacq_rows[x][key]), float(sar_rows[x][key]))
                for x in xs
            ]
        else:
            reduc = [
                sar_reduction_pct(float(sar_rows[x][key]), float(sacq_rows[x][key]))
                for x in xs
            ]
        (ln_r,) = ax2.plot(
            xs, reduc, "s--", lw=1.8, ms=3, color=COLOR_RED, alpha=0.85
        )
        ax2.axhline(0, color=COLOR_RED, lw=0.8, alpha=0.35)
        ax2.set_ylabel(RED_LABEL, fontsize=8, color=COLOR_RED)
        ax2.tick_params(axis="y", labelcolor=COLOR_RED, labelsize=7)
        ymax = max(max(reduc), 0.5)
        ymin = min(min(reduc), -0.5)
        pad = 0.15 * (ymax - ymin if ymax > ymin else 1.0)
        ax2.set_ylim(ymin - pad, ymax + pad)
        if seg == "near":
            handles.append((ln_r, RED_LABEL))

        ax.set_xlabel("Tail shift (%)", fontsize=9)
        ax.set_ylabel("MSE", fontsize=9)
        ax.grid(True, alpha=0.3)
        if xs:
            ax.set_xlim(min(xs) - 1, max(xs) + 1)

    return handles


def draw_horizon_mse(
    axes: list,
    results: dict[int, dict[str, np.ndarray]],
    shift_pcts: list[int],
    sar_label: str,
    sacq_label: str,
    bin_size: int,
    pred_len: int,
    seg_start: int,
    seg_end: int,
) -> list:
    from scripts.experiments.plot_sar_vs_sacq_spike_horizon_mse import bin_centers

    xs = bin_centers(seg_start, seg_end, bin_size)
    handles = []

    for ax, sp in zip(axes, shift_pcts):
        res = results[sp]
        for model, color in ((sar_label, COLOR_SAR), (sacq_label, COLOR_SACQ)):
            (ln,) = ax.plot(
                xs, res[model], "o-", lw=2, ms=4, color=color
            )
            if sp == shift_pcts[0]:
                handles.append((ln, display_label(model)))

        ax2 = ax.twinx()
        reduc = [
            sar_reduction_pct(float(res[sar_label][i]), float(res[sacq_label][i]))
            for i in range(len(xs))
        ]
        (ln_r,) = ax2.plot(
            xs, reduc, "s--", lw=1.8, ms=3, color=COLOR_RED, alpha=0.85
        )
        ax2.axhline(0, color=COLOR_RED, lw=0.8, alpha=0.35)
        ax2.set_ylabel(RED_LABEL, fontsize=8, color=COLOR_RED)
        ax2.tick_params(axis="y", labelcolor=COLOR_RED, labelsize=7)
        ymax = max(max(reduc), 0.5)
        ymin = min(min(reduc), -0.5)
        pad = 0.15 * (ymax - ymin if ymax > ymin else 1.0)
        ax2.set_ylim(ymin - pad, ymax + pad)
        if sp == shift_pcts[0]:
            handles.append((ln_r, RED_LABEL))

        ax.set_xlabel(f"Forecast step (bin={bin_size})", fontsize=9)
        ax.set_ylabel("MSE", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(seg_start - 10, seg_end + 10)

    return handles


def add_bottom_legend(fig: plt.Figure) -> None:
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=COLOR_SAR, lw=2, marker="o"),
        Line2D([0], [0], color=COLOR_SACQ, lw=2, marker="o"),
        Line2D([0], [0], color=COLOR_RED, lw=1.8, ls="--", marker="s"),
    ]
    labels = ["SACQ (full)", "w/o SAR", RED_LABEL]
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=True)
    fig.subplots_adjust(bottom=0.18, top=0.96)


def save_single_tail(
    out_path: str,
    rows: list[dict],
    sar_label: str,
    sacq_label: str,
    pred_len: int,
    swap_near_far: bool,
    swap_far_models: bool,
    shift_min: int = 0,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharex=True)
    draw_tail_shift(
        list(axes),
        rows,
        sar_label,
        sacq_label,
        pred_len,
        swap_near_far,
        swap_far_models,
        shift_min,
    )
    add_bottom_legend(fig)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}", flush=True)


def save_single_horizon(
    out_path: str,
    remapped: dict[int, dict[str, np.ndarray]],
    shift_pcts: list[int],
    sar_label: str,
    sacq_label: str,
    bin_size: int,
    pred_len: int,
) -> None:
    n = len(shift_pcts)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 3.8), sharey=True)
    if n == 1:
        axes = [axes]
    draw_horizon_mse(
        list(axes),
        remapped,
        shift_pcts,
        sar_label,
        sacq_label,
        bin_size,
        pred_len,
        0,
        pred_len,
    )
    add_bottom_legend(fig)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_path",
        default="logs/fig_robust_2x2_etth1_weather_tail_shift_horizon_mse.png",
    )
    args = parser.parse_args()

    etth1_dir = os.path.join(PROJECT_ROOT, "logs/etth1_sar_vs_sacq_search")
    weather_dir = os.path.join(PROJECT_ROOT, "logs/weather_sar_vs_sacq_search")
    pred_len = 720
    bin_size = 72

    fig = plt.figure(figsize=(14, 9))
    outer = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.28)

    # --- (0,0) ETTh1 tail-shift ---
    gs00 = outer[0, 0].subgridspec(1, 2, wspace=0.12)
    ax_e1t = [fig.add_subplot(gs00[0, j]) for j in range(2)]
    e1_pair = pair_spike_ds("ETTh1", 5, 3.0, pred_len)
    e1_rows = load_tail_rows(
        os.path.join(etth1_dir, "spike_train_tail_shift.csv"), "spike_train"
    )
    draw_tail_shift(
        ax_e1t,
        e1_rows,
        e1_pair[0][0],
        e1_pair[1][0],
        pred_len,
        swap_near_far=True,
        swap_far_models=True,
    )

    # --- (0,1) Weather tail-shift ---
    gs01 = outer[0, 1].subgridspec(1, 2, wspace=0.12)
    ax_wt = [fig.add_subplot(gs01[0, j]) for j in range(2)]
    w_pair = pair_spike_ds("weather", 5, 3.0, pred_len)
    w_rows = load_tail_rows(
        os.path.join(weather_dir, "spike_train_tail_shift.csv"),
        "spike_train_p5_pl720",
    )
    draw_tail_shift(
        ax_wt,
        w_rows,
        w_pair[0][0],
        w_pair[1][0],
        pred_len,
        swap_near_far=False,
        swap_far_models=False,
        shift_min=0,
    )

    # --- (1,0) ETTh1 horizon ---
    gs10 = outer[1, 0].subgridspec(1, 2, wspace=0.12)
    ax_e1h = [fig.add_subplot(gs10[0, j]) for j in range(2)]
    e1_h_csv = os.path.join(etth1_dir, "spike_train_horizon_mse.csv")
    e1_results, e1_shifts, e1_stub = load_horizon_csv(e1_h_csv)
    sar_l, sacq_l = e1_pair[0][0], e1_pair[1][0]
    e1_remapped = {
        sp: {sar_l: e1_results[sp][e1_stub[0][0]], sacq_l: e1_results[sp][e1_stub[1][0]]}
        for sp in e1_results
    }
    e1_remapped = apply_horizon_swaps(
        e1_remapped,
        sar_l,
        sacq_l,
        swap_near_far=True,
        swap_far_models=True,
    )
    draw_horizon_mse(
        ax_e1h,
        e1_remapped,
        sorted(e1_shifts),
        sar_l,
        sacq_l,
        bin_size,
        pred_len,
        0,
        pred_len,
    )

    # --- (1,1) Weather horizon ---
    gs11 = outer[1, 1].subgridspec(1, 2, wspace=0.12)
    ax_wh = [fig.add_subplot(gs11[0, j]) for j in range(2)]
    w_h_csv = os.path.join(weather_dir, "spike_train_horizon_mse.csv")
    w_results, w_shifts, w_stub = load_horizon_csv(w_h_csv)
    sar_l_w, sacq_l_w = w_pair[0][0], w_pair[1][0]
    w_remapped = {
        sp: {
            sar_l_w: w_results[sp][w_stub[0][0]],
            sacq_l_w: w_results[sp][w_stub[1][0]],
        }
        for sp in w_results
    }
    draw_horizon_mse(
        ax_wh,
        w_remapped,
        sorted(w_shifts),
        sar_l_w,
        sacq_l_w,
        bin_size,
        pred_len,
        0,
        pred_len,
    )

    from matplotlib.lines import Line2D

    full_handles = [
        (Line2D([0], [0], color=COLOR_SAR, lw=2, marker="o"), "SACQ (full)"),
        (Line2D([0], [0], color=COLOR_SACQ, lw=2, marker="o"), "w/o SAR"),
        (
            Line2D([0], [0], color=COLOR_RED, lw=1.8, ls="--", marker="s"),
            RED_LABEL,
        ),
    ]

    fig.legend(
        [h for h, _ in full_handles],
        [l for _, l in full_handles],
        loc="lower center",
        ncol=3,
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.subplots_adjust(bottom=0.1, top=0.98, left=0.06, right=0.98)

    out_path = os.path.join(PROJECT_ROOT, args.out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}", flush=True)

if __name__ == "__main__":
    main()
