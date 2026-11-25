"""Plot Horizon-Stratified Robustness results (SAR vs SACQ).

Reads CSV from eval_sar_vs_sacq_horizon_robust.py and produces:
  - segment_mse_{group}.png       near / mid / far MSE vs severity (3 panels)
  - horizon_slope_{group}.png     error slope bar at max non-clean severity
  - rho_index_{group}.png         reverse-contamination index ρ = Δnear/Δfar
  - summary_panel.png             2×2 combined figure for paper

Example::
    python scripts/experiments/plot_sar_vs_sacq_horizon_robust.py --split test
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.experiments.eval_sar_vs_sacq_horizon_robust import (
    CORRUPTION_MODES,
    MODEL_COLORS,
    horizon_bounds,
)

SEGMENTS = ("near", "mid", "far")
MODEL_TAGS = ("sar_slc", "sacq_slc")
MODEL_LABELS = {
    "sar_slc": "SAR+logcosh",
    "sacq_slc": "SACQ+logcosh",
}

GROUP_TITLES = {
    "tail_shift": "Tail level shift (last 50% history, OT)",
    "tail_spike": "Tail spike cluster (last 128 steps, OT)",
    "scaffold": "Far-half coarse scaffold noise (decoder)",
}


def _load_rows(out_dir: str, split: str, group: str | None = None) -> pd.DataFrame:
    if group is None:
        path = os.path.join(out_dir, f"results_{split}_horizon_stratified.csv")
    else:
        path = os.path.join(out_dir, f"results_{split}_{group}.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _clean_row(df: pd.DataFrame, tag: str, group: str) -> pd.Series | None:
    sub = df[(df["model_tag"] == tag) & (df["corruption_group"] == group) & (df["corruption"] == "clean")]
    if sub.empty:
        return None
    return sub.iloc[0]


def _group_subset(df: pd.DataFrame, group: str) -> pd.DataFrame:
    if "corruption_group" in df.columns:
        return df[df["corruption_group"] == group].copy()
    return df.copy()


def _severity_order(df: pd.DataFrame) -> list[str]:
    sub = df[df["corruption"] != "clean"].drop_duplicates(subset=["corruption", "severity"])
    sub = sub.sort_values("severity")
    return ["clean"] + sub["corruption"].tolist()


def _xlabels(group: str, order: list[str]) -> list[str]:
    labels = []
    for k in order:
        if k == "clean":
            labels.append("clean")
            continue
        if group == "tail_shift":
            labels.append(k.replace("tailshift_", "") + "%")
        elif group == "tail_spike":
            # tailspike_p5_k3 -> p5,k3σ
            parts = k.replace("tailspike_", "").split("_")
            labels.append(parts[0].replace("p", "") + "%,+" + parts[1].replace("k", "") + "σ")
        elif group == "scaffold":
            labels.append("ε=" + k.replace("scaffold_eps", "").replace("p", "."))
        else:
            labels.append(k)
    return labels


def compute_derived(df: pd.DataFrame, group: str) -> pd.DataFrame:
    """Add ΔMSE segments and ρ index relative to clean baseline."""
    rows = []
    for tag in MODEL_TAGS:
        base = _clean_row(df, tag, group)
        if base is None:
            continue
        sub = _group_subset(df, group)
        sub = sub[sub["model_tag"] == tag]
        for _, r in sub.iterrows():
            d_near = r["mse_near"] - base["mse_near"]
            d_mid = r["mse_mid"] - base["mse_mid"]
            d_far = r["mse_far"] - base["mse_far"]
            d_all = r["mse_all"] - base["mse_all"]
            rho = d_near / d_far if abs(d_far) > 1e-12 else np.nan
            slope = (r["mse_far"] - r["mse_near"]) / max(base["mse_near"], 1e-12)
            rows.append(
                {
                    "model_tag": tag,
                    "model_label": MODEL_LABELS[tag],
                    "corruption_group": group,
                    "corruption": r["corruption"],
                    "severity": r["severity"],
                    "delta_mse_near": d_near,
                    "delta_mse_mid": d_mid,
                    "delta_mse_far": d_far,
                    "delta_mse_all": d_all,
                    "rho": rho,
                    "horizon_slope": slope,
                    "mse_near": r["mse_near"],
                    "mse_mid": r["mse_mid"],
                    "mse_far": r["mse_far"],
                    "mse_all": r["mse_all"],
                }
            )
    return pd.DataFrame(rows)


def plot_segment_curves(df: pd.DataFrame, group: str, out_dir: str, pred_len: int, split: str):
    sub = _group_subset(df, group)
    order = _severity_order(sub)
    xlabels = _xlabels(group, order)
    xs = np.arange(len(order))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharex=True)
    seg_titles = {
        "near": f"Near [0, {horizon_bounds(pred_len)['near'][1]})",
        "mid": (
            f"Mid [{horizon_bounds(pred_len)['near'][1]}, "
            f"{horizon_bounds(pred_len)['mid'][1]})"
        ),
        "far": f"Far [{horizon_bounds(pred_len)['mid'][1]}, {pred_len})",
    }

    for ax, seg in zip(axes, SEGMENTS):
        for tag in MODEL_TAGS:
            msub = sub[sub["model_tag"] == tag]
            ys = []
            for k in order:
                row = msub[msub["corruption"] == k]
                ys.append(float(row.iloc[0][f"mse_{seg}"]) if len(row) else np.nan)
            ax.plot(xs, ys, "o-", lw=2, ms=7, color=MODEL_COLORS[tag], label=MODEL_LABELS[tag])
        ax.set_title(seg_titles[seg], fontsize=10)
        ax.set_ylabel("MSE")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(xs)
        ax.set_xticklabels(xlabels, rotation=35, ha="right", fontsize=8)

    fig.suptitle(
        f"Horizon-stratified MSE | {GROUP_TITLES.get(group, group)} | "
        f"ETTh1 sl=512 pl={pred_len} ({split})",
        fontsize=11,
        y=1.03,
    )
    axes[0].legend(fontsize=8, loc="best")
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"segment_mse_{group}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}", flush=True)


def plot_horizon_slope(derived: pd.DataFrame, group: str, out_dir: str, pred_len: int, split: str):
    corrupt = derived[derived["corruption"] != "clean"]
    if corrupt.empty:
        return
    max_sev = corrupt["severity"].max()
    sub = corrupt[np.isclose(corrupt["severity"], max_sev)]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    tags = [t for t in MODEL_TAGS if t in sub["model_tag"].values]
    xs = np.arange(len(tags))
    vals = [float(sub[sub["model_tag"] == t]["horizon_slope"].iloc[0]) for t in tags]
    colors = [MODEL_COLORS[t] for t in tags]
    ax.bar(xs, vals, color=colors, alpha=0.88, width=0.55)
    ax.set_xticks(xs)
    ax.set_xticklabels([MODEL_LABELS[t] for t in tags])
    ax.set_ylabel(r"Horizon slope $(\mathrm{MSE}_{far}-\mathrm{MSE}_{near})/\mathrm{MSE}_{clean,near}$")
    ax.set_title(f"Error growth along horizon @ max severity\n{GROUP_TITLES.get(group, group)}")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"horizon_slope_{group}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}", flush=True)


def plot_rho_index(derived: pd.DataFrame, group: str, out_dir: str, pred_len: int, split: str):
    corrupt = derived[derived["corruption"] != "clean"].copy()
    if corrupt.empty:
        return
    corrupt = corrupt.sort_values("severity")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for tag in MODEL_TAGS:
        sub = corrupt[corrupt["model_tag"] == tag]
        if sub.empty:
            continue
        ax.plot(
            sub["severity"],
            sub["rho"],
            "o-",
            lw=2,
            ms=7,
            color=MODEL_COLORS[tag],
            label=MODEL_LABELS[tag],
        )
    ax.axhline(1.0, color="k", ls="--", lw=0.8, label=r"$\rho=1$ (balanced)")
    ax.set_xlabel("Corruption severity")
    ax.set_ylabel(r"Reverse-contamination $\rho=\Delta\mathrm{MSE}_{near}/\Delta\mathrm{MSE}_{far}$")
    ax.set_title(
        f"Near vs far degradation ratio | {GROUP_TITLES.get(group, group)}\n"
        f"ETTh1 pl={pred_len} ({split}); ρ>1 ⇒ near hurt more than far"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"rho_index_{group}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}", flush=True)


def plot_delta_segments(derived: pd.DataFrame, group: str, out_dir: str, pred_len: int, split: str):
    corrupt = derived[derived["corruption"] != "clean"].copy()
    if corrupt.empty:
        return
    max_sev = corrupt["severity"].max()
    sub = corrupt[np.isclose(corrupt["severity"], max_sev)]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.25
    x = np.arange(len(SEGMENTS))
    for i, tag in enumerate(MODEL_TAGS):
        row = sub[sub["model_tag"] == tag]
        if row.empty:
            continue
        vals = [float(row.iloc[0][f"delta_mse_{s}"]) for s in SEGMENTS]
        ax.bar(x + (i - 0.5) * width, vals, width, color=MODEL_COLORS[tag], label=MODEL_LABELS[tag])
    ax.set_xticks(x)
    ax.set_xticklabels(["Near", "Mid", "Far"])
    ax.set_ylabel("ΔMSE vs clean")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(f"Segment degradation @ max severity | {GROUP_TITLES.get(group, group)}")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"delta_mse_{group}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}", flush=True)


def plot_summary_panel(df: pd.DataFrame, out_dir: str, pred_len: int, split: str):
    groups = [g for g in CORRUPTION_MODES if g in df["corruption_group"].unique()]
    if not groups:
        return

    n = len(groups)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8), squeeze=False)

    for col, group in enumerate(groups):
        derived = compute_derived(df, group)
        sub = _group_subset(df, group)
        order = _severity_order(sub)
        xs = np.arange(len(order))

        ax_seg = axes[0, col]
        for tag in MODEL_TAGS:
            msub = sub[sub["model_tag"] == tag]
            for seg, ls in zip(SEGMENTS, ["-", "--", ":"]):
                ys = []
                for k in order:
                    row = msub[msub["corruption"] == k]
                    ys.append(float(row.iloc[0][f"mse_{seg}"]) if len(row) else np.nan)
                ax_seg.plot(xs, ys, ls, lw=1.8, color=MODEL_COLORS[tag], label=f"{MODEL_LABELS[tag]} {seg}")
        ax_seg.set_title(GROUP_TITLES.get(group, group), fontsize=9)
        ax_seg.set_ylabel("MSE")
        ax_seg.grid(True, alpha=0.25)
        ax_seg.set_xticks(xs)
        ax_seg.set_xticklabels(_xlabels(group, order), rotation=40, ha="right", fontsize=7)

        ax_rho = axes[1, col]
        corrupt = derived[derived["corruption"] != "clean"]
        for tag in MODEL_TAGS:
            csub = corrupt[corrupt["model_tag"] == tag].sort_values("severity")
            if csub.empty:
                continue
            ax_rho.plot(
                csub["severity"],
                csub["rho"],
                "o-",
                lw=2,
                ms=6,
                color=MODEL_COLORS[tag],
                label=MODEL_LABELS[tag],
            )
        ax_rho.axhline(1.0, color="k", ls="--", lw=0.7)
        ax_rho.set_xlabel("Severity")
        ax_rho.set_ylabel(r"$\rho$")
        ax_rho.grid(True, alpha=0.25)
        if col == 0:
            ax_seg.legend(fontsize=6, ncol=2, loc="upper left")
            ax_rho.legend(fontsize=7)

    fig.suptitle(
        f"Horizon-Stratified Robustness: SAR vs SACQ | ETTh1 pl={pred_len} ({split})",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    out_path = os.path.join(out_dir, "summary_panel.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}", flush=True)


def plot_group(df: pd.DataFrame, group: str, out_dir: str, pred_len: int, split: str):
    plot_segment_curves(df, group, out_dir, pred_len, split)
    derived = compute_derived(df, group)
    derived_path = os.path.join(out_dir, f"derived_{split}_{group}.csv")
    derived.to_csv(derived_path, index=False)
    print(f"saved -> {derived_path}", flush=True)
    plot_horizon_slope(derived, group, out_dir, pred_len, split)
    plot_rho_index(derived, group, out_dir, pred_len, split)
    plot_delta_segments(derived, group, out_dir, pred_len, split)


def plot_all(out_dir: str, split: str, pred_len: int, groups: list[str] | None = None):
    df = _load_rows(out_dir, split)
    want = groups or [g for g in CORRUPTION_MODES if g in df["corruption_group"].unique()]
    for group in want:
        plot_group(df, group, out_dir, pred_len, split)
    if len(want) > 1:
        plot_summary_panel(df, out_dir, pred_len, split)


def main():
    parser = argparse.ArgumentParser(description="Plot SAR vs SACQ horizon-stratified robustness")
    parser.add_argument(
        "--out_dir",
        default="logs/etth1_sar_vs_sacq_horizon_robust",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--pred_len", type=int, default=720)
    parser.add_argument(
        "--group",
        type=str,
        default="all",
        choices=["all", "tail_shift", "tail_spike", "scaffold"],
    )
    args = parser.parse_args()

    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    groups = None if args.group == "all" else [args.group]
    plot_all(out_dir, args.split, args.pred_len, groups)


if __name__ == "__main__":
    main()
