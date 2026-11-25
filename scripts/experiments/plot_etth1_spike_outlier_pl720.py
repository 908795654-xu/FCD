"""Plot MSE/MAE growth vs spike-outlier ratio (ETTh1 pl=720)."""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

VARIANT_ORDER = [
    "SAR+MSE",
    "SAR+S2",
    "SAR+logcosh",
    "Flatten+MSE",
    "Flatten+logcosh",
]
COLORS = {
    "SAR+MSE": "#1f77b4",
    "SAR+S2": "#9467bd",
    "SAR+logcosh": "#d62728",
    "Flatten+MSE": "#ff7f0e",
    "Flatten+logcosh": "#2ca02c",
}
LINESTYLES = {
    "SAR+MSE": "-",
    "SAR+S2": "-.",
    "SAR+logcosh": "-",
    "Flatten+MSE": "--",
    "Flatten+logcosh": ":",
}


def load_summary(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["status"] == "ok"].copy()
    df["outlier_pct"] = df["outlier_pct"].astype(float)
    df["mse"] = df["mse"].astype(float)
    df["mae"] = df["mae"].astype(float)
    return df.sort_values(["variant_tag", "outlier_pct"])


def _rel_series(sub: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray, float]:
    base = float(sub.loc[sub["outlier_pct"] == 0, metric].iloc[0])
    xs = sub["outlier_pct"].values.astype(float)
    ys = sub[metric].values.astype(float) / base
    return xs, ys, base


def plot_curves(
    df: pd.DataFrame,
    out_dir: str,
    metric: str,
    relative: bool = True,
    noise_mult: Optional[float] = None,
    ylim_pad: float = 0.03,
):
    fig, ax = plt.subplots(figsize=(8, 5))
    for tag in VARIANT_ORDER:
        sub = df[df["variant_tag"] == tag].sort_values("outlier_pct")
        if sub.empty or (sub["outlier_pct"] == 0).sum() == 0:
            continue
        xs, ys, _ = _rel_series(sub, metric) if relative else (
            sub["outlier_pct"].values,
            sub[metric].values.astype(float),
            1.0,
        )
        if not relative:
            xs = sub["outlier_pct"].values
            ys = sub[metric].values.astype(float)
        lw = 2.6 if tag in ("SAR+logcosh", "SAR+S2") else 1.8
        ax.plot(
            xs,
            ys,
            "o-",
            label=tag,
            color=COLORS[tag],
            lw=lw,
            ls=LINESTYLES[tag],
            ms=7,
        )

    ylab = f"测试 {metric.upper()}"
    if relative:
        ylab += " / 洁净基线"
        title_metric = f"{metric.upper()} 相对增长"
    else:
        title_metric = f"测试 {metric.upper()}"
    ax.set_xlabel("Train spike-outlier ratio (%)")
    ax.set_ylabel(ylab)
    nm = f", {noise_mult:g}×σ noise" if noise_mult else ""
    ax.set_title(f"ETTh1 pl=720: {title_metric} vs outlier ratio{nm}")
    ax.set_xticks(sorted(df["outlier_pct"].unique()))
    if relative and len(df) > 0:
        all_rel = []
        for tag in VARIANT_ORDER:
            sub = df[df["variant_tag"] == tag]
            if len(sub):
                _, ys, _ = _rel_series(sub.sort_values("outlier_pct"), metric)
                all_rel.extend(ys.tolist())
        lo, hi = min(all_rel), max(all_rel)
        ax.set_ylim(max(0.85, lo - ylim_pad), hi + ylim_pad)
        ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.grid(alpha=0.25)
    ax.legend()
    suffix = f"{metric}_relative" if relative else metric
    path = os.path.join(out_dir, f"spike_outlier_{suffix}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("wrote", path)


def smoothness_report(df: pd.DataFrame) -> str:
    """Second-difference magnitude on 1/3/5% points (lower = smoother curve)."""
    lines = ["\n=== 曲线平滑度（1/3/5% 二阶差分绝对值之和，越小越平滑）===\n"]
    for tag in VARIANT_ORDER:
        sub = df[df["variant_tag"] == tag].sort_values("outlier_pct")
        sub = sub[sub["outlier_pct"] > 0]
        if len(sub) < 3:
            continue
        for metric in ("mse", "mae"):
            base = float(df[(df["variant_tag"] == tag) & (df["outlier_pct"] == 0)][metric].iloc[0])
            rel = sub[metric].values / base
            d2 = np.abs(np.diff(rel, n=2))
            lines.append(f"  {tag} {metric}: {d2.sum():.6f}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        default="logs/etth1_spike_outlier_pl720/summary.csv",
    )
    parser.add_argument("--out_dir", default="logs/etth1_spike_outlier_pl720/figures")
    parser.add_argument("--noise_mult", type=float, default=None)
    args = parser.parse_args()

    csv_path = os.path.join(PROJECT_ROOT, args.summary)
    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    df = load_summary(csv_path)
    nm = args.noise_mult
    if nm is None and "noise_mult" in df.columns and len(df):
        nm = float(df["noise_mult"].iloc[0])

    plot_curves(df, out_dir, "mse", relative=True, noise_mult=nm)
    plot_curves(df, out_dir, "mae", relative=True, noise_mult=nm)
    plot_curves(df, out_dir, "mse", relative=False, noise_mult=nm)
    plot_curves(df, out_dir, "mae", relative=False, noise_mult=nm)

    # combined panel
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    panel_rel = {metric: [] for metric in ("mse", "mae")}
    for ax, metric in zip(axes, ("mse", "mae")):
        for tag in VARIANT_ORDER:
            sub = df[df["variant_tag"] == tag].sort_values("outlier_pct")
            if sub.empty or (sub["outlier_pct"] == 0).sum() == 0:
                continue
            xs, rel, _ = _rel_series(sub, metric)
            panel_rel[metric].extend(rel.tolist())
            lw = 2.6 if tag in ("SAR+logcosh", "SAR+S2") else 1.8
            ax.plot(
                xs,
                rel,
                "o-",
                label=tag,
                color=COLORS[tag],
                lw=lw,
                ls=LINESTYLES[tag],
                ms=6,
            )
        ax.set_xlabel("Train spike-outlier ratio (%)")
        ax.set_ylabel(f"Test {metric.upper()} / clean baseline")
        ax.set_title(metric.upper())
        lo, hi = min(panel_rel[metric]), max(panel_rel[metric])
        ax.set_ylim(max(0.85, lo - 0.03), hi + 0.03)
        ax.axhline(1.0, color="gray", ls=":", lw=1)
        ax.grid(alpha=0.25)
        ax.set_xticks(sorted(df["outlier_pct"].unique()))
    axes[0].legend(fontsize=8, ncol=2)
    combo = os.path.join(out_dir, "spike_outlier_mse_mae_panel.png")
    sub_nm = f" ({nm:g}×σ)" if nm else ""
    fig.suptitle(
        f"ETTh1 pl=720: spike-outlier robustness{sub_nm} (vs clean baseline)",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(combo, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", combo)

    report = smoothness_report(df)
    print(report)
    with open(os.path.join(out_dir, "smoothness_report.txt"), "w") as f:
        f.write(report)

    # degradation % panel
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric in zip(axes, ("mse", "mae")):
        for tag in VARIANT_ORDER:
            sub = df[df["variant_tag"] == tag].sort_values("outlier_pct")
            if sub.empty or (sub["outlier_pct"] == 0).sum() == 0:
                continue
            base = float(sub.loc[sub["outlier_pct"] == 0, metric].iloc[0])
            delta = (sub[metric].values / base - 1) * 100
            lw = 2.6 if tag in ("SAR+logcosh", "SAR+S2") else 1.8
            ax.plot(
                sub["outlier_pct"],
                delta,
                "o-",
                label=tag,
                color=COLORS[tag],
                lw=lw,
                ls=LINESTYLES[tag],
                ms=6,
            )
        ax.axhline(0, color="gray", ls=":", lw=1)
        ax.set_xlabel("Train spike-outlier ratio (%)")
        ax.set_ylabel(f"Test {metric.upper()} degradation vs clean (%)")
        ax.set_title(metric.upper())
        ax.grid(alpha=0.25)
        ax.set_xticks(sorted(df["outlier_pct"].unique()))
    axes[0].legend(fontsize=8)
    deg_path = os.path.join(out_dir, "spike_outlier_degradation_panel.png")
    sub_nm = f" ({nm:g}×σ)" if nm else ""
    fig.suptitle(f"ETTh1 pl=720: spike-outlier degradation{sub_nm}", y=1.02)
    fig.tight_layout()
    fig.savefig(deg_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", deg_path)


if __name__ == "__main__":
    main()
