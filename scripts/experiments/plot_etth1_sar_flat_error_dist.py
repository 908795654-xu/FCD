#!/usr/bin/env python3
"""Plot test-set residual (pred-true) distributions for ETTh1 pl720 models."""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MODELS_BY_PRED_LEN = {
    720: {
        "SAR + logcosh (sl=512)": {
            "setting": "H1s512p720SARslc_PatchTST_ETTh1_sl512_pl720_hSARslc_0",
            "color": "#d62728",
        },
        "Flatten + logcosh (sl=512)": {
            "setting": "ETTh1_512_720_PatchTST_ETTh1_sl512_pl720_hFlatslc_0",
            "color": "#2ca02c",
        },
        "Flatten + MSE (sl=512)": {
            "setting": "ETTh1_512_720_PatchTST_ETTh1_sl512_pl720_Exp_0",
            "color": "#1f77b4",
        },
    },
    96: {
        "SAR + logcosh (sl=512)": {
            "setting": "H1s512p96SARslc_PatchTST_ETTh1_sl512_pl96_hSARslc_0",
            "color": "#d62728",
        },
        "Flatten + logcosh (sl=512)": {
            "setting": "ETTh1_512_96_PatchTST_ETTh1_sl512_pl96_hFlatslc_0",
            "color": "#2ca02c",
        },
        "Flatten + MSE (sl=512)": {
            "setting": "ETTh1_512_96_PatchTST_ETTh1_sl512_pl96_Exp_0",
            "color": "#1f77b4",
        },
    },
}


def load_residuals(setting: str) -> np.ndarray:
    root = os.path.join(PROJECT_ROOT, "results", setting)
    pred = np.load(os.path.join(root, "pred.npy"))
    true = np.load(os.path.join(root, "true.npy"))
    return (pred - true).reshape(-1)


def subsample(x: np.ndarray, max_n: int, seed: int = 2021) -> np.ndarray:
    if x.size <= max_n:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.size, size=max_n, replace=False)
    return x[idx]


def plot_error_distributions(
    models: dict,
    pred_len: int,
    out_dir: str,
    bins: int = 120,
    xlim: tuple[float, float] | None = None,
    kde_n: int = 200_000,
):
    os.makedirs(out_dir, exist_ok=True)
    residuals = {}
    for label, meta in models.items():
        residuals[label] = load_residuals(meta["setting"])

    # --- 1) residual histogram + KDE ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    all_err = np.concatenate(list(residuals.values()))
    if xlim is None:
        lo, hi = np.percentile(all_err, [0.5, 99.5])
        pad = 0.05 * (hi - lo)
        xlim = (lo - pad, hi + pad)

    for label, err in residuals.items():
        color = models[label]["color"]
        ax.hist(
            err,
            bins=bins,
            range=xlim,
            density=True,
            alpha=0.35,
            color=color,
            label=label,
        )
        xs = np.linspace(xlim[0], xlim[1], 400)
        kde = stats.gaussian_kde(subsample(err, kde_n), bw_method=0.15)
        ax.plot(xs, kde(xs), color=color, lw=2.0)

    ax.axvline(0.0, color="black", ls=":", lw=1.0, alpha=0.5)
    ax.set_xlim(xlim)
    ax.set_xlabel("residual (pred - true, scaled)")
    ax.set_ylabel("density")
    ax.set_title(f"ETTh1 pl={pred_len} test residual distribution")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # --- 2) |residual| (absolute error) ---
    ax = axes[1]
    abs_lim = np.percentile(np.abs(all_err), 99.5) * 1.05
    abs_bins = np.linspace(0, abs_lim, bins)
    for label, err in residuals.items():
        color = models[label]["color"]
        ae = np.abs(err)
        ax.hist(ae, bins=abs_bins, density=True, alpha=0.35, color=color, label=label)
        xs = np.linspace(0, abs_lim, 400)
        kde = stats.gaussian_kde(subsample(ae, kde_n), bw_method=0.12)
        ax.plot(xs, kde(xs), color=color, lw=2.0)
        ax.axvline(np.median(ae), color=color, ls="--", lw=1.0, alpha=0.8)

    ax.set_xlabel("|residual|")
    ax.set_ylabel("density")
    ax.set_title("absolute error (dashed = median)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    p1 = os.path.join(out_dir, "fig_error_dist_residual_abs.png")
    fig.savefig(p1, dpi=160)
    plt.close(fig)

    # --- 3) squared error (per-element) ---
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    se_lim = np.percentile(np.concatenate([e**2 for e in residuals.values()]), 99.5) * 1.05
    se_bins = np.linspace(0, se_lim, bins)
    for label, err in residuals.items():
        color = models[label]["color"]
        se = err**2
        ax.hist(se, bins=se_bins, density=True, alpha=0.35, color=color, label=label)
        xs = np.linspace(0, se_lim, 400)
        kde = stats.gaussian_kde(subsample(se, kde_n), bw_method=0.12)
        ax.plot(xs, kde(xs), color=color, lw=2.0)
    ax.set_xlabel("squared error (pred-true)^2")
    ax.set_ylabel("density")
    ax.set_title("per-element squared error")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p2 = os.path.join(out_dir, "fig_error_dist_squared.png")
    fig.savefig(p2, dpi=160)
    plt.close(fig)

    # --- 4) horizon-wise mean |error| ---
    fig, ax = plt.subplots(figsize=(8, 4))
    h = np.arange(1, pred_len + 1)
    for label, meta in models.items():
        root = os.path.join(PROJECT_ROOT, "results", meta["setting"])
        pred = np.load(os.path.join(root, "pred.npy"))
        true = np.load(os.path.join(root, "true.npy"))
        mae_h = np.mean(np.abs(pred - true), axis=(0, 2))
        ax.plot(h, mae_h, color=meta["color"], lw=1.5, label=label)
    ax.set_xlabel("horizon step")
    ax.set_ylabel("mean |residual|")
    ax.set_title("MAE vs horizon (avg over samples & channels)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p3 = os.path.join(out_dir, "fig_error_dist_by_horizon.png")
    fig.savefig(p3, dpi=160)
    plt.close(fig)

    # summary csv
    csv_path = os.path.join(out_dir, "error_dist_summary.csv")
    with open(csv_path, "w") as f:
        f.write("model,n,mean,std,median,mae,rmse,p95_abs,p99_abs\n")
        for label, err in residuals.items():
            ae = np.abs(err)
            f.write(
                f"{label},{err.size},{err.mean():.6f},{err.std():.6f},"
                f"{np.median(err):.6f},{ae.mean():.6f},{np.sqrt(np.mean(err**2)):.6f},"
                f"{np.percentile(ae,95):.6f},{np.percentile(ae,99):.6f}\n"
            )

    print("saved:", p1, p2, p3, csv_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_len", type=int, default=720, choices=[96, 720])
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--bins", type=int, default=120)
    args = parser.parse_args()
    models = MODELS_BY_PRED_LEN[args.pred_len]
    out = args.out_dir or f"logs/etth1_pl{args.pred_len}_sar_vs_flat_error_dist"
    out = os.path.join(PROJECT_ROOT, out)
    plot_error_distributions(models, args.pred_len, out, bins=args.bins)


if __name__ == "__main__":
    main()
