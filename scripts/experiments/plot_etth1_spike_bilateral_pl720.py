"""Compare train-only vs bilateral spike CSV results (ETTh1 pl=720)."""
from __future__ import annotations

import argparse
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

VARIANT_ORDER = [
    "SAR+logcosh_scaled",
    "SACQ+logcosh_scaled",
    "Flatten+logcosh_scaled",
]
FACTORIAL_2X2_VARIANTS = [
    "SACQ+logcosh_scaled",
    "Flatten+logcosh_scaled",
]
COLORS = {
    "SAR+logcosh_scaled": "#2ca02c",
    "SACQ+logcosh_scaled": "#9467bd",
    "Flatten+logcosh_scaled": "#ff7f0e",
}


def load_ok(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df[df["status"] == "ok"].copy()


def plot_factorial_2x2(df: pd.DataFrame, out_dir: str, noise_mult: float, metric: str = "mse"):
    """2×2 train-vs-test spike grid (absolute MSE/MAE), shared colorbar."""
    os.makedirs(out_dir, exist_ok=True)
    variants = FACTORIAL_2X2_VARIANTS
    pcts = sorted(set(df["train_outlier_pct"].astype(float)) | set(df["test_outlier_pct"].astype(float)))

    grids: list[np.ndarray] = []
    for tag in variants:
        sub = df[df["variant_tag"] == tag]
        grid = np.full((len(pcts), len(pcts)), np.nan)
        for _, row in sub.iterrows():
            ti = pcts.index(float(row["train_outlier_pct"]))
            te = pcts.index(float(row["test_outlier_pct"]))
            grid[ti, te] = float(row[metric])
        grids.append(grid)

    vmin = float(np.nanmin(grids))
    vmax = float(np.nanmax(grids))

    fig, axes = plt.subplots(1, len(variants), figsize=(5.0 * len(variants) + 0.8, 4.8))
    if len(variants) == 1:
        axes = [axes]

    im = None
    for ax, tag, grid in zip(axes, variants, grids):
        im = ax.imshow(grid, cmap="YlOrRd", aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(pcts)))
        ax.set_yticks(range(len(pcts)))
        ax.set_xticklabels([f"{p:g}" for p in pcts])
        ax.set_yticklabels([f"{p:g}" for p in pcts])
        ax.set_xlabel("Test spike %")
        if ax is axes[0]:
            ax.set_ylabel("Train spike %")
        ax.set_title(tag.replace("+logcosh_scaled", ""))
        for i in range(len(pcts)):
            for j in range(len(pcts)):
                val = grid[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9, color="black")

    assert im is not None
    fig.subplots_adjust(right=0.88, wspace=0.28)
    cbar = fig.colorbar(im, ax=axes, fraction=0.05, pad=0.02)
    cbar.set_label(metric.upper())

    path = os.path.join(out_dir, f"bilateral_{metric}_factorial_2x2.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {path}")

    if metric == "mse":
        write_factorial_2x2_caption(out_dir, noise_mult)


def write_factorial_2x2_caption(out_dir: str, noise_mult: float) -> None:
    caption = f"""图：ETTh1 数据集（pred_len=720）上，训练段与测试段 CSV spike 污染的 2×2 因子实验结果。

实验设定
  · 在 ETTh1 原始 CSV 中，按 2×2 因子分别控制训练段（前 12 个月）与测试段（后 4 个月）的 spike 污染比例（0% 或 5%）。
  · 每个被污染点叠加 ±{noise_mult:g}× 该段通道标准差的随机脉冲；验证段保持干净。
  · 标准化仍基于洁净训练数据拟合；图中颜色与数值为测试集 MSE（对污染后的测试标签计算）。
  · 左：SACQ；右：Flatten。两子图共用同一颜色标尺。

如何读图
  · 纵轴 = 训练 CSV 的 spike 比例；横轴 = 测试 CSV 的 spike 比例。
  · 每个格子给出对应 (train%, test%) 组合下的 Test MSE；颜色越深表示误差越大。

主要结论
  1. 横轴（测试污染）主导 MSE 上升：test 从 0%→5% 时 MSE 约从 0.43 升至 0.78–0.81，说明测试段数据质量是主要瓶颈。
  2. 纵轴（仅训练污染）对 clean test 影响极小：train 5% / test 0% 与全洁净接近，说明在脏数据上训练并不损害干净测试。
  3. 双侧污染（5%/5%）略优于仅测试污染（0%/5%），表明训练阶段见过 spike 对部署端也脏时有边际缓解。
  4. 四个条件下 SACQ 的 MSE 均低于 Flatten，说明结构化解码头在非理想 CSV 数据下仍具一致优势。

论文 caption 示例（英文）
  Factorial ablation of CSV-level spike outliers (±{noise_mult:g}σ, 5% of scalars) on ETTh1 with pred_len=720.
  Rows/columns denote train/test contamination rates; validation remains clean. Cell values are test MSE
  against polluted test labels. SACQ consistently achieves lower MSE than Flatten across all four conditions;
  test-side corruption drives most of the degradation.
"""
    path = os.path.join(out_dir, "bilateral_mse_factorial_2x2_caption.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(caption)
    print(f"saved -> {path}")


def plot_factorial_bars(df: pd.DataFrame, out_dir: str, noise_mult: float, metric: str = "mse"):
    """Grouped bars: one panel per (train,test) combo, 3 models compared."""
    os.makedirs(out_dir, exist_ok=True)
    combos = (
        df[["train_outlier_pct", "test_outlier_pct"]]
        .drop_duplicates()
        .sort_values(["train_outlier_pct", "test_outlier_pct"])
    )
    labels = [
        f"tr{int(r.train_outlier_pct)}/te{int(r.test_outlier_pct)}"
        for r in combos.itertuples()
    ]
    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for k, tag in enumerate(VARIANT_ORDER):
        ys = []
        for r in combos.itertuples():
            row = df[
                (df["variant_tag"] == tag)
                & (df["train_outlier_pct"].astype(float) == float(r.train_outlier_pct))
                & (df["test_outlier_pct"].astype(float) == float(r.test_outlier_pct))
            ]
            ys.append(float(row[metric].iloc[0]) if len(row) else np.nan)
        ax.bar(x + (k - 1) * w, ys, w, label=tag.replace("+logcosh_scaled", ""), color=COLORS[tag])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"Test {metric.upper()}")
    ax.set_title(f"ETTh1 pl=720 | Train×Test CSV spike (±{noise_mult:g}σ)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = os.path.join(out_dir, f"bilateral_{metric}_factorial_bars.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {path}")


def plot_bilateral_summary(df: pd.DataFrame, out_dir: str, noise_mult: float):
    """MSE vs train/test spike pct (square grid when both vary)."""
    os.makedirs(out_dir, exist_ok=True)
    for metric in ("mse", "mae"):
        fig, axes = plt.subplots(1, len(VARIANT_ORDER), figsize=(5 * len(VARIANT_ORDER), 4), sharey=True)
        if len(VARIANT_ORDER) == 1:
            axes = [axes]
        for ax, tag in zip(axes, VARIANT_ORDER):
            sub = df[df["variant_tag"] == tag]
            if sub.empty:
                continue
            # same train & test pct on diagonal
            diag = sub[
                sub["train_outlier_pct"].astype(float) == sub["test_outlier_pct"].astype(float)
            ].sort_values("train_outlier_pct")
            if diag.empty:
                continue
            base = diag[diag["train_outlier_pct"].astype(float) == 0]
            base_val = float(base[metric].iloc[0]) if len(base) else float(diag[metric].iloc[0])
            xs = diag["train_outlier_pct"].astype(float).values
            ys = diag[metric].astype(float).values / base_val
            ax.plot(xs, ys, "o-", color=COLORS[tag], lw=2)
            ax.axhline(1.0, color="gray", ls="--", lw=0.8)
            ax.set_title(tag)
            ax.set_xlabel("Train & test spike %")
            ax.set_ylabel(f"Relative {metric.upper()}")
            ax.grid(alpha=0.25)
        fig.suptitle(
            f"ETTh1 pl=720 bilateral CSV spike (±{noise_mult:g}σ, val clean)",
            fontsize=11,
        )
        fig.tight_layout()
        path = os.path.join(out_dir, f"bilateral_{metric}_vs_pct.png")
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"saved -> {path}")


def plot_train_only_vs_bilateral(
    train_only: pd.DataFrame,
    bilateral: pd.DataFrame,
    out_dir: str,
    pct: float,
    noise_mult: float,
):
    """Bar chart at fixed pct: train-only (test clean) vs bilateral (test also spiked)."""
    os.makedirs(out_dir, exist_ok=True)
    bi = bilateral[
        (bilateral["train_outlier_pct"].astype(float) == pct)
        & (bilateral["test_outlier_pct"].astype(float) == pct)
    ]
    if bi.empty:
        print(f"skip compare pct={pct}: no bilateral rows")
        return

    to_tag_map = {
        "SAR+logcosh_scaled": "SAR+logcosh",
        "SACQ+logcosh_scaled": None,
        "Flatten+logcosh_scaled": "Flatten+logcosh",
    }
    labels = []
    mse_train_only = []
    mse_bilateral = []
    for tag in VARIANT_ORDER:
        sub_bi = bi[bi["variant_tag"] == tag]
        if sub_bi.empty:
            continue
        to_tag = to_tag_map.get(tag)
        sub_to = pd.DataFrame()
        if to_tag:
            sub_to = train_only[train_only["outlier_pct"].astype(float) == pct]
            sub_to = sub_to[sub_to["variant_tag"] == to_tag]
        labels.append(tag.replace("+logcosh_scaled", ""))
        mse_bilateral.append(float(sub_bi["mse"].iloc[0]))
        mse_train_only.append(float(sub_to["mse"].iloc[0]) if len(sub_to) else np.nan)

    if not labels:
        return
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, mse_train_only, w, label="Train spike only (clean test CSV)", color="#aec7e8")
    ax.bar(x + w / 2, mse_bilateral, w, label="Bilateral CSV (train+test spike)", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Test MSE")
    ax.set_title(
        f"ETTh1 pl=720 @ {pct:g}% spike (±{noise_mult:g}σ): "
        "train-only vs bilateral CSV"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = os.path.join(out_dir, f"train_only_vs_bilateral_p{int(pct)}.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bilateral_csv",
        default="logs/etth1_spike_bilateral_pl720/summary.csv",
    )
    parser.add_argument(
        "--train_only_csv",
        default="logs/etth1_spike_outlier_pl720/summary.csv",
        help="optional; for train-only vs bilateral bar chart",
    )
    parser.add_argument("--out_dir", default="logs/etth1_spike_bilateral_pl720/figures")
    parser.add_argument("--noise_mult", type=float, default=3.0)
    parser.add_argument("--compare_pct", type=float, default=5.0)
    args = parser.parse_args()

    bilateral_path = os.path.join(PROJECT_ROOT, args.bilateral_csv)
    if not os.path.exists(bilateral_path):
        raise SystemExit(f"Missing {bilateral_path}; run training first.")

    bi = load_ok(bilateral_path)
    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    for metric in ("mse", "mae"):
        plot_factorial_2x2(bi, out_dir, args.noise_mult, metric=metric)
        plot_factorial_bars(bi, out_dir, args.noise_mult, metric=metric)
    plot_bilateral_summary(bi, out_dir, args.noise_mult)

    train_only_path = os.path.join(PROJECT_ROOT, args.train_only_csv)
    if os.path.exists(train_only_path):
        to = load_ok(train_only_path)
        plot_train_only_vs_bilateral(to, bi, out_dir, args.compare_pct, args.noise_mult)


if __name__ == "__main__":
    main()
