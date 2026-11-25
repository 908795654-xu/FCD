"""Final drift figure: Flatten (clean) vs SACQ clean vs SACQ drift-adapted."""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.experiments.run_etth2_drift_shift import (
    VARIANTS,
    evaluate_variant,
    make_shifted_dataset,
    drift_strength_profile,
)
import torch

OUT = os.path.join(PROJECT_ROOT, "logs/etth2_drift_shift", "gradual_ramp")
SHIFT_CSV = "ETTh2_shift_gradual_ramp.csv"

# 主对比：Flatten vs S1/S2/S12（洁净训练）+ S12d（漂移微调）
MAIN_TAGS = ["Flatten", "S1", "S2", "S12", "S1d", "S2d", "S12d"]


def smooth(y, win=41):
    if len(y) < win:
        return y
    k = np.ones(win) / win
    return np.convolve(y, k, mode="same")


def main():
    shift_idx, n_total, _, info = make_shifted_dataset(
        "ETTh2", "gradual_ramp", root_path=os.path.join(PROJECT_ROOT, "dataset")
    )
    drift_t, _, drift_strength = drift_strength_profile(shift_idx, n_total, info["params"])
    strength_map = {int(t): float(s) for t, s in zip(drift_t, drift_strength)}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    curves = {}
    for v in VARIANTS:
        if v["tag"] not in MAIN_TAGS:
            continue
        ckpt = os.path.join(PROJECT_ROOT, "checkpoints", v["setting"], "checkpoint.pth")
        if not os.path.isfile(ckpt):
            print("skip", v["tag"])
            continue
        print("eval", v["tag"])
        curves[v["tag"]] = {**v, **evaluate_variant(v, ckpt, device, SHIFT_CSV, shift_idx)}

    # clean baseline for ratio
    clean = {}
    for tag in MAIN_TAGS:
        if tag not in curves:
            continue
        v = curves[tag]
        ckpt = os.path.join(PROJECT_ROOT, "checkpoints", v["setting"], "checkpoint.pth")
        clean[tag] = evaluate_variant(v, ckpt, device, "ETTh2.csv", shift_idx)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for tag in MAIN_TAGS:
        if tag not in curves:
            continue
        c = curves[tag]
        post = c["t"] >= shift_idx
        ax.plot(
            c["t"][post] - shift_idx,
            smooth(c["mse_t"][post]),
            label=tag,
            color=c["color"],
            ls=c["ls"],
            lw=2.4 if tag == "Flatten" else 1.6,
        )
    ax.set_xlabel("漂移后时间步")
    ax.set_ylabel("逐步 MSE（平滑）")
    ax.set_title("渐进漂移测试：绝对误差")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1]
    for tag in MAIN_TAGS:
        if tag not in curves or tag not in clean:
            continue
        cs, cc = curves[tag], clean[tag]
        post = cs["t"] >= shift_idx
        ratio = cs["mse_t"][post] / np.maximum(cc["mse_t"][post], 1e-8)
        ax.plot(
            cs["t"][post] - shift_idx,
            smooth(ratio),
            label=tag,
            color=cs["color"],
            ls=cs["ls"],
            lw=2.4 if tag == "Flatten" else 1.6,
        )
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.set_xlabel("漂移后时间步")
    ax.set_ylabel("MSE(漂移) / MSE(洁净)")
    ax.set_title("相对洁净测试的恶化倍数")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)

    fig.suptitle(
        f"ETTh2 pl=720 | 漂移起点={shift_idx} | gradual_ramp (shift_ratio=0.72)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT, "drift_final_flatten_vs_sacq.png")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)

    # MSE vs drift strength (binned)
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(min(drift_strength), max(drift_strength), 30)
    bc = 0.5 * (bins[:-1] + bins[1:])
    for tag in MAIN_TAGS:
        if tag not in curves or tag not in clean:
            continue
        cs, cc = curves[tag], clean[tag]
        post = cs["t"] >= shift_idx
        ds = np.array([strength_map.get(int(t), 0.0) for t in cs["t"][post]])
        rat = cs["mse_t"][post] / np.maximum(cc["mse_t"][post], 1e-8)
        means = []
        for i in range(len(bc)):
            m = (ds >= bins[i]) & (ds < bins[i + 1])
            means.append(float(rat[m].mean()) if m.any() else np.nan)
        ax.plot(bc, means, "o-", label=tag, color=cs["color"], ls=cs["ls"], lw=2 if tag == "Flatten" else 1.4)
    ax.set_xlabel("漂移强度")
    ax.set_ylabel("MSE 恶化倍数（相对洁净）")
    ax.set_title("随漂移加剧：Flatten 应更陡，SACQ 更缓")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    out2 = os.path.join(OUT, "drift_final_vs_intensity.png")
    fig.tight_layout()
    fig.savefig(out2, dpi=170)
    plt.close(fig)

    rows = []
    for tag in MAIN_TAGS:
        if tag not in curves:
            continue
        cs, cc = curves[tag], clean[tag]
        post = cs["t"] >= shift_idx
        rows.append(
            {
                "tag": tag,
                "mse_shift": float(cs["mse"]),
                "mse_clean": float(cc["mse"]),
                "post_ratio_mean": float((cs["mse_t"][post] / np.maximum(cc["mse_t"][post], 1e-8)).mean()),
            }
        )
    with open(os.path.join(OUT, "final_metrics.json"), "w") as f:
        json.dump({"shift_idx": shift_idx, "models": rows}, f, indent=2)
    print("wrote", out, out2)


if __name__ == "__main__":
    main()
