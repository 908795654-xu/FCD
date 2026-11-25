"""Sweep drift intensity; plot MSE vs level_mult_end (Flatten vs SACQ variants)."""
from __future__ import annotations

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
    ensure_dir,
    make_shifted_dataset,
)
from scripts.experiments.shift_utils import drift_strength_profile
import torch

LEVEL_ENDS = np.linspace(1.05, 1.65, 13)
PRESET_BASE = {
    "shift_type": "gradual_ramp",
    "shift_ratio": 0.72,
    "old_slope": 0.1,
    "slope_end": -0.22,
}


def main():
    out_dir = os.path.join(PROJECT_ROOT, "logs/etth2_drift_shift", "intensity_sweep")
    ensure_dir(out_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    shift_idx = int(0.72 * 17420)

    variants = VARIANTS
    curves = {v["tag"]: [] for v in variants}

    for lev in LEVEL_ENDS:
        overrides = dict(PRESET_BASE, level_mult_end=float(lev))
        _, n_total, dst_csv, info = make_shifted_dataset(
            "ETTh2",
            "gradual_ramp",
            dst_name=f"ETTh2_shift_ramp_le{lev:.2f}.csv",
            root_path=os.path.join(PROJECT_ROOT, "dataset"),
            overrides=overrides,
        )
        data_path = os.path.basename(dst_csv)
        drift_t, _, drift_strength = drift_strength_profile(shift_idx, n_total, info["params"])
        mean_strength = float(drift_strength.mean())
        print(f"level_end={lev:.2f} mean_strength={mean_strength:.3f}")

        for variant in variants:
            ckpt = os.path.join(PROJECT_ROOT, "checkpoints", variant["setting"], "checkpoint.pth")
            res = evaluate_variant(variant, ckpt, device, data_path, shift_idx)
            post = res["t"] >= shift_idx
            post_mse = float(res["mse_t"][post].mean())
            pre = res["t"] < shift_idx
            pre_mse = float(res["mse_t"][pre].mean())
            curves[variant["tag"]].append(
                {"level_end": lev, "strength": mean_strength, "post_mse": post_mse, "ratio": post_mse / pre_mse}
            )

    fig, ax = plt.subplots(figsize=(10, 5))
    for variant in variants:
        tag = variant["tag"]
        xs = [c["strength"] for c in curves[tag]]
        ys = [c["ratio"] for c in curves[tag]]
        lw = 2.6 if tag == "Flatten" else 1.8
        ax.plot(xs, ys, "o-", label=tag, color=variant["color"], lw=lw, ms=5)
    ax.set_xlabel("平均漂移强度（测试段）")
    ax.set_ylabel("漂移后 MSE / 漂移前 MSE")
    ax.set_title("ETTh2 pl=720：随漂移加剧的相对恶化（洁净训练）")
    ax.grid(alpha=0.25)
    ax.legend()
    path = os.path.join(out_dir, "drift_intensity_sweep.png")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
