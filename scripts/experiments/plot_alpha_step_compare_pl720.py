"""Compare patch-alpha A01g vs per-step alpha A01gSt on pl=720."""
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.experiments import plot_sacq_curves as psc

OUT = os.path.join(PROJECT_ROOT, "logs/etth2_alpha_lr_boost/curves")
PATCH = "TwS2Zc01A01g_PatchTST_ETTh2_sl512_pl720_hA01g_0"
STEP = "TwS2Zc01A01gSt_PatchTST_ETTh2_sl512_pl720_hA01gSt_0"
PL = 720

METRICS = {
    "Flatten (TwF2)": 0.383078,
    "Zc01 patch-α": 0.382145,
    "A01g patch-α": 0.383335,
    "A01gSt step-α": 0.388084,
}


def main():
    os.makedirs(OUT, exist_ok=True)
    psc.run_setting(PATCH, PL, OUT, -1, 0, 0, out_basename="pl720_a01g_patch")
    psc.run_setting(STEP, PL, OUT, -1, 0, 0, out_basename="pl720_a01g_step")

    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    for ax, sub, title in zip(
        axes,
        ["pl720_a01g_patch", "pl720_a01g_step"],
        [
            "Patch-α: one gate per output patch (16 steps shared)",
            "Step-α: learnable gate at every forecast timestep",
        ],
    ):
        ax.imshow(mpimg.imread(os.path.join(OUT, f"{sub}.png")))
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    lines = [f"{k}: test MSE={v:.6f}" for k, v in METRICS.items()]
    fig.suptitle(
        "ETTh2 pl=720 | α_init=0.1 + gate×0.3 + sign×0.15\n" + "  |  ".join(lines),
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    cmp_path = os.path.join(OUT, "compare_pl720_patch_vs_step_alpha.png")
    fig.savefig(cmp_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {cmp_path}")


if __name__ == "__main__":
    main()
