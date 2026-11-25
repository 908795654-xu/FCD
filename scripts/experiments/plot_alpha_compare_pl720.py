"""Side-by-side: Zc01 (alpha~0) vs A01g (alpha_init=0.1 + gate_loss) on pl=720."""
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.experiments import plot_sacq_curves as psc

OUT = os.path.join(PROJECT_ROOT, "logs/etth2_alpha_lr_boost/curves")
OLD = "TwS2Zc01A20s512p720_PatchTST_ETTh2_sl512_pl720_hA20_0"
NEW = "TwS2Zc01A01g_PatchTST_ETTh2_sl512_pl720_hA01g_0"
PL = 720


def main():
    os.makedirs(OUT, exist_ok=True)
    args = type(
        "A",
        (),
        {
            "seed": 2021,
            "batch_size": 128,
            "num_workers": 4,
            "lr": 1e-4,
            "e_layers": 3,
            "n_heads": 4,
            "d_model": 16,
            "d_ff": 128,
            "dropout": 0.3,
            "fc_dropout": 0.3,
            "patch_len": 16,
            "stride": 8,
        },
    )()

    psc.run_setting(OLD, PL, OUT, -1, 0, 0, out_basename="pl720_old_zc01")
    psc.run_setting(NEW, PL, OUT, -1, 0, 0, out_basename="pl720_new_a01g")

    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    for ax, sub, title in zip(
        axes,
        ["pl720_old_zc01", "pl720_new_a01g"],
        [
            "Before: Zc01 + α_init≈1e-4 (α LR×2) — output ~1e-8, flat α",
            "After: Zc01 + α_init=0.1 + gate_loss×0.3 + sign×0.15",
        ],
    ):
        ax.imshow(mpimg.imread(os.path.join(OUT, f"{sub}.png")))
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle("ETTh2 pl=720: truth / pred / α (ch0, max-α-var sample)", fontsize=12)
    fig.tight_layout()
    cmp_path = os.path.join(OUT, "compare_pl720_alpha_init_vs_gate.png")
    fig.savefig(cmp_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {cmp_path}")


if __name__ == "__main__":
    main()
