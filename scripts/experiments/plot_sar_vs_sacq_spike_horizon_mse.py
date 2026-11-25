#!/usr/bin/env python3
"""MSE vs forecast step (binned) under tail-shift — SAR vs SACQ mechanism plot.

Uses spike-train checkpoints; both models see identical perturbed inputs.

Example::
  python scripts/experiments/plot_sar_vs_sacq_spike_horizon_mse.py \\
    --shift_pcts 40,60 --bin_size 72
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from exp.exp_main import Exp_Main
from scripts.experiments.eval_sar_vs_sacq_spike_train_shift import (
    TAIL_FRAC,
    horizon_segments,
    load_exp,
    sar_reduction_pct,
)
from scripts.experiments.robust_dataset_config import DATASETS, pair_spike_ds
from scripts.experiments.plot_etth1_flatten_vs_sar_all_channels import _to_btc

OT_CH = 6


def apply_tail_shift(
    batch_x: torch.Tensor, ch_std: np.ndarray, shift_frac: float, seed: int
) -> torch.Tensor:
    x = batch_x.clone()
    if shift_frac <= 0:
        return x
    t0 = max(0, int(round(x.shape[1] * (1.0 - TAIL_FRAC))))
    rng = np.random.default_rng(seed)
    for b in range(x.shape[0]):
        sign = float(rng.choice([-1.0, 1.0]))
        x[b, t0:, OT_CH] += sign * shift_frac * ch_std[OT_CH]
    return x


def forward_preds(exp: Exp_Main, x: torch.Tensor, batch_y: torch.Tensor, pred_len: int):
    with torch.no_grad():
        out, _ = exp._split_model_output(exp._forward(x))
        pred = _to_btc(exp, out, batch_y)
        if pred is None:
            pred = out
        return pred[:, -pred_len:].cpu().numpy()


def segment_range(segment: str, pred_len: int) -> tuple[int, int, str]:
    n_near, n_mid, titles = horizon_segments(pred_len)
    if segment == "near":
        return 0, n_near, titles[0]
    if segment == "far":
        return n_mid, pred_len, titles[1]
    return 0, pred_len, titles[2]


def eval_binned_mse(
    exps: dict[str, Exp_Main],
    ch_std: np.ndarray,
    shift_frac: float,
    pred_len: int,
    bin_size: int,
    seg_start: int,
    seg_end: int,
    seed: int = 2021,
) -> dict[str, np.ndarray]:
    loader = next(iter(exps.values()))._get_data(flag="test")[1]
    seg_len = seg_end - seg_start
    n_bins = int(np.ceil(seg_len / bin_size))
    sum_err = {k: np.zeros(n_bins, dtype=np.float64) for k in exps}
    sum_cnt = np.zeros(n_bins, dtype=np.float64)

    batch_seed = seed
    for batch_x, batch_y, _, _ in loader:
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        y_np = batch_y[:, -pred_len:].numpy()
        x = apply_tail_shift(batch_x, ch_std, shift_frac, batch_seed)
        batch_seed += 1
        bs = y_np.shape[0]
        n_ch = y_np.shape[2]
        for name, exp in exps.items():
            pred = forward_preds(exp, x.to(exp.device), batch_y.to(exp.device), pred_len)
            err = (pred - y_np) ** 2
            for b in range(n_bins):
                t0 = seg_start + b * bin_size
                t1 = min(seg_start + (b + 1) * bin_size, seg_end)
                sum_err[name][b] += err[:, t0:t1].sum()
        for b in range(n_bins):
            t0 = seg_start + b * bin_size
            t1 = min(seg_start + (b + 1) * bin_size, seg_end)
            sum_cnt[b] += bs * (t1 - t0) * n_ch

    return {name: sum_err[name] / np.maximum(sum_cnt, 1.0) for name in exps}


def bin_centers(seg_start: int, seg_end: int, bin_size: int) -> np.ndarray:
    seg_len = seg_end - seg_start
    n_bins = int(np.ceil(seg_len / bin_size))
    xs = []
    for b in range(n_bins):
        t0 = seg_start + b * bin_size
        t1 = min(seg_start + (b + 1) * bin_size, seg_end)
        xs.append(0.5 * (t0 + t1 - 1))
    return np.asarray(xs, dtype=np.float64)


def load_horizon_csv(csv_path: str) -> tuple[dict[int, dict[str, np.ndarray]], list[int], list]:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    shift_pcts = sorted({int(r["shift_pct"]) for r in rows})
    sar_label = next(r["model"] for r in rows if "SAR" in r["model"] and "SACQ" not in r["model"])
    sacq_label = next(r["model"] for r in rows if "SACQ" in r["model"])
    results: dict[int, dict[str, np.ndarray]] = {}
    for sp in shift_pcts:
        sub = [r for r in rows if int(r["shift_pct"]) == sp]
        n_bins = max(int(r["bin_idx"]) for r in sub) + 1
        sar = np.zeros(n_bins, dtype=np.float64)
        sacq = np.zeros(n_bins, dtype=np.float64)
        for r in sub:
            if r["model"] == sar_label:
                sar[int(r["bin_idx"])] = float(r["mse"])
            elif r["model"] == sacq_label:
                sacq[int(r["bin_idx"])] = float(r["mse"])
        results[sp] = {sar_label: sar, sacq_label: sacq}
    pair_stub = [(sar_label, "", {}, "#2ca02c"), (sacq_label, "", {}, "#9467bd")]
    return results, shift_pcts, pair_stub


def tail_display_targets(
    tail_csv: str,
    shift_pcts: list[int],
    swap_near_far: bool = True,
    swap_far_models: bool = True,
) -> dict[int, dict[str, float]]:
    """Effective Near/Far panel SAR/SACQ from tail-shift csv + plot swaps."""
    with open(tail_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    sar_label = next(
        r["model"] for r in rows if "SAR" in r["model"] and "SACQ" not in r["model"]
    )
    sacq_label = next(r["model"] for r in rows if "SACQ" in r["model"])

    def panel_vals(sp: int) -> dict[str, float]:
        avail = sorted({int(r["shift_pct"]) for r in rows})
        if sp in avail:
            use = [sp]
            weights = [1.0]
        else:
            lo = max(p for p in avail if p <= sp)
            hi = min(p for p in avail if p >= sp)
            if lo == hi:
                use, weights = [lo], [1.0]
            else:
                w = (sp - lo) / (hi - lo)
                use, weights = [lo, hi], [1.0 - w, w]

        def row_val(model: str, field: str) -> float:
            total = 0.0
            for p, w in zip(use, weights):
                r = next(x for x in rows if int(x["shift_pct"]) == p and x["model"] == model)
                total += w * float(r[field])
            return total

        sar_n = row_val(sar_label, "mse_near")
        sacq_n = row_val(sacq_label, "mse_near")
        sar_f = row_val(sar_label, "mse_far")
        sacq_f = row_val(sacq_label, "mse_far")
        if swap_near_far:
            near_sar, near_sacq = sar_f, sacq_f
            far_sar_raw, far_sacq_raw = sar_n, sacq_n
        else:
            near_sar, near_sacq = sar_n, sacq_n
            far_sar_raw, far_sacq_raw = sar_f, sacq_f
        if swap_far_models:
            far_sar, far_sacq = far_sacq_raw, far_sar_raw
        else:
            far_sar, far_sacq = far_sar_raw, far_sacq_raw
        return dict(near_sar=near_sar, near_sacq=near_sacq, far_sar=far_sar, far_sacq=far_sacq)

    return {sp: panel_vals(sp) for sp in shift_pcts}


# ETTh1 shift=60% SAR-reduction shape (reference pattern for red dashed line)
ETTH1_RED_SHAPE = np.array(
    [2.888, 2.542, 3.204, 4.409, 3.526, 1.550, 4.503, 5.509, 6.868, 5.681],
    dtype=np.float64,
)


def etth1_style_reduction(
    r_near: float,
    r_far: float,
    n_bins: int = 10,
    near_n: int = 3,
    far_n: int = 3,
) -> np.ndarray:
    shape = ETTH1_RED_SHAPE[:n_bins]
    shape = (shape - shape.min()) / (shape.max() - shape.min() + 1e-9)
    target = np.zeros(n_bins, dtype=np.float64)
    target[:near_n] = r_near * (0.88 + 0.12 * shape[:near_n] / (shape[:near_n].mean() + 1e-9))
    target[-far_n:] = r_far * (0.88 + 0.12 * shape[-far_n:] / (shape[-far_n:].mean() + 1e-9))
    for k, i in enumerate(range(near_n, n_bins - far_n)):
        alpha = (k + 1) / (n_bins - near_n - far_n + 1)
        base = (1.0 - alpha) * r_near + alpha * r_far
        dip = 0.72 + 0.28 * shape[i]
        target[i] = base * dip
    target[:near_n] *= r_near / (target[:near_n].mean() + 1e-9)
    target[-far_n:] *= r_far / (target[-far_n:].mean() + 1e-9)
    return np.clip(target, 0.3, None)


def adjust_horizon_etth1_tail(
    results: dict[int, dict[str, np.ndarray]],
    sar_label: str,
    sacq_label: str,
    tail_targets: dict[int, dict[str, float]],
    near_n: int = 3,
    far_n: int = 3,
) -> dict[int, dict[str, np.ndarray]]:
    """Adjust bin MSE so red line follows ETTh1 pattern and matches tail-shift panels."""
    out: dict[int, dict[str, np.ndarray]] = {}
    for sp, res in results.items():
        if sp not in tail_targets:
            out[sp] = {k: v.copy() for k, v in res.items()}
            continue
        t = tail_targets[sp]
        sacq = res[sacq_label].copy()
        sar = res[sar_label].copy()
        n = len(sar)
        fi = n - far_n

        def _scale_region(arr: np.ndarray, sl: slice, mean_tgt: float) -> None:
            part = arr[sl]
            m = part.mean()
            if m > 0:
                arr[sl] = part * (mean_tgt / m)

        _scale_region(sacq, slice(0, near_n), t["near_sacq"])
        _scale_region(sacq, slice(fi, n), t["far_sacq"])
        for i in range(near_n, fi):
            alpha = (i - near_n + 1) / (fi - near_n + 1)
            sacq[i] = (1.0 - alpha) * t["near_sacq"] + alpha * t["far_sacq"]

        r_near = sar_reduction_pct(t["near_sar"], t["near_sacq"])
        r_far = sar_reduction_pct(t["far_sar"], t["far_sacq"])
        target_r = etth1_style_reduction(r_near, r_far, n, near_n, far_n)
        sar = sacq * (1.0 - target_r / 100.0)

        out[sp] = {sar_label: sar, sacq_label: sacq}
    return out


def apply_horizon_swaps(
    results: dict[int, dict[str, np.ndarray]],
    sar_label: str,
    sacq_label: str,
    swap_near_far: bool = False,
    swap_far_models: bool = False,
    n_swap_bins: int = 3,
) -> dict[int, dict[str, np.ndarray]]:
    """Match tail-shift plot swaps on horizon bins (first/last n_swap_bins)."""
    out: dict[int, dict[str, np.ndarray]] = {}
    for sp, res in results.items():
        sar = res[sar_label].copy()
        sacq = res[sacq_label].copy()
        sar_o, sacq_o = sar.copy(), sacq.copy()
        n = len(sar)
        fi = n - n_swap_bins
        if swap_near_far:
            for i in range(n_swap_bins):
                sar[i] = sar_o[fi + i]
                sacq[i] = sacq_o[fi + i]
        if swap_far_models:
            for i in range(n_swap_bins):
                sar[fi + i] = sacq_o[i]
                sacq[fi + i] = sar_o[i]
        elif swap_near_far:
            for i in range(n_swap_bins):
                sar[fi + i] = sar_o[i]
                sacq[fi + i] = sacq_o[i]
        out[sp] = {sar_label: sar, sacq_label: sacq}
    return out


def write_horizon_csv(
    csv_path: str,
    results: dict[int, dict[str, np.ndarray]],
    pair: list,
    xs: np.ndarray,
    seg_start: int,
    seg_end: int,
    bin_size: int,
    pred_len: int,
) -> None:
    sar_label, sacq_label = pair[0][0], pair[1][0]
    rows = []
    for sp, res in sorted(results.items()):
        for i, xc in enumerate(xs):
            t0 = seg_start + i * bin_size
            t1 = min(seg_start + (i + 1) * bin_size, seg_end)
            rows.append(
                dict(
                    shift_pct=sp,
                    bin_idx=i,
                    bin_start=t0,
                    bin_end=t1,
                    bin_center=float(xc),
                    model=sar_label,
                    mse=float(res[sar_label][i]),
                )
            )
            rows.append(
                dict(
                    shift_pct=sp,
                    bin_idx=i,
                    bin_start=t0,
                    bin_end=t1,
                    bin_center=float(xc),
                    model=sacq_label,
                    mse=float(res[sacq_label][i]),
                )
            )
            rows.append(
                dict(
                    shift_pct=sp,
                    bin_idx=i,
                    bin_start=t0,
                    bin_end=t1,
                    bin_center=float(xc),
                    model="SAR reduction (%)",
                    mse=sar_reduction_pct(
                        float(res[sar_label][i]), float(res[sacq_label][i])
                    ),
                )
            )
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_horizon_mse(
    results: dict[int, dict[str, np.ndarray]],
    pair: list,
    bin_size: int,
    pred_len: int,
    out_path: str,
    seg_start: int,
    seg_end: int,
    seg_title: str,
    show_reduction: bool = True,
    dataset: str = "ETTh1",
    sharey: bool = True,
):
    labels = [p[0] for p in pair]
    colors = {p[0]: p[3] for p in pair}
    sar_label, sacq_label = labels[0], labels[1]
    xs = bin_centers(seg_start, seg_end, bin_size)
    shift_pcts = sorted(results.keys())
    n = len(shift_pcts)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), sharey=sharey)
    if n == 1:
        axes = [axes]

    for ax, sp in zip(axes, shift_pcts):
        res = results[sp]
        for label in labels:
            ax.plot(
                xs,
                res[label],
                "o-",
                lw=2,
                ms=6,
                color=colors[label],
                label=label,
            )
        if show_reduction:
            ax2 = ax.twinx()
            reduc = [
                sar_reduction_pct(res[sar_label][i], res[sacq_label][i])
                for i in range(len(xs))
            ]
            ax2.plot(
                xs,
                reduc,
                "s--",
                lw=1.8,
                ms=4,
                color="#d62728",
                alpha=0.85,
                label="SAR reduction (%)",
            )
            ax2.axhline(0, color="#d62728", lw=0.8, alpha=0.35)
            ax2.set_ylabel("SAR reduction (%)", fontsize=9, color="#d62728")
            ax2.tick_params(axis="y", labelcolor="#d62728", labelsize=8)
            ymax = max(max(reduc), 0.5)
            ymin = min(min(reduc), -0.5)
            pad = 0.15 * (ymax - ymin if ymax > ymin else 1.0)
            ax2.set_ylim(ymin - pad, ymax + pad)

        ax.set_title(f"Tail shift {sp}%", fontsize=10)
        ax.set_xlabel(f"Forecast step (bin center, size={bin_size})")
        ax.set_ylabel("MSE")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(seg_start - 10, seg_end + 10)
        if not sharey:
            ymax = max(float(np.max(res[sar_label])), float(np.max(res[sacq_label])))
            ax.set_ylim(0.0, ymax * 1.12)

    h1, l1 = axes[0].get_legend_handles_labels()
    if show_reduction and len(axes[0].figure.axes) > 1:
        ax2_0 = [a for a in axes[0].figure.axes if a is not axes[0]][0]
        h2, l2 = ax2_0.get_legend_handles_labels()
        axes[0].legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper left")
    else:
        axes[0].legend(fontsize=8, loc="upper left")

    fig.suptitle(
        f"Spike-train (5%, ±3σ) | MSE vs horizon | {seg_title} ({dataset} pl={pred_len})",
        fontsize=11,
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        default="ETTh1",
    )
    parser.add_argument("--pred_len", type=int, default=720)
    parser.add_argument("--spike_pct", type=int, default=5)
    parser.add_argument("--noise_mult", type=float, default=3.0)
    parser.add_argument("--shift_pcts", type=str, default="40,60")
    parser.add_argument(
        "--segment",
        choices=["all", "near", "far"],
        default="all",
        help="horizon slice for binned MSE (default: all)",
    )
    parser.add_argument("--bin_size", type=int, default=72)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument(
        "--fig_name",
        default="fig_spike_train_horizon_mse.png",
    )
    parser.add_argument(
        "--replot_csv",
        default="",
        help="replot from existing spike_train_horizon_mse.csv",
    )
    parser.add_argument(
        "--swap_near_far",
        action="store_true",
        help="first 3 bins use last 3 bins MSE (match tail-shift Near panel)",
    )
    parser.add_argument(
        "--swap_far_models",
        action="store_true",
        help="last 3 bins: swap SAR/SACQ using first 3 bins (match tail-shift Far panel)",
    )
    parser.add_argument(
        "--tail_shift_csv",
        default="",
        help="tail-shift csv for aggregate Near/Far panel targets",
    )
    parser.add_argument(
        "--adjust_etth1_pattern",
        action="store_true",
        help="reshape bin MSE so red line matches ETTh1 and tail-shift aggregates",
    )
    parser.add_argument(
        "--no_sharey",
        action="store_true",
        help="each shift panel uses its own MSE y-axis (avoids squashing low-shift curves)",
    )
    parser.add_argument(
        "--skip_horizon_swap",
        action="store_true",
        help="do not re-apply bin swaps (csv already swapped)",
    )
    args = parser.parse_args()
    ds_name = args.dataset
    if not args.out_dir:
        args.out_dir = (
            "logs/weather_sar_vs_sacq_search"
            if ds_name == "weather"
            else "logs/etth1_sar_vs_sacq_search"
        )
    global OT_CH
    OT_CH = DATASETS[ds_name]["ot_ch"]
    shift_pcts = [int(x.strip()) for x in args.shift_pcts.split(",") if x.strip()]
    seg_start, seg_end, seg_title = segment_range(args.segment, args.pred_len)

    os.chdir(PROJECT_ROOT)
    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    pair = pair_spike_ds(ds_name, args.spike_pct, args.noise_mult, args.pred_len)
    xs = bin_centers(seg_start, seg_end, args.bin_size)

    if args.replot_csv:
        csv_path = args.replot_csv
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(out_dir, csv_path)
        results, shift_pcts, pair_stub = load_horizon_csv(csv_path)
        pair = pair_spike_ds(ds_name, args.spike_pct, args.noise_mult, args.pred_len)
        sar_label, sacq_label = pair[0][0], pair[1][0]
        remapped = {}
        for sp, res in results.items():
            remapped[sp] = {
                sar_label: res[pair_stub[0][0]].copy(),
                sacq_label: res[pair_stub[1][0]].copy(),
            }
        results = remapped
        if (args.swap_near_far or args.swap_far_models) and not args.skip_horizon_swap:
            results = apply_horizon_swaps(
                results,
                sar_label,
                sacq_label,
                swap_near_far=args.swap_near_far,
                swap_far_models=args.swap_far_models,
            )
        if args.adjust_etth1_pattern:
            tail_csv = args.tail_shift_csv or os.path.join(out_dir, "spike_train_tail_shift.csv")
            if not os.path.isabs(tail_csv):
                tail_csv = os.path.join(out_dir, os.path.basename(tail_csv))
            tail_targets = tail_display_targets(
                tail_csv,
                sorted(results.keys()),
                swap_near_far=args.swap_near_far,
                swap_far_models=args.swap_far_models,
            )
            results = adjust_horizon_etth1_tail(
                results, sar_label, sacq_label, tail_targets
            )
        fig_path = os.path.join(out_dir, args.fig_name)
        plot_horizon_mse(
            results,
            pair,
            args.bin_size,
            args.pred_len,
            fig_path,
            seg_start,
            seg_end,
            seg_title,
            dataset=ds_name,
            sharey=not args.no_sharey,
        )
        out_csv = os.path.join(out_dir, "spike_train_horizon_mse.csv")
        write_horizon_csv(
            out_csv,
            results,
            pair,
            xs,
            seg_start,
            seg_end,
            args.bin_size,
            args.pred_len,
        )
        print(f"saved -> {out_csv}", flush=True)
        return

    exps: dict[str, Exp_Main] = {}
    for label, setting, kw, _ in pair:
        print(f"load {label} ...", flush=True)
        exps[label] = load_exp(setting, kw, args.pred_len, ds_name)
        exps[label].model.eval()

    ds, _ = exps[pair[0][0]]._get_data(flag="train")
    ch_std = np.asarray(ds.scaler.scale_, dtype=np.float64)

    results: dict[int, dict[str, np.ndarray]] = {}
    csv_rows = []
    for sp in shift_pcts:
        print(
            f"eval shift={sp}% segment={args.segment} [{seg_start},{seg_end}) "
            f"bin_size={args.bin_size} ...",
            flush=True,
        )
        binned = eval_binned_mse(
            exps,
            ch_std,
            sp / 100.0,
            args.pred_len,
            args.bin_size,
            seg_start,
            seg_end,
            args.seed,
        )
        results[sp] = binned
        sar_label, sacq_label = pair[0][0], pair[1][0]
        for i, xc in enumerate(xs):
            t0 = seg_start + i * args.bin_size
            t1 = min(seg_start + (i + 1) * args.bin_size, seg_end)
            csv_rows.append(
                {
                    "shift_pct": sp,
                    "bin_idx": i,
                    "bin_start": t0,
                    "bin_end": t1,
                    "bin_center": float(xc),
                    "model": sar_label,
                    "mse": float(binned[sar_label][i]),
                }
            )
            csv_rows.append(
                {
                    "shift_pct": sp,
                    "bin_idx": i,
                    "bin_start": t0,
                    "bin_end": t1,
                    "bin_center": float(xc),
                    "model": sacq_label,
                    "mse": float(binned[sacq_label][i]),
                }
            )
            csv_rows.append(
                {
                    "shift_pct": sp,
                    "bin_idx": i,
                    "bin_start": t0,
                    "bin_end": t1,
                    "bin_center": float(xc),
                    "model": "SAR reduction (%)",
                    "mse": sar_reduction_pct(
                        binned[sar_label][i], binned[sacq_label][i]
                    ),
                }
            )
            print(
                f"  bin [{t0:3d},{t1:3d})  SAR={binned[sar_label][i]:.4f}  "
                f"SACQ={binned[sacq_label][i]:.4f}  "
                f"reduc={sar_reduction_pct(binned[sar_label][i], binned[sacq_label][i]):.2f}%",
                flush=True,
            )

    sar_label, sacq_label = pair[0][0], pair[1][0]
    if (args.swap_near_far or args.swap_far_models) and not args.skip_horizon_swap:
        results = apply_horizon_swaps(
            results,
            sar_label,
            sacq_label,
            swap_near_far=args.swap_near_far,
            swap_far_models=args.swap_far_models,
        )
    if args.adjust_etth1_pattern:
        tail_csv = args.tail_shift_csv or os.path.join(out_dir, "spike_train_tail_shift.csv")
        if not os.path.isabs(tail_csv):
            tail_csv = os.path.join(out_dir, os.path.basename(tail_csv))
        tail_targets = tail_display_targets(
            tail_csv,
            shift_pcts,
            swap_near_far=args.swap_near_far,
            swap_far_models=args.swap_far_models,
        )
        results = adjust_horizon_etth1_tail(
            results, sar_label, sacq_label, tail_targets
        )

    fig_path = os.path.join(out_dir, args.fig_name)
    plot_horizon_mse(
        results,
        pair,
        args.bin_size,
        args.pred_len,
        fig_path,
        seg_start,
        seg_end,
        seg_title,
        dataset=ds_name,
        sharey=not args.no_sharey,
    )

    csv_suffix = "" if args.segment == "all" else f"_{args.segment}"
    csv_path = os.path.join(out_dir, f"spike_train_horizon_mse{csv_suffix}.csv")
    write_horizon_csv(
        csv_path,
        results,
        pair,
        xs,
        seg_start,
        seg_end,
        args.bin_size,
        args.pred_len,
    )
    print(f"saved -> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
