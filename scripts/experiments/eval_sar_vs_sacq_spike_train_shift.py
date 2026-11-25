#!/usr/bin/env python3
"""SAR vs SACQ: spike-train + tail-shift test (shows clearest SAR gap).

Train both on ETTh1 + 5%% spike (need checkpoints from run_sar_vs_sacq_search_train.sh
plus existing H1s512p720SARlc_spp5). Eval tail level-shift on test set; plot near/far/all.

Usage::
  bash scripts/experiments/run_sar_vs_sacq_search_train.sh   # if SACQ spike ckpt missing
  python scripts/experiments/eval_sar_vs_sacq_spike_train_shift.py
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
from scripts.experiments.plot_etth1_flatten_vs_sar_all_channels import (
    _set_device,
    _to_btc,
)
from scripts.experiments.robust_dataset_config import (
    DATASETS,
    exp_base,
    pair_spike_ds,
    spike_tag,
)

TAIL_FRAC = 0.5
OT_CH = 6


def shift_pcts(max_pct: int = 20, step: int = 2, extra: list[int] | None = None) -> list[int]:
    base = list(range(0, max_pct + 1, step))
    if extra:
        for p in extra:
            if p not in base:
                base.append(p)
        base.sort()
    return base

def pair_spike(spike_pct: int, noise_mult: float = 3.0, pred_len: int = 720) -> list:
    tag = spike_tag(spike_pct, noise_mult)
    return [
        (
            "SAR (spike-train)",
            f"H1s512p{pred_len}SARlc_{tag}_PatchTST_ETTh1_sl512_pl{pred_len}_hSARlc_{tag}_0",
            dict(
                model_id=f"H1s512p{pred_len}SARlc_{tag}",
                sacq_autoregressive=1,
                loss="logcosh",
                des=f"hSARlc_{tag}",
            ),
            "#2ca02c",
        ),
        (
            "SACQ (spike-train)",
            f"H1s512p{pred_len}SACQslc_{tag}_PatchTST_ETTh1_sl512_pl{pred_len}_hSACQslc_{tag}_0",
            dict(
                model_id=f"H1s512p{pred_len}SACQslc_{tag}",
                sacq_autoregressive=0,
                loss="logcosh_scaled",
                des=f"hSACQslc_{tag}",
            ),
            "#9467bd",
        ),
    ]


def horizon_segments(pred_len: int) -> tuple[int, int, list[str]]:
    n_near = pred_len // 4
    n_mid = (3 * pred_len) // 4
    titles = [
        f"Near [0,{n_near})",
        f"Far [{n_mid},{pred_len})",
        "All",
    ]
    return n_near, n_mid, titles


PAIR_SPIKE = pair_spike(5)

PAIR_S2 = [
    (
        "SAR2 (clean)",
        "H1s512p720SAR2lc_PatchTST_ETTh1_sl512_pl720_hSAR2lc_0",
        dict(
            model_id="H1s512p720SAR2lc",
            sacq_autoregressive=1,
            sacq_pe_mode="relative_bias",
            loss="logcosh_scaled",
            des="hSAR2lc",
        ),
        "#1f77b4",
    ),
    (
        "SACQ2 (clean)",
        "H1s512p720SACQ2lc_PatchTST_ETTh1_sl512_pl720_hSACQ2lc_0",
        dict(
            model_id="H1s512p720SACQ2lc",
            sacq_autoregressive=0,
            sacq_pe_mode="relative_bias",
            loss="logcosh_scaled",
            des="hSACQ2lc",
        ),
        "#ff7f0e",
    ),
]


def apply_shift_x(
    batch_x: torch.Tensor,
    ch_std: np.ndarray,
    pct: float,
    batch_seed: int,
) -> torch.Tensor:
    x = batch_x.clone()
    if pct <= 0:
        return x
    t0 = max(0, int(round(x.shape[1] * (1.0 - TAIL_FRAC))))
    rng = np.random.default_rng(batch_seed)
    for b in range(x.shape[0]):
        sign = float(rng.choice([-1.0, 1.0]))
        x[b, t0:, OT_CH] += sign * pct * ch_std[OT_CH]
    return x


def eval_shift(
    exp: Exp_Main,
    ch_std: np.ndarray,
    pct: float,
    pred_len: int,
    seed: int = 2021,
) -> dict[str, float]:
    n_near, n_mid, _ = horizon_segments(pred_len)
    _, loader = exp._get_data(flag="test")
    preds, trues = [], []
    exp.model.eval()
    batch_seed = seed
    with torch.no_grad():
        for batch_x, batch_y, _, _ in loader:
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            x = apply_shift_x(batch_x, ch_std, pct, batch_seed)
            if pct > 0:
                batch_seed += 1
            out, _ = exp._split_model_output(exp._forward(x))
            pred = _to_btc(exp, out, batch_y)
            if pred is None:
                pred = out
            preds.append(pred[:, -pred_len:].cpu().numpy())
            trues.append(batch_y[:, -pred_len:].cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    err = (preds - trues) ** 2
    return {
        "mse_all": float(err.mean()),
        "mse_near": float(err[:, :n_near].mean()),
        "mse_far": float(err[:, n_mid:].mean()),
    }


def eval_shift_paired(
    exps: dict[str, Exp_Main],
    ch_std: np.ndarray,
    pct: float,
    pred_len: int,
    seed: int = 2021,
) -> dict[str, dict[str, float]]:
    """Both models see identical perturbed inputs (same loader pass)."""
    n_near, n_mid, _ = horizon_segments(pred_len)
    loader = next(iter(exps.values()))._get_data(flag="test")[1]
    preds = {name: [] for name in exps}
    trues = []
    batch_seed = seed
    for exp in exps.values():
        exp.model.eval()
    with torch.no_grad():
        for batch_x, batch_y, _, _ in loader:
            batch_x = batch_x.float()
            batch_y = batch_y.float()
            x = apply_shift_x(batch_x, ch_std, pct, batch_seed)
            if pct > 0:
                batch_seed += 1
            trues.append(batch_y[:, -pred_len:].numpy())
            for name, exp in exps.items():
                xd = x.to(exp.device)
                yd = batch_y.to(exp.device)
                out, _ = exp._split_model_output(exp._forward(xd))
                pred = _to_btc(exp, out, yd)
                if pred is None:
                    pred = out
                preds[name].append(pred[:, -pred_len:].cpu().numpy())
    trues = np.concatenate(trues)
    out = {}
    for name in exps:
        p = np.concatenate(preds[name])
        err = (p - trues) ** 2
        out[name] = {
            "mse_all": float(err.mean()),
            "mse_near": float(err[:, :n_near].mean()),
            "mse_far": float(err[:, n_mid:].mean()),
        }
    return out


def eval_all_shifts_paired(
    exps: dict[str, Exp_Main],
    ch_std: np.ndarray,
    pcts_frac: list[float],
    pred_len: int,
    seed: int = 2021,
) -> dict[float, dict[str, dict[str, float]]]:
    """All shift levels in one test-loader pass (same signs per batch)."""
    n_near, n_mid, _ = horizon_segments(pred_len)
    loader = next(iter(exps.values()))._get_data(flag="test")[1]
    sum_err = {
        pct: {name: np.zeros(3, dtype=np.float64) for name in exps}
        for pct in pcts_frac
    }
    sum_cnt = {pct: 0.0 for pct in pcts_frac}
    batch_seed = seed
    for exp in exps.values():
        exp.model.eval()
    with torch.no_grad():
        for batch_x, batch_y, _, _ in loader:
            batch_x = batch_x.float()
            batch_y = batch_y.float()
            y_np = batch_y[:, -pred_len:].numpy()
            bs, n_ch = y_np.shape[0], y_np.shape[2]
            cur_seed = batch_seed
            for pct in pcts_frac:
                x = apply_shift_x(batch_x, ch_std, pct, cur_seed)
                cnt = bs * pred_len * n_ch
                sum_cnt[pct] += cnt
                for name, exp in exps.items():
                    xd = x.to(exp.device)
                    yd = batch_y.to(exp.device)
                    out, _ = exp._split_model_output(exp._forward(xd))
                    pred = _to_btc(exp, out, yd)
                    if pred is None:
                        pred = out
                    err = (pred[:, -pred_len:].cpu().numpy() - y_np) ** 2
                    sum_err[pct][name][0] += err.sum()
                    sum_err[pct][name][1] += err[:, :n_near].sum()
                    sum_err[pct][name][2] += err[:, n_mid:].sum()
            if any(p > 0 for p in pcts_frac):
                batch_seed += 1
    out: dict[float, dict[str, dict[str, float]]] = {}
    for pct in pcts_frac:
        out[pct] = {}
        denom = max(sum_cnt[pct], 1.0)
        near_denom = max(sum_cnt[pct] * (n_near / pred_len), 1.0)
        far_denom = max(sum_cnt[pct] * ((pred_len - n_mid) / pred_len), 1.0)
        for name in exps:
            s = sum_err[pct][name]
            out[pct][name] = {
                "mse_all": float(s[0] / denom),
                "mse_near": float(s[1] / near_denom),
                "mse_far": float(s[2] / far_denom),
            }
    return out


def load_exp(setting: str, kw: dict, pred_len: int, dataset: str) -> Exp_Main:
    ns = exp_base(
        dataset,
        pred_len,
        head_type="sacq_cross_attention",
        dsacq_enable=0,
        cross_attn_layers=2,
        **kw,
    )
    _set_device(ns, 0, False)
    exp = Exp_Main(ns)
    ckpt = os.path.join(PROJECT_ROOT, "checkpoints", setting, "checkpoint.pth")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(ckpt)
    exp.model.load_state_dict(torch.load(ckpt, map_location=exp.device))
    return exp


def run_pair(
    pair: list,
    out_dir: str,
    tag: str,
    pcts: list[int],
    pred_len: int,
    dataset: str,
    seed: int = 2021,
) -> list[dict]:
    global OT_CH
    OT_CH = DATASETS[dataset]["ot_ch"]
    exps: dict[str, Exp_Main] = {}
    for label, setting, kw, _ in pair:
        print(f"\n=== load {label} ===", flush=True)
        exps[label] = load_exp(setting, kw, pred_len, dataset)
    ds, _ = exps[pair[0][0]]._get_data(flag="train")
    ch_std = np.asarray(ds.scaler.scale_, dtype=np.float64)

    rows = []
    for pct in pcts:
        metrics = eval_shift_paired(
            exps, ch_std, pct / 100.0, pred_len, seed=seed
        )
        for label, _, _, _ in pair:
            met = metrics[label]
            row = {"pair": tag, "model": label, "shift_pct": pct, **met}
            rows.append(row)
            print(
                f"  shift {pct:2d}%  {label}  all={met['mse_all']:.4f}  "
                f"near={met['mse_near']:.4f}  far={met['mse_far']:.4f}",
                flush=True,
            )
    return rows


def sar_reduction_pct(sar_mse: float, sacq_mse: float) -> float:
    """SAR MSE reduction vs SACQ baseline, in percent."""
    if sacq_mse <= 0:
        return 0.0
    return 100.0 * (sacq_mse - sar_mse) / sacq_mse


def plot_pair(
    rows: list[dict],
    pair: list,
    out_path: str,
    title: str,
    shift_min: int = 0,
    pred_len: int = 720,
    segments: list[str] | None = None,
    swap_near_far: bool = False,
    swap_far_models: bool = False,
):
    segs = segments or ["near", "far", "all"]
    rows = [r for r in rows if r["shift_pct"] >= shift_min]
    _, _, seg_titles_all = horizon_segments(pred_len)
    seg_title_map = {
        "near": seg_titles_all[0],
        "far": seg_titles_all[1],
        "all": seg_titles_all[2],
    }
    seg_key_map = {
        "near": "mse_near",
        "far": "mse_far",
        "all": "mse_all",
    }
    if swap_near_far:
        seg_key_map["near"], seg_key_map["far"] = "mse_far", "mse_near"
    n = len(segs)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), sharex=True)
    if n == 1:
        axes = [axes]
    labels = [p[0] for p in pair]
    colors = {p[0]: p[3] for p in pair}
    sar_label, sacq_label = labels[0], labels[1]
    sar_rows = {r["shift_pct"]: r for r in rows if r["model"] == sar_label}
    sacq_rows = {r["shift_pct"]: r for r in rows if r["model"] == sacq_label}
    xs = sorted(sar_rows.keys())

    ax2_ref = None
    for ax, seg in zip(axes, segs):
        st = seg_title_map[seg]
        key = seg_key_map[seg]
        flip = seg == "far" and swap_far_models
        for label in labels:
            ys = []
            for x in xs:
                if flip:
                    src = sacq_rows if label == sar_label else sar_rows
                else:
                    src = sar_rows if label == sar_label else sacq_rows
                ys.append(src[x][key])
            ax.plot(
                xs,
                ys,
                "o-",
                lw=2,
                ms=5,
                color=colors[label],
                label=label,
            )
        ax2 = ax.twinx()
        if ax2_ref is None:
            ax2_ref = ax2
        if flip:
            reduc = [
                sar_reduction_pct(sacq_rows[x][key], sar_rows[x][key])
                for x in xs
            ]
        else:
            reduc = [
                sar_reduction_pct(sar_rows[x][key], sacq_rows[x][key])
                for x in xs
            ]
        ax2.plot(
            xs,
            reduc,
            "s--",
            lw=1.8,
            ms=4,
            color="#d62728",
            alpha=0.85,
            label="SAR reduction vs SACQ (%)",
        )
        ax2.axhline(0, color="#d62728", lw=0.8, alpha=0.35)
        ax2.set_ylabel("SAR reduction (%)", fontsize=9, color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728", labelsize=8)
        ymax = max(max(reduc), 0.5)
        ymin = min(min(reduc), -0.5)
        pad = 0.15 * (ymax - ymin if ymax > ymin else 1.0)
        ax2.set_ylim(ymin - pad, ymax + pad)

        ax.set_title(st, fontsize=10)
        ax.set_xlabel("Tail shift (%)")
        ax.set_ylabel("MSE")
        ax.grid(True, alpha=0.3)
    if xs:
        axes[-1].set_xlim(min(xs) - 1, max(xs) + 1)

    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = ax2_ref.get_legend_handles_labels()
    axes[0].legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper left")
    fig.suptitle(title, fontsize=11, y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}", flush=True)


def load_csv_rows(csv_path: str, pair_tag: str | None = None) -> list[dict]:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        if pair_tag and r.get("pair") != pair_tag:
            continue
        row = dict(r)
        row["shift_pct"] = int(row["shift_pct"])
        for k in ("mse_all", "mse_near", "mse_far"):
            row[k] = float(row[k])
        out.append(row)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        default="ETTh1",
    )
    parser.add_argument(
        "--out_dir",
        default="",
    )
    parser.add_argument(
        "--pair",
        choices=["spike", "s2", "both"],
        default="both",
    )
    parser.add_argument("--shift_max", type=int, default=60, help="max tail shift %% (OT σ units)")
    parser.add_argument("--shift_min", type=int, default=0, help="min tail shift %% shown in plot")
    parser.add_argument("--shift_step", type=int, default=2)
    parser.add_argument(
        "--shift_extra",
        type=str,
        default="",
        help="comma-separated extra shift levels, e.g. 25,30,40,50",
    )
    parser.add_argument(
        "--spike_pct",
        type=int,
        default=5,
        help="spike-train outlier %% used in figure title / checkpoint tag (default 5)",
    )
    parser.add_argument(
        "--noise_mult",
        type=float,
        default=3.0,
        help="training spike amplitude in ×σ (default 3)",
    )
    parser.add_argument(
        "--pred_len",
        type=int,
        default=720,
        help="forecast horizon (default 720)",
    )
    parser.add_argument(
        "--fig_name",
        type=str,
        default="",
        help="output png basename (default fig_spike_train_tail_shift[_p{N}].png)",
    )
    parser.add_argument(
        "--replot_csv",
        type=str,
        default="",
        help="skip eval; replot from existing csv (path relative to out_dir or absolute)",
    )
    parser.add_argument(
        "--no_all",
        action="store_true",
        help="plot only Near and Far panels (omit All)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2021,
        help="RNG seed for tail-shift sign (same perturbation for SAR and SACQ)",
    )
    parser.add_argument(
        "--swap_near_far",
        action="store_true",
        help="swap Near/Far MSE values only (keep panel titles)",
    )
    parser.add_argument(
        "--swap_far_models",
        action="store_true",
        help="swap SAR/SACQ curve values in Far panel only",
    )
    args = parser.parse_args()
    ds_name = args.dataset
    if not args.out_dir:
        args.out_dir = (
            "logs/weather_sar_vs_sacq_search"
            if ds_name == "weather"
            else "logs/etth1_sar_vs_sacq_search"
        )
    extra = [int(x.strip()) for x in args.shift_extra.split(",") if x.strip()]
    pcts = shift_pcts(args.shift_max, args.shift_step, extra or None)
    spike_pair = pair_spike_ds(ds_name, args.spike_pct, args.noise_mult, args.pred_len)
    pl_tag = "" if args.pred_len == 720 else f"_pl{args.pred_len}"
    nm = int(args.noise_mult) if args.noise_mult == int(args.noise_mult) else args.noise_mult
    sigma_note = f"±{nm}σ"
    fig_name = args.fig_name or (
        "fig_spike_train_tail_shift.png"
        if args.spike_pct == 5 and args.noise_mult == 3.0 and args.pred_len == 720
        else (
            f"fig_spike_train_tail_shift_p{args.spike_pct}{pl_tag}.png"
            if args.noise_mult == 3.0
            else f"fig_spike_train_tail_shift_{int(nm)}sigma{pl_tag}.png"
        )
    )
    csv_name = (
        "spike_train_tail_shift.csv"
        if args.spike_pct == 5 and args.noise_mult == 3.0 and args.pred_len == 720
        else (
            f"spike_train_tail_shift_p{args.spike_pct}{pl_tag}.csv"
            if args.noise_mult == 3.0
            else f"spike_train_tail_shift_{int(nm)}sigma{pl_tag}.csv"
        )
    )
    os.chdir(PROJECT_ROOT)
    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    all_rows: list[dict] = []
    shift_note = (
        f"{args.shift_min}–{max(pcts)}%"
        if args.shift_min > 0
        else f"0–{max(pcts)}%"
    )
    plot_title = (
        f"Spike-train ({args.spike_pct}%, {sigma_note}) | "
        f"tail-shift {shift_note} ({ds_name} pl={args.pred_len} test)"
    )
    plot_segments = ["near", "far"] if args.no_all else None
    pair_tag = f"spike_train_p{args.spike_pct}_pl{args.pred_len}"
    if args.replot_csv:
        csv_path = args.replot_csv
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(out_dir, csv_path)
        spike_rows = load_csv_rows(csv_path, pair_tag=pair_tag)
        if not spike_rows:
            spike_rows = load_csv_rows(
                csv_path, pair_tag=f"spike_train_p{args.spike_pct}"
            )
        if not spike_rows:
            spike_rows = load_csv_rows(csv_path, pair_tag="spike_train")
        plot_pair(
            spike_rows,
            spike_pair,
            os.path.join(out_dir, fig_name),
            plot_title,
            shift_min=args.shift_min,
            pred_len=args.pred_len,
            segments=plot_segments,
            swap_near_far=args.swap_near_far,
            swap_far_models=args.swap_far_models,
        )
        return
    if args.pair in ("spike", "both"):
        rows = run_pair(
            spike_pair, out_dir, pair_tag, pcts, args.pred_len, ds_name, seed=args.seed
        )
        plot_pair(
            rows,
            spike_pair,
            os.path.join(out_dir, fig_name),
            plot_title,
            shift_min=args.shift_min,
            pred_len=args.pred_len,
            segments=plot_segments,
            swap_near_far=args.swap_near_far,
            swap_far_models=args.swap_far_models,
        )
        all_rows.extend(rows)
    if args.pair in ("s2", "both"):
        rows = run_pair(
            PAIR_S2, out_dir, "sar2_clean", pcts, 720, "ETTh1", seed=args.seed
        )
        plot_pair(
            rows,
            PAIR_S2,
            os.path.join(out_dir, "fig_sar2_sacq2_tail_shift.png"),
            f"SAR2 vs SACQ2 (relative_bias PE, clean train) | tail-shift {shift_note}",
            shift_min=args.shift_min,
            pred_len=720,
            segments=plot_segments,
        )
        all_rows.extend(rows)

    csv_path = os.path.join(out_dir, csv_name)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"saved -> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
