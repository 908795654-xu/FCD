"""Train-set forward eval with input corruptions (4 sl512 pl720 models).

Corruptions (inference-time on batch_x history, metrics vs clean future):
  - mask: zero random timesteps (15% / 30% / 45%)
  - spike on OT: +3σ / +5σ impulses at random times
  - shift on OT: level shift on trailing history (RevIN-visible), magnitude p% × train σ_OT
  - spike: +kσ on OT uses k × train scaler.scale_ (raw σ units in norm space)
  - shift: trailing-segment level shift in normalized σ units (pct, not × scale_)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from copy import deepcopy

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
    _etth1_base,
    _set_device,
    _to_btc,
)
from utils.metrics import metric

SEQ_LEN = 512
OT_CH = 6  # ETTh1 M: OT is last column


def resolve_models(pred_len: int) -> list[dict]:
    flat_id = f"ETTh1_512_{pred_len}"
    return [
        {
            "tag": "flat_mse",
            "label": "Flatten+MSE",
            "setting": f"{flat_id}_PatchTST_ETTh1_sl{SEQ_LEN}_pl{pred_len}_Exp_0",
            "kwargs": dict(
                model_id=flat_id,
                head_type="flatten",
                loss="mse",
                des="Exp",
                sacq_autoregressive=0,
            ),
        },
        {
            "tag": "flat_slc",
            "label": "Flatten+logcosh",
            "setting": f"{flat_id}_PatchTST_ETTh1_sl{SEQ_LEN}_pl{pred_len}_hFlatslc_0",
            "kwargs": dict(
                model_id=flat_id,
                head_type="flatten",
                loss="logcosh_scaled",
                des="hFlatslc",
                sacq_autoregressive=0,
            ),
        },
        {
            "tag": "sar_mse",
            "label": "SAR+MSE",
            "setting": f"H1s{SEQ_LEN}p{pred_len}SAR_PatchTST_ETTh1_sl{SEQ_LEN}_pl{pred_len}_hSAR_0",
            "kwargs": dict(
                model_id=f"H1s{SEQ_LEN}p{pred_len}SAR",
                head_type="sacq_cross_attention",
                loss="mse",
                des="hSAR",
                sacq_autoregressive=1,
            ),
        },
        {
            "tag": "sar_slc",
            "label": "SAR+logcosh",
            "setting": f"H1s{SEQ_LEN}p{pred_len}SARslc_PatchTST_ETTh1_sl{SEQ_LEN}_pl{pred_len}_hSARslc_0",
            "kwargs": dict(
                model_id=f"H1s{SEQ_LEN}p{pred_len}SARslc",
                head_type="sacq_cross_attention",
                loss="logcosh_scaled",
                des="hSARslc",
                sacq_autoregressive=1,
            ),
        },
        {
            "tag": "sacq_slc",
            "label": "SACQ+logcosh",
            "setting": f"H1s{SEQ_LEN}p{pred_len}SACQslc_PatchTST_ETTh1_sl{SEQ_LEN}_pl{pred_len}_hSACQslc_0",
            "kwargs": dict(
                model_id=f"H1s{SEQ_LEN}p{pred_len}SACQslc",
                head_type="sacq_cross_attention",
                loss="logcosh_scaled",
                des="hSACQslc",
                sacq_autoregressive=0,
            ),
        },
    ]

MODEL_COLORS = {
    "flat_mse": "#ff7f0e",
    "flat_slc": "#d62728",
    "sar_mse": "#1f77b4",
    "sar_slc": "#2ca02c",
    "sacq_slc": "#9467bd",
}

# Spike MSE figure: 4-way only (SACQ+logcosh omitted — overlaps SAR+logcosh).
SPIKE_PLOT_TAGS = ["flat_mse", "flat_slc", "sar_mse", "sar_slc"]


def eval_models(pred_len: int) -> list[dict]:
    return [m for m in resolve_models(pred_len) if m["tag"] in SPIKE_PLOT_TAGS]

SPIKE_SIGMA_MULTS = [1, 2, 3, 5, 7]
MASK_RATIOS_PCT = [0, 5, 10, 15, 20, 25, 30]
SHIFT_PCTS = list(range(0, 22, 2))  # 0, 2, 4, ..., 20
# Full-series constant shift is canceled by RevIN norm; use trailing segment only.
SHIFT_TAIL_FRAC = 0.5

CORRUPTIONS = [
    ("clean", {}),
    ("mask_15", {"kind": "mask", "ratio": 0.15}),
    ("mask_30", {"kind": "mask", "ratio": 0.30}),
    ("mask_45", {"kind": "mask", "ratio": 0.45}),
    ("spike_3sigma", {"kind": "spike", "sigma_mult": 3.0}),
    ("spike_5sigma", {"kind": "spike", "sigma_mult": 5.0}),
    ("shift_10_20", {"kind": "shift", "low": 0.10, "high": 0.20, "tail_frac": SHIFT_TAIL_FRAC}),
]


def mask_corruptions() -> list[tuple[str, dict]]:
    items = []
    for p in MASK_RATIOS_PCT:
        key = "clean" if p == 0 else f"mask_{p}"
        if p == 0:
            items.append((key, {}))
        else:
            items.append((key, {"kind": "mask", "ratio": p / 100.0}))
    return items


def shift_corruptions() -> list[tuple[str, dict]]:
    items = []
    for p in SHIFT_PCTS:
        key = "clean" if p == 0 else f"shift_{p}"
        if p == 0:
            items.append((key, {}))
        else:
            items.append(
                (key, {"kind": "shift", "pct": p / 100.0, "tail_frac": SHIFT_TAIL_FRAC})
            )
    return items


def spike_corruptions() -> list[tuple[str, dict]]:
    items = [("clean", {})]
    for m in SPIKE_SIGMA_MULTS:
        key = f"spike_{m}sigma" if m != int(m) else f"spike_{int(m)}sigma"
        items.append((key, {"kind": "spike", "sigma_mult": float(m)}))
    return items


CORRUPTION_GROUPS = {
    "mask": {
        "keys": ["clean", "mask_15", "mask_30", "mask_45"],
        "xticklabels": ["clean", "mask 15%", "mask 30%", "mask 45%"],
        "title": "Random timestep mask (history)",
    },
    "spike": {
        "keys": ["clean", "spike_3sigma", "spike_5sigma"],
        "xticklabels": ["clean", "spike +3σ (OT)", "spike +5σ (OT)"],
        "title": "Random impulse on OT (history)",
    },
    "shift": {
        "keys": ["clean", "shift_10_20"],
        "xticklabels": ["clean", "level shift 10–20% σ_OT"],
        "title": "Random level shift on OT (history)",
    },
}


def make_args(m: dict, pred_len: int):
    kw = dict(
        dsacq_enable=0,
        cross_attn_layers=2,
        train_epochs=50,
        batch_size=128,
        patience=20,
    )
    kw.update(m["kwargs"])
    return _etth1_base(pred_len, seq_len=SEQ_LEN, **kw)


def get_train_stds(exp: Exp_Main) -> np.ndarray:
    """Per-channel train σ in *raw* units (sklearn StandardScaler.scale_)."""
    ds, _ = exp._get_data(flag="train")
    if hasattr(ds, "scaler") and hasattr(ds.scaler, "scale_"):
        return np.asarray(ds.scaler.scale_, dtype=np.float64)
    data = ds.data_x if hasattr(ds, "data_x") else None
    if data is None:
        return np.ones(7, dtype=np.float64)
    return np.std(data, axis=0, ddof=0).clip(min=1e-8)


def corruption_seed(base: int, corruption_name: str) -> int:
    """Stable per-corruption seed (do not use hash(); it varies across runs)."""
    digest = hashlib.md5(corruption_name.encode()).hexdigest()
    return base + int(digest[:8], 16) % 10000


def _shift_time_slice(tlen: int, tail_frac: float) -> slice:
    if tail_frac >= 1.0:
        return slice(None)
    t0 = max(0, int(round(tlen * (1.0 - tail_frac))))
    return slice(t0, None)


def corrupt_batch(
    batch_x: torch.Tensor,
    cfg: dict,
    ch_std: np.ndarray,
    rng: np.random.Generator,
) -> torch.Tensor:
    if not cfg or cfg.get("kind") == "clean":
        return batch_x
    x = batch_x.clone()
    bsz, tlen, nch = x.shape
    kind = cfg["kind"]

    if kind == "mask":
        ratio = cfg["ratio"]
        n_mask = max(1, int(round(tlen * ratio)))
        for b in range(bsz):
            idx = rng.choice(tlen, size=n_mask, replace=False)
            x[b, idx, :] = 0.0

    elif kind == "spike":
        mult = cfg["sigma_mult"]
        ch = OT_CH
        for b in range(bsz):
            t_idx = int(rng.integers(0, tlen))
            sign = float(rng.choice([-1.0, 1.0]))
            x[b, t_idx, ch] += sign * mult * ch_std[ch]

    elif kind == "shift":
        if "pct" in cfg:
            low = high = cfg["pct"]
        else:
            low, high = cfg["low"], cfg["high"]
        tail_frac = float(cfg.get("tail_frac", SHIFT_TAIL_FRAC))
        t_sl = _shift_time_slice(tlen, tail_frac)
        ch = OT_CH
        for b in range(bsz):
            sign = float(rng.choice([-1.0, 1.0]))
            mag = sign * float(rng.uniform(low, high))
            x[b, t_sl, ch] += mag

    return x


@torch.no_grad()
def eval_split(
    exp: Exp_Main, corrupt_cfg: dict, ch_std: np.ndarray, seed: int, split: str = "train"
) -> dict:
    _, loader = exp._get_data(flag=split)
    f_dim = -1 if exp.args.features == "MS" else 0
    pl = exp.args.pred_len
    rng = np.random.default_rng(seed)

    preds, trues, inputx = [], [], []
    exp.model.eval()
    for batch_x, batch_y, _, _ in loader:
        batch_x = batch_x.float().to(exp.device)
        batch_y = batch_y.float().to(exp.device)
        batch_x = corrupt_batch(batch_x, corrupt_cfg, ch_std, rng)

        out, _ = exp._split_model_output(exp._forward(batch_x))
        pred_btc = _to_btc(exp, out, batch_y)
        if pred_btc is None:
            pred_btc = out

        preds.append(pred_btc[:, -pl:, f_dim:].detach().cpu().numpy())
        trues.append(batch_y[:, -pl:, f_dim:].detach().cpu().numpy())
        inputx.append(batch_x.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    inputx = np.concatenate(inputx, axis=0)
    mae, mse, rmse, mape, mspe, rse, corr, smape, mase, ql50, pwmae = metric(
        preds, trues, inputx=inputx
    )
    return {
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
        "rse": float(rse),
        "corr": float(corr),
        "n_samples": int(preds.shape[0]),
    }


def _plot_group_curves(
    rows: list[dict],
    out_dir: str,
    split: str,
    group_name: str,
    group_cfg: dict,
    colors: dict,
    labels: dict,
    tags: list[str],
    pred_len: int,
):
    keys = group_cfg["keys"]
    xlabels = group_cfg["xticklabels"]
    prefix = f"{split}_corruption"

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.2))
    xs = np.arange(len(keys))
    for tag in tags:
        sub = []
        for k in keys:
            sub.append(next(r for r in rows if r["model_tag"] == tag and r["corruption"] == k))
        axes[0].plot(xs, [r["mse"] for r in sub], "o-", lw=2, ms=8, color=colors[tag], label=labels[tag])
        axes[1].plot(xs, [r["mae"] for r in sub], "s--", lw=1.8, ms=7, color=colors[tag], label=labels[tag])

    split_cap = split.capitalize()
    for ax, ylab in zip(axes, [f"{split_cap} MSE", f"{split_cap} MAE"]):
        ax.set_xticks(xs)
        ax.set_xticklabels(xlabels, fontsize=9)
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(
        f"{split_cap} sl=512 pl={pred_len} | {group_cfg['title']}",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{prefix}_{group_name}_mse_mae.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_results(rows: list[dict], out_dir: str, split: str = "train", pred_len: int = 720):
    tags = [m["tag"] for m in eval_models(pred_len)]
    labels = {m["tag"]: m["label"] for m in eval_models(pred_len)}
    corrupts = [c[0] for c in CORRUPTIONS]
    colors = MODEL_COLORS
    prefix = f"{split}_corruption"

    for gname, gcfg in CORRUPTION_GROUPS.items():
        _plot_group_curves(rows, out_dir, split, gname, gcfg, colors, labels, tags, pred_len)

    # relative MSE increase vs clean per model
    base_mse = {}
    for r in rows:
        if r["corruption"] == "clean":
            base_mse[(r["model_tag"],)] = r["mse"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(corrupts))
    width = 0.2
    for i, tag in enumerate(tags):
        ys_mse, ys_mae = [], []
        for c in corrupts:
            row = next(x for x in rows if x["model_tag"] == tag and x["corruption"] == c)
            b = base_mse.get((tag,), row["mse"])
            ys_mse.append((row["mse"] / b - 1) * 100 if b > 0 else 0)
            brow = next(x for x in rows if x["model_tag"] == tag and x["corruption"] == "clean")
            ys_mae.append((row["mae"] / brow["mae"] - 1) * 100 if brow["mae"] > 0 else 0)
        off = (i - 1.5) * width
        axes[0].bar(x + off, ys_mse, width, label=labels[tag], color=colors[tag], alpha=0.9)
        axes[1].bar(x + off, ys_mae, width, label=labels[tag], color=colors[tag], alpha=0.9)

    for ax, ylab in zip(axes, ["MSE", "MAE"]):
        ax.set_xticks(x)
        ax.set_xticklabels(corrupts, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(f"Δ{ylab} vs clean {split} (%)")
        ax.axhline(0, color="k", lw=0.6)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_title(f"{split.capitalize()}-set corruption (sl=512, pl={pred_len})")
    fig.suptitle("Corrupt history @ infer; metric vs clean future labels", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_delta_pct.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # absolute MSE / MAE curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x_ord = {c: i for i, c in enumerate(corrupts)}
    for tag in tags:
        sub = sorted(
            [r for r in rows if r["model_tag"] == tag],
            key=lambda r: x_ord[r["corruption"]],
        )
        xs = [x_ord[r["corruption"]] for r in sub]
        axes[0].plot(xs, [r["mse"] for r in sub], "o-", lw=2, ms=7, color=colors[tag], label=labels[tag])
        axes[1].plot(xs, [r["mae"] for r in sub], "s--", lw=1.8, ms=6, color=colors[tag], label=labels[tag])
    for ax, ylab in zip(axes, ["test MSE", "test MAE"] if split == "test" else ["MSE", "MAE"]):
        ax.set_xticks(range(len(corrupts)))
        ax.set_xticklabels(corrupts, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(f"{split.capitalize()} MSE/MAE vs corruption type", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_mse_mae_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # legacy single MSE line plot
    fig, ax = plt.subplots(figsize=(10, 5))
    for tag in tags:
        sub = sorted([r for r in rows if r["model_tag"] == tag], key=lambda r: x_ord[r["corruption"]])
        ax.plot(
            [r["corruption"] for r in sub],
            [r["mse"] for r in sub],
            "o-",
            lw=2,
            ms=6,
            color=colors[tag],
            label=labels[tag],
        )
    ax.set_ylabel(f"{split} MSE")
    ax.set_xlabel("corruption")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_mse.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_spike_mse_only(rows: list[dict], out_dir: str, split: str, pred_len: int):
    tags = SPIKE_PLOT_TAGS
    colors = MODEL_COLORS
    order = [c[0] for c in spike_corruptions()]
    rows = [r for r in rows if r["corruption"] in order]
    xlabels = ["clean"] + [f"+{m}σ" for m in SPIKE_SIGMA_MULTS]
    xs = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(7.5, 5))  # 3:2, aligned with mask plot
    for tag in tags:
        sub = [next(r for r in rows if r["model_tag"] == tag and r["corruption"] == k) for k in order]
        ax.plot(xs, [r["mse"] for r in sub], "o-", lw=2, ms=8, color=colors[tag])

    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, fontsize=10)
    ax.set_ylabel(f"{split.capitalize()} MSE")
    ax.set_title(rf"ETTh1 $T_f$={pred_len}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{split}_corruption_spike_mse.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}", flush=True)


def plot_mask_mse_only(rows: list[dict], out_dir: str, split: str, pred_len: int):
    present = {r["model_tag"] for r in rows}
    tags = [m["tag"] for m in resolve_models(pred_len) if m["tag"] in present]
    colors = MODEL_COLORS
    order = [c[0] for c in mask_corruptions()]
    rows = [r for r in rows if r["corruption"] in order]
    xlabels = [f"{p}%" for p in MASK_RATIOS_PCT]
    xs = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(7.5, 5))  # 3:2
    for tag in tags:
        sub = [next(r for r in rows if r["model_tag"] == tag and r["corruption"] == k) for k in order]
        ax.plot(xs, [r["mse"] for r in sub], "o-", lw=2, ms=8, color=colors[tag])

    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, fontsize=10)
    ax.set_ylabel(f"{split.capitalize()} MSE")
    ax.set_title(rf"ETTh1 $T_f$={pred_len}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{split}_corruption_mask_mse.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}", flush=True)


def plot_shift_mse_only(rows: list[dict], out_dir: str, split: str):
    tags = SPIKE_PLOT_TAGS
    labels = {m["tag"]: m["label"] for m in eval_models(720)}
    colors = MODEL_COLORS
    order = [c[0] for c in shift_corruptions()]
    rows = [r for r in rows if r["corruption"] in order]
    xlabels = [f"{p}%" for p in SHIFT_PCTS]
    xs = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(10, 4.8))
    for tag in tags:
        sub = [next(r for r in rows if r["model_tag"] == tag and r["corruption"] == k) for k in order]
        ax.plot(xs, [r["mse"] for r in sub], "o-", lw=2, ms=7, color=colors[tag], label=labels[tag])

    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel(f"{split.capitalize()} MSE")
    tail_pct = int(round(SHIFT_TAIL_FRAC * 100))
    ax.set_xlabel(f"OT level shift on last {tail_pct}% of history (× train σ_OT)")
    ax.set_title(f"{split.capitalize()} sl=512 pl=720 | level-shift robustness (4 models)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{split}_corruption_shift_mse.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}", flush=True)


def run_shift_sweep(args, out_dir: str):
    csv_path = os.path.join(out_dir, f"results_{args.split}_shift.csv")
    corruptions = shift_corruptions()
    rows = []
    for m in eval_models(args.pred_len):
        print(f"\n=== {m['tag']} (shift sweep) ===", flush=True)
        ns = make_args(m, args.pred_len)
        _set_device(ns, args.gpu, False)
        exp = Exp_Main(ns)
        ckpt = os.path.join("checkpoints", m["setting"], "checkpoint.pth")
        if not os.path.isfile(ckpt):
            print(f"SKIP missing {ckpt}", flush=True)
            continue
        exp.model.load_state_dict(torch.load(ckpt, map_location=exp.device))
        ch_std = get_train_stds(exp)
        for cname, ccfg in corruptions:
            print(f"  {cname} ...", flush=True)
            met = eval_split(
                exp, ccfg, ch_std, seed=corruption_seed(args.seed, cname), split=args.split
            )
            pct = 0 if cname == "clean" else int(cname.split("_")[1])
            row = {
                "model_tag": m["tag"],
                "model_label": m["label"],
                "setting": m["setting"],
                "corruption": cname,
                "shift_pct": pct,
                "split": args.split,
                **met,
            }
            rows.append(row)
            print(f"    mse={met['mse']:.4f}", flush=True)

    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        plot_shift_mse_only(rows, out_dir, args.split)
        print(f"saved -> {csv_path}", flush=True)


def run_mask_sweep(args, out_dir: str):
    csv_path = os.path.join(out_dir, f"results_{args.split}_mask.csv")
    corruptions = mask_corruptions()
    rows = []
    for m in eval_models(args.pred_len):
        print(f"\n=== {m['tag']} (mask sweep) ===", flush=True)
        ns = make_args(m, args.pred_len)
        _set_device(ns, args.gpu, False)
        exp = Exp_Main(ns)
        ckpt = os.path.join("checkpoints", m["setting"], "checkpoint.pth")
        if not os.path.isfile(ckpt):
            print(f"SKIP missing {ckpt}", flush=True)
            continue
        exp.model.load_state_dict(torch.load(ckpt, map_location=exp.device))
        ch_std = get_train_stds(exp)
        for cname, ccfg in corruptions:
            print(f"  {cname} ...", flush=True)
            met = eval_split(
                exp, ccfg, ch_std, seed=corruption_seed(args.seed, cname), split=args.split
            )
            pct = 0 if cname == "clean" else int(cname.split("_")[1])
            row = {
                "model_tag": m["tag"],
                "model_label": m["label"],
                "setting": m["setting"],
                "corruption": cname,
                "mask_pct": pct,
                "split": args.split,
                **met,
            }
            rows.append(row)
            print(f"    mse={met['mse']:.4f}", flush=True)

    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        plot_mask_mse_only(rows, out_dir, args.split, args.pred_len)
        print(f"saved -> {csv_path}", flush=True)


def run_spike_rerun_sigma(args, out_dir: str, sigma_mult: float = 5.0):
    """Re-evaluate one spike level with a new RNG seed; patch results_{split}_spike.csv."""
    import pandas as pd

    cname = f"spike_{int(sigma_mult)}sigma"
    ccfg = {"kind": "spike", "sigma_mult": float(sigma_mult)}
    csv_path = os.path.join(out_dir, f"results_{args.split}_spike.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"missing {csv_path}; run spike_sweep first")

    rows = pd.read_csv(csv_path).to_dict("records")
    seed = int(args.spike5_seed)
    print(f"rerun {cname} with seed={seed} (plot models: {SPIKE_PLOT_TAGS})", flush=True)

    for m in eval_models(args.pred_len):
        print(f"\n=== {m['tag']} ({cname} rerun) ===", flush=True)
        ns = make_args(m, args.pred_len)
        _set_device(ns, args.gpu, False)
        exp = Exp_Main(ns)
        ckpt = os.path.join("checkpoints", m["setting"], "checkpoint.pth")
        if not os.path.isfile(ckpt):
            print(f"SKIP missing {ckpt}", flush=True)
            continue
        exp.model.load_state_dict(torch.load(ckpt, map_location=exp.device))
        ch_std = get_train_stds(exp)
        met = eval_split(exp, ccfg, ch_std, seed=seed, split=args.split)
        new_row = {
            "model_tag": m["tag"],
            "model_label": m["label"],
            "setting": m["setting"],
            "corruption": cname,
            "sigma_mult": float(sigma_mult),
            "split": args.split,
            **met,
        }
        print(f"    mse={met['mse']:.4f}", flush=True)
        updated = False
        for i, row in enumerate(rows):
            if row["model_tag"] == m["tag"] and row["corruption"] == cname:
                rows[i] = new_row
                updated = True
                break
        if not updated:
            rows.append(new_row)

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    plot_spike_mse_only(rows, out_dir, args.split, args.pred_len)
    print(f"patched -> {csv_path}", flush=True)


def run_spike_sweep(args, out_dir: str):
    csv_path = os.path.join(out_dir, f"results_{args.split}_spike.csv")
    corruptions = spike_corruptions()
    rows = []
    for m in eval_models(args.pred_len):
        print(f"\n=== {m['tag']} (spike sweep) ===", flush=True)
        ns = make_args(m, args.pred_len)
        _set_device(ns, args.gpu, False)
        exp = Exp_Main(ns)
        ckpt = os.path.join("checkpoints", m["setting"], "checkpoint.pth")
        if not os.path.isfile(ckpt):
            print(f"SKIP missing {ckpt}", flush=True)
            continue
        exp.model.load_state_dict(torch.load(ckpt, map_location=exp.device))
        ch_std = get_train_stds(exp)
        for cname, ccfg in corruptions:
            print(f"  {cname} ...", flush=True)
            met = eval_split(
                exp, ccfg, ch_std, seed=corruption_seed(args.seed, cname), split=args.split
            )
            row = {
                "model_tag": m["tag"],
                "model_label": m["label"],
                "setting": m["setting"],
                "corruption": cname,
                "sigma_mult": ccfg.get("sigma_mult", 0),
                "split": args.split,
                **met,
            }
            rows.append(row)
            print(f"    mse={met['mse']:.4f}", flush=True)

    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        plot_spike_mse_only(rows, out_dir, args.split, args.pred_len)
        print(f"saved -> {csv_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        default=None,
        help="default: train_corruption_sl512 (pl720) or train_corruption_sl512_pl{pred_len}",
    )
    parser.add_argument("--pred_len", type=int, default=720)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=[
            "full",
            "spike_sweep",
            "plot_spike",
            "mask_sweep",
            "plot_mask",
            "shift_sweep",
            "plot_shift",
            "rerun_spike5",
        ],
        help="spike_sweep | rerun_spike5 | mask_sweep | shift_sweep | plot_*",
    )
    parser.add_argument(
        "--spike5_seed",
        type=int,
        default=3407,
        help="RNG seed for rerun_spike5 only (default 3407)",
    )
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help="only plot from existing results_{split}.csv",
    )
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    if args.out_dir is None:
        sub = "train_corruption_sl512" if args.pred_len == 720 else f"train_corruption_sl512_pl{args.pred_len}"
        args.out_dir = f"logs/etth1_pl720_seqlen_sweep/{sub}"
    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.mode == "shift_sweep":
        run_shift_sweep(args, out_dir)
        return
    if args.mode == "plot_shift":
        import pandas as pd

        csv_path = os.path.join(out_dir, f"results_{args.split}_shift.csv")
        rows = pd.read_csv(csv_path).to_dict("records")
        plot_shift_mse_only(rows, out_dir, args.split)
        return
    if args.mode == "mask_sweep":
        run_mask_sweep(args, out_dir)
        return
    if args.mode == "plot_mask":
        import pandas as pd

        csv_path = os.path.join(out_dir, f"results_{args.split}_mask.csv")
        rows = pd.read_csv(csv_path).to_dict("records")
        plot_mask_mse_only(rows, out_dir, args.split, args.pred_len)
        return
    if args.mode == "spike_sweep":
        run_spike_sweep(args, out_dir)
        return
    if args.mode == "rerun_spike5":
        run_spike_rerun_sigma(args, out_dir, sigma_mult=5.0)
        return
    if args.mode == "plot_spike":
        import pandas as pd

        csv_path = os.path.join(out_dir, f"results_{args.split}_spike.csv")
        rows = pd.read_csv(csv_path).to_dict("records")
        plot_spike_mse_only(rows, out_dir, args.split, args.pred_len)
        return

    csv_path = os.path.join(out_dir, f"results_{args.split}.csv")
    if args.plot_only:
        import pandas as pd

        rows = pd.read_csv(csv_path).to_dict("records")
        plot_results(rows, out_dir, split=args.split, pred_len=args.pred_len)
        print(f"plotted from {csv_path}", flush=True)
        return

    meta = {
        "seq_len": SEQ_LEN,
        "pred_len": args.pred_len,
        "split": args.split,
        "note": "mask on full history; spike/shift on OT channel in history",
        "corruptions": CORRUPTIONS,
        "models": [{k: m[k] for k in ("tag", "label", "setting")} for m in eval_models(args.pred_len)],
    }
    with open(os.path.join(out_dir, f"experiment_meta_{args.split}.json"), "w") as f:
        json.dump(meta, f, indent=2)

    rows = []
    for m in eval_models(args.pred_len):
        print(f"\n=== {m['tag']} ===", flush=True)
        ns = make_args(m, args.pred_len)
        _set_device(ns, args.gpu, False)
        exp = Exp_Main(ns)
        ckpt = os.path.join("checkpoints", m["setting"], "checkpoint.pth")
        if not os.path.isfile(ckpt):
            print(f"SKIP missing {ckpt}", flush=True)
            continue
        exp.model.load_state_dict(torch.load(ckpt, map_location=exp.device))
        ch_std = get_train_stds(exp)

        for cname, ccfg in CORRUPTIONS:
            print(f"  {cname} ...", flush=True)
            met = eval_split(
                exp, ccfg, ch_std, seed=corruption_seed(args.seed, cname), split=args.split
            )
            row = {
                "model_tag": m["tag"],
                "model_label": m["label"],
                "setting": m["setting"],
                "corruption": cname,
                "split": args.split,
                **met,
            }
            rows.append(row)
            print(f"    mse={met['mse']:.4f} mae={met['mae']:.4f}", flush=True)

    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        plot_results(rows, out_dir, split=args.split, pred_len=args.pred_len)
        print(f"\nsaved -> {csv_path}", flush=True)
        print(f"figures -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
