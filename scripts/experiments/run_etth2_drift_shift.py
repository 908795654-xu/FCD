"""ETTh2 synthetic drift: Flatten vs SACQ S1/S2/S12 (clean train, shifted test).

Train on clean ETTh2; evaluate per-timestep MSE on gradually shifted test stream.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_provider.data_factory import data_provider
from models import PatchTST
from scripts.experiments.shift_utils import (
    SHIFT_PRESETS,
    drift_strength_profile,
    make_shifted_dataset,
    shift_csv_name,
)
from utils.metrics import metric

SEQ_LEN = 512
PRED_LEN = 720
TEST_BORDER1 = 12 * 30 * 24 + 4 * 30 * 24 - SEQ_LEN

S1_DUAL = dict(
    sacq_dual_struct_gate=0,
    sacq_dual_res_fuse=0,
    sacq_dual_detach=1,
    sacq_dual_scaffold_ln=0,
)

VARIANTS = [
    {
        "tag": "Flatten",
        "setting": "TwF2s512p720_PatchTST_ETTh2_sl512_pl720_hF_0",
        "head_type": "flatten",
        "sacq_scheme": "default",
        "sacq_pe_mode": "default",
        "use_dual": False,
        "color": "#d62728",
        "ls": "-",
    },
    {
        "tag": "S1",
        "setting": "H2s512p720S1_PatchTST_ETTh2_sl512_pl720_hS1_0",
        "head_type": "sacq_cross_attention",
        "sacq_scheme": "dual_residual",
        "sacq_pe_mode": "default",
        "use_dual": True,
        "color": "#1f77b4",
        "ls": "-",
    },
    {
        "tag": "S2",
        "setting": "H2s512p720S2_PatchTST_ETTh2_sl512_pl720_hS2_0",
        "head_type": "sacq_cross_attention",
        "sacq_scheme": "default",
        "sacq_pe_mode": "relative_bias",
        "use_dual": False,
        "color": "#ff7f0e",
        "ls": "--",
    },
    {
        "tag": "S12",
        "setting": "H2s512p720S12_PatchTST_ETTh2_sl512_pl720_hS12_0",
        "head_type": "sacq_cross_attention",
        "sacq_scheme": "dual_residual",
        "sacq_pe_mode": "relative_bias",
        "use_dual": True,
        "color": "#2ca02c",
        "ls": "-",
    },
    {
        "tag": "S1d",
        "setting": "H2s512p720S1d_PatchTST_ETTh2_sl512_pl720_hS1d_0",
        "head_type": "sacq_cross_attention",
        "sacq_scheme": "dual_residual",
        "sacq_pe_mode": "default",
        "use_dual": True,
        "color": "#1f77b4",
        "ls": ":",
    },
    {
        "tag": "S2d",
        "setting": "H2s512p720S2d_PatchTST_ETTh2_sl512_pl720_hS2d_0",
        "head_type": "sacq_cross_attention",
        "sacq_scheme": "default",
        "sacq_pe_mode": "relative_bias",
        "use_dual": False,
        "color": "#ff7f0e",
        "ls": ":",
    },
    {
        "tag": "S12d",
        "setting": "H2s512p720S12d_PatchTST_ETTh2_sl512_pl720_hS12d_0",
        "head_type": "sacq_cross_attention",
        "sacq_scheme": "dual_residual",
        "sacq_pe_mode": "relative_bias",
        "use_dual": True,
        "color": "#2ca02c",
        "ls": ":",
    },
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_eval_args(data_path: str, variant: dict) -> SimpleNamespace:
    kw = dict(S1_DUAL) if variant["use_dual"] else {}
    return SimpleNamespace(
        random_seed=2021,
        is_training=0,
        model="PatchTST",
        data="ETTh2",
        root_path="./dataset/",
        data_path=data_path,
        features="M",
        target="OT",
        freq="h",
        seq_len=SEQ_LEN,
        label_len=48,
        pred_len=PRED_LEN,
        enc_in=7,
        dec_in=7,
        c_out=7,
        d_model=16,
        n_heads=4,
        e_layers=3,
        d_layers=1,
        d_ff=128,
        individual=0,
        moving_avg=25,
        factor=1,
        distil=False,
        dropout=0.3,
        pe="zeros",
        learn_pe=True,
        sacq_autoregressive=0,
        decomposition=0,
        kernel_size=25,
        affine_group_size=0,
        fc_dropout=0.3,
        head_dropout=0.0,
        patch_len=16,
        stride=8,
        head_type=variant["head_type"],
        cross_attn_layers=2,
        sacq_scheme=variant["sacq_scheme"],
        sacq_pe_mode=variant["sacq_pe_mode"],
        sacq_pe_rel_buckets=32,
        padding_patch="end",
        revin=1,
        affine=0,
        subtract_last=0,
        sacq_overlap_avg=0,
        sacq_alpha_mode="learnable",
        sacq_alpha_value=0.5,
        sacq_alpha_granularity="patch",
        sacq_drift_alpha_enable=1,
        sacq_alpha_temperature=0.5,
        dsacq_enable=0,
        dsacq_mem_gate_enable=0,
        dsacq_drift_gate_enable=0,
        embed="timeF",
        activation="gelu",
        num_workers=4,
        batch_size=128,
        **kw,
    )


def evaluate_variant(
    variant: dict, ckpt_path: str, device: torch.device, data_path: str, shift_idx: int
):
    eval_args = build_eval_args(data_path, variant)
    _, loader = data_provider(eval_args, "test")
    model = PatchTST.Model(eval_args).float().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    err_map: dict[int, list[float]] = defaultdict(list)
    alpha_map: dict[int, list[float]] = defaultdict(list)
    drift_only_err: list[float] = []
    preds, trues, inputx = [], [], []
    sample_base = 0

    with torch.no_grad():
        for batch_x, batch_y, _, _ in loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            out = model(batch_x)
            outputs, aux = out if isinstance(out, tuple) else (out, {})
            outputs = outputs[:, -PRED_LEN:, 0:1]
            batch_y = batch_y[:, -PRED_LEN:, 0:1]

            pred_np = outputs.detach().cpu().numpy()
            true_np = batch_y.detach().cpu().numpy()
            preds.append(pred_np)
            trues.append(true_np)
            inputx.append(batch_x.detach().cpu().numpy())

            alpha_seq = aux.get("sacq_alpha_seq") if aux else None
            alpha_np = alpha_seq.detach().cpu().numpy() if torch.is_tensor(alpha_seq) else None

            bsz = pred_np.shape[0]
            for j in range(bsz):
                gi = sample_base + j
                # 仅当输入窗口末端已进入漂移区，才计入漂移段误差
                input_end = TEST_BORDER1 + gi + SEQ_LEN
                mse_h = ((pred_np[j] - true_np[j]) ** 2).mean(axis=1)
                for h in range(PRED_LEN):
                    t_abs = TEST_BORDER1 + SEQ_LEN + gi + h
                    err_map[t_abs].append(float(mse_h[h]))
                    if alpha_np is not None:
                        alpha_map[t_abs].append(float(alpha_np[j, :, h].mean()))
                    if input_end >= shift_idx:
                        drift_only_err.append(float(mse_h[h]))
            sample_base += bsz

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    inputx = np.concatenate(inputx, axis=0)
    mae, mse, rmse, mape, mspe, rse, corr, smape, mase, ql50, pwmae = metric(
        preds, trues, inputx=inputx
    )

    t_sorted = sorted(err_map.keys())
    mse_t = np.array([np.mean(err_map[t]) for t in t_sorted], dtype=np.float64)
    alpha_t = np.array(
        [np.mean(alpha_map[t]) if t in alpha_map else np.nan for t in t_sorted],
        dtype=np.float64,
    )
    t_arr = np.array(t_sorted, dtype=np.int64)

    return {
        "tag": variant["tag"],
        "mse": float(mse),
        "mae": float(mae),
        "rse": float(rse),
        "mse_drift_only": float(np.mean(drift_only_err)) if drift_only_err else float("nan"),
        "t": t_arr,
        "mse_t": mse_t,
        "alpha_t": alpha_t,
    }


def slope_after_shift(t: np.ndarray, mse_t: np.ndarray, shift_idx: int, window: int = 400) -> float:
    post = (t >= shift_idx) & (t < shift_idx + window) & np.isfinite(mse_t)
    if post.sum() < 10:
        return float("nan")
    x = (t[post] - shift_idx).astype(np.float64)
    y = mse_t[post]
    x = (x - x.mean()) / (x.std() + 1e-8)
    return float(np.polyfit(x, y, 1)[0])


def plot_vs_clean_baseline(out_dir: str, shift_idx: int, curves_shift: list[dict], device, preset: str):
    """MSE on shifted data divided by MSE on clean data (same checkpoints)."""
    curves_clean = []
    for variant in VARIANTS:
        ckpt = os.path.join(PROJECT_ROOT, "checkpoints", variant["setting"], "checkpoint.pth")
        if not os.path.isfile(ckpt):
            continue
        res = evaluate_variant(variant, ckpt, device, "ETTh2.csv", shift_idx)
        curves_clean.append({**variant, **res})

    fig, ax = plt.subplots(figsize=(12, 5))
    for vs, vc in zip(curves_shift, curves_clean):
        if vs["tag"] != vc["tag"]:
            continue
        post = vs["t"] >= shift_idx
        t = vs["t"][post] - shift_idx
        ratio = vs["mse_t"][post] / np.maximum(vc["mse_t"][post], 1e-8)
        # smooth
        win = 31
        if len(ratio) >= win:
            kernel = np.ones(win) / win
            ratio = np.convolve(ratio, kernel, mode="same")
        ax.plot(t, ratio, label=vs["tag"], color=vs["color"], ls=vs["ls"], lw=1.6)
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.set_xlabel("漂移后时间步")
    ax.set_ylabel("MSE(漂移测试) / MSE(洁净测试)")
    ax.set_title("相对洁净基线的恶化倍数（>1 表示漂移导致误差上升）")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", ncol=4, fontsize=9)
    path = os.path.join(out_dir, "drift_mse_ratio_vs_clean.png")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_results(
    out_dir: str,
    shift_idx: int,
    curves: list[dict],
    drift_t: np.ndarray,
    drift_strength: np.ndarray,
    preset: str,
) -> str:
    ensure_dir(out_dir)

    # --- 1) absolute time ---
    fig, ax = plt.subplots(figsize=(12, 5))
    for c in curves:
        ax.plot(c["t"], c["mse_t"], label=c["tag"], color=c["color"], ls=c["ls"], linewidth=1.5)
    ax.axvline(shift_idx, color="black", ls="--", lw=1.2, label=f"漂移起点 @ {shift_idx}")
    ax.set_xlabel("绝对时间索引")
    ax.set_ylabel("逐步 MSE")
    ax.set_title(f"ETTh2 pl={PRED_LEN} | 洁净训练 → {preset} 漂移测试")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    p1 = os.path.join(out_dir, "drift_mse_vs_time.png")
    fig.tight_layout()
    fig.savefig(p1, dpi=160)
    plt.close(fig)

    # --- 2) post-shift: MSE vs drift strength (binned) ---
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(drift_strength.min(), drift_strength.max(), 40)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    for c in curves:
        post_mask = c["t"] >= shift_idx
        t_post = c["t"][post_mask]
        mse_post = c["mse_t"][post_mask]
        # map t to drift strength
        idx_map = {int(t): float(s) for t, s in zip(drift_t, drift_strength)}
        ds = np.array([idx_map.get(int(t), np.nan) for t in t_post])
        valid = np.isfinite(ds)
        if valid.sum() < 20:
            continue
        digit = np.digitize(ds[valid], bins) - 1
        digit = np.clip(digit, 0, len(bin_centers) - 1)
        means = np.full(len(bin_centers), np.nan)
        for b in range(len(bin_centers)):
            sel = digit == b
            if sel.any():
                means[b] = mse_post[valid][sel].mean()
        ax.plot(bin_centers, means, label=c["tag"], color=c["color"], ls=c["ls"], linewidth=2.0)
    ax.set_xlabel("漂移强度（level 偏移 + |slope 变化|）")
    ax.set_ylabel("MSE（漂移段分箱均值）")
    ax.set_title("随漂移逐步加剧：MSE 上升曲线")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    p2 = os.path.join(out_dir, "drift_mse_vs_intensity.png")
    fig.tight_layout()
    fig.savefig(p2, dpi=160)
    plt.close(fig)

    # --- 3) normalized degradation (ratio to pre-shift mean) ---
    fig, ax = plt.subplots(figsize=(12, 5))
    steps = None
    for c in curves:
        pre = c["t"] < shift_idx
        base = float(c["mse_t"][pre].mean()) if pre.any() else float(c["mse_t"].mean())
        post = c["t"] >= shift_idx
        rel = c["mse_t"][post] / max(base, 1e-8)
        st = c["t"][post] - shift_idx
        ax.plot(st, rel, label=c["tag"], color=c["color"], ls=c["ls"], linewidth=1.5)
        if steps is None:
            steps = st
    ax.set_xlabel("漂移后步数")
    ax.set_ylabel("MSE / 漂移前均值")
    ax.set_title("相对恶化：Flatten 应陡升，SACQ 应缓升")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    p3 = os.path.join(out_dir, "drift_mse_relative_post.png")
    fig.tight_layout()
    fig.savefig(p3, dpi=160)
    plt.close(fig)

    # --- 4) summary bar: slope in first window after shift ---
    tags = [c["tag"] for c in curves]
    slopes = [slope_after_shift(c["t"], c["mse_t"], shift_idx) for c in curves]
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [c["color"] for c in curves]
    ax.bar(tags, slopes, color=colors, alpha=0.85)
    ax.set_ylabel(f"漂移后 {400} 步内 MSE 线性斜率")
    ax.set_title("恶化速度（越大越快）")
    ax.grid(axis="y", alpha=0.25)
    p4 = os.path.join(out_dir, "drift_degradation_slope.png")
    fig.tight_layout()
    fig.savefig(p4, dpi=160)
    plt.close(fig)

    # --- 5) combined panel for paper ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [1, 1.1]})
    ax0, ax1 = axes
    for c in curves:
        ax0.plot(c["t"], c["mse_t"], label=c["tag"], color=c["color"], ls=c["ls"], lw=1.4)
    ax0.axvline(shift_idx, color="k", ls="--", lw=1)
    ax0.set_ylabel("MSE")
    ax0.set_title(f"ETTh2 数据漂移实验 ({preset})")
    ax0.grid(alpha=0.2)
    ax0.legend(fontsize=8, ncol=4, loc="upper left")

    for c in curves:
        pre = c["t"] < shift_idx
        base = float(c["mse_t"][pre].mean()) if pre.any() else 1.0
        post = c["t"] >= shift_idx
        ax1.plot(
            c["t"][post] - shift_idx,
            c["mse_t"][post] / base,
            label=c["tag"],
            color=c["color"],
            ls=c["ls"],
            lw=1.6,
        )
    ax1.set_xlabel("漂移后时间步")
    ax1.set_ylabel("MSE / 漂移前基线")
    ax1.set_title("相对恶化（SACQ 应更平缓）")
    ax1.grid(alpha=0.2)
    ax1.legend(fontsize=8, ncol=4, loc="upper left")
    fig.tight_layout()
    p_main = os.path.join(out_dir, "drift_main_panel.png")
    fig.savefig(p_main, dpi=170, bbox_inches="tight")
    plt.close(fig)

    return p_main


def write_results_md(path: str, summary: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# ETTh2 数据漂移实验（洁净训练 → 漂移测试）\n\n")
        f.write(f"- 预设: `{summary['preset']}`\n")
        f.write(f"- shift_idx: {summary['shift_idx']}\n")
        f.write(f"- pred_len: {PRED_LEN}, seq_len: {SEQ_LEN}\n\n")
        f.write("## 全程 Test MSE（漂移测试集）\n\n")
        f.write("| 模型 | MSE | MAE | 漂移后400步斜率 |\n")
        f.write("|------|-----|-----|----------------|\n")
        for row in summary["models"]:
            f.write(
                f"| {row['tag']} | {row['mse']:.4f} | {row['mae']:.4f} | "
                f"{row.get('slope_post', float('nan')):.6f} |\n"
            )
        f.write("\n## 图\n\n")
        for k, v in summary.get("plots", {}).items():
            f.write(f"- `{k}`: {v}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=str, default="gradual_ramp", choices=list(SHIFT_PRESETS.keys()))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="logs/etth2_drift_shift")
    args = parser.parse_args()

    out_dir = os.path.join(PROJECT_ROOT, args.out_dir, args.preset)
    art = os.path.join(out_dir, "artifacts")
    ensure_dir(art)

    shift_idx, n_total, dst_csv, info = make_shifted_dataset(
        "ETTh2", args.preset, root_path=os.path.join(PROJECT_ROOT, "dataset")
    )
    data_path = os.path.basename(dst_csv)
    drift_t, _, drift_strength = drift_strength_profile(shift_idx, n_total, info["params"])

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    curves = []
    summary_rows = []

    for variant in VARIANTS:
        ckpt = os.path.join(PROJECT_ROOT, "checkpoints", variant["setting"], "checkpoint.pth")
        if not os.path.isfile(ckpt):
            print("missing", ckpt)
            continue
        print("eval", variant["tag"], "...")
        res = evaluate_variant(variant, ckpt, device, data_path, shift_idx)
        variant["color"] = variant["color"]
        variant["ls"] = variant["ls"]
        curves.append({**variant, **res})

        per_t = os.path.join(art, f"{variant['tag']}_per_t.csv")
        with open(per_t, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_abs", "mse_t", "alpha_t"])
            for t, m, a in zip(res["t"], res["mse_t"], res["alpha_t"]):
                w.writerow([int(t), float(m), "" if np.isnan(a) else float(a)])

        sp = slope_after_shift(res["t"], res["mse_t"], shift_idx)
        summary_rows.append(
            {"tag": res["tag"], "mse": res["mse"], "mae": res["mae"], "slope_post": sp}
        )
        print(f"  {variant['tag']} MSE={res['mse']:.4f} slope_post={sp:.6f}")

    main_plot = plot_results(out_dir, shift_idx, curves, drift_t, drift_strength, args.preset)
    ratio_plot = plot_vs_clean_baseline(out_dir, shift_idx, curves, device, args.preset)

    summary = {
        "preset": args.preset,
        "shift_idx": int(shift_idx),
        "n_total": int(n_total),
        "data_path": data_path,
        "info": info,
        "models": summary_rows,
        "plots": {
            "main": main_plot,
            "time": os.path.join(out_dir, "drift_mse_vs_time.png"),
            "intensity": os.path.join(out_dir, "drift_mse_vs_intensity.png"),
            "relative": os.path.join(out_dir, "drift_mse_relative_post.png"),
            "slope_bar": os.path.join(out_dir, "drift_degradation_slope.png"),
            "ratio_vs_clean": ratio_plot,
        },
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    write_results_md(os.path.join(out_dir, "RESULTS.md"), summary)
    print("wrote", main_plot)


if __name__ == "__main__":
    main()
