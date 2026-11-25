import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_provider.data_factory import data_provider
from models import PatchTST
from utils.metrics import metric


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def make_shifted_etth1(
    src_csv,
    dst_csv,
    shift_ratio=0.8,
    shock_steps=300,
    recovery_steps=200,
    level_mult_peak=1.5,
    level_mult_steady=1.15,
    old_slope=0.1,
    slope_peak=-0.2,
    slope_steady=-0.05,
):
    df = pd.read_csv(src_csv)
    n = len(df)
    shift_idx = int(n * shift_ratio)
    numeric_cols = [c for c in df.columns if c != "date"]
    after_len = n - shift_idx
    mult = np.full(after_len, level_mult_steady, dtype=np.float32)
    slope = np.full(after_len, slope_steady, dtype=np.float32)
    shock_steps = max(1, min(int(shock_steps), after_len))
    mult[:shock_steps] = level_mult_peak
    slope[:shock_steps] = slope_peak
    rec_start = shock_steps
    rec_end = min(shock_steps + max(int(recovery_steps), 0), after_len)
    if rec_end > rec_start:
        ratio = np.linspace(0.0, 1.0, rec_end - rec_start, dtype=np.float32)
        mult[rec_start:rec_end] = level_mult_peak + (level_mult_steady - level_mult_peak) * ratio
        slope[rec_start:rec_end] = slope_peak + (slope_steady - slope_peak) * ratio
    slope_diff = slope - old_slope
    trend_delta = np.concatenate(
        [np.array([0.0], dtype=np.float32), np.cumsum(slope_diff[:-1], dtype=np.float32)]
    )
    shifted = df.copy()
    shifted_block = shifted.loc[shift_idx:, numeric_cols].values.astype(np.float32)
    shifted_block = shifted_block * mult[:, None] + trend_delta[:, None]
    shifted.loc[shift_idx:, numeric_cols] = shifted_block
    shifted.to_csv(dst_csv, index=False)
    return shift_idx, n


def run_train_cmd(cmd, gpu_id, log_path):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"train failed: see {log_path}")


def train_stage1(args, out_dir):
    setting = "TwS1_PatchTST_ETTh1_sl512_pl720_s1_0"
    ckpt = os.path.join("checkpoints", setting, "checkpoint.pth")
    if os.path.exists(ckpt):
        return setting
    cmd = [
        "python",
        "-u",
        "run_longExp.py",
        "--is_training",
        "1",
        "--random_seed",
        str(args.seed),
        "--root_path",
        "./dataset/",
        "--data_path",
        "ETTh1.csv",
        "--model_id",
        "TwS1",
        "--model",
        "PatchTST",
        "--data",
        "ETTh1",
        "--features",
        "M",
        "--freq",
        "h",
        "--seq_len",
        "512",
        "--pred_len",
        "720",
        "--enc_in",
        "7",
        "--dec_in",
        "7",
        "--c_out",
        "7",
        "--e_layers",
        str(args.e_layers),
        "--n_heads",
        str(args.n_heads),
        "--d_model",
        str(args.d_model),
        "--d_ff",
        str(args.d_ff),
        "--dropout",
        str(args.dropout),
        "--fc_dropout",
        str(args.fc_dropout),
        "--head_dropout",
        "0",
        "--patch_len",
        str(args.patch_len),
        "--stride",
        str(args.stride),
        "--train_epochs",
        str(args.epochs_stage1),
        "--patience",
        str(args.patience),
        "--batch_size",
        str(args.batch_size),
        "--learning_rate",
        str(args.lr),
        "--itr",
        "1",
        "--num_workers",
        str(args.num_workers),
        "--head_type",
        "sacq_cross_attention",
        "--sacq_alpha_mode",
        "fixed",
        "--sacq_alpha_value",
        "0.5",
        "--dsacq_enable",
        "0",
        "--dsacq_mem_gate_enable",
        "0",
        "--dsacq_drift_gate_enable",
        "0",
        "--short_setting",
        "1",
        "--des",
        "s1",
        "--skip_visual",
        "1",
    ]
    run_train_cmd(cmd, args.gpu, os.path.join(out_dir, "logs", "stage1_train.log"))
    return setting


def train_stage2(args, out_dir, init_setting):
    setting = "TwS2_PatchTST_ETTh1_sl512_pl720_s2_0"
    ckpt = os.path.join("checkpoints", setting, "checkpoint.pth")
    if os.path.exists(ckpt):
        return setting
    cmd = [
        "python",
        "-u",
        "run_longExp.py",
        "--is_training",
        "1",
        "--random_seed",
        str(args.seed),
        "--root_path",
        "./dataset/",
        "--data_path",
        "ETTh1.csv",
        "--model_id",
        "TwS2",
        "--model",
        "PatchTST",
        "--data",
        "ETTh1",
        "--features",
        "M",
        "--freq",
        "h",
        "--seq_len",
        "512",
        "--pred_len",
        "720",
        "--enc_in",
        "7",
        "--dec_in",
        "7",
        "--c_out",
        "7",
        "--e_layers",
        str(args.e_layers),
        "--n_heads",
        str(args.n_heads),
        "--d_model",
        str(args.d_model),
        "--d_ff",
        str(args.d_ff),
        "--dropout",
        str(args.dropout),
        "--fc_dropout",
        str(args.fc_dropout),
        "--head_dropout",
        "0",
        "--patch_len",
        str(args.patch_len),
        "--stride",
        str(args.stride),
        "--train_epochs",
        str(args.epochs_stage2),
        "--patience",
        str(args.patience),
        "--batch_size",
        str(args.batch_size),
        "--learning_rate",
        str(args.lr),
        "--itr",
        "1",
        "--num_workers",
        str(args.num_workers),
        "--head_type",
        "sacq_cross_attention",
        "--sacq_alpha_mode",
        "learnable",
        "--sacq_alpha_value",
        "0.5",
        "--dsacq_enable",
        "0",
        "--dsacq_mem_gate_enable",
        "0",
        "--dsacq_drift_gate_enable",
        "0",
        "--sacq_gate_loss_weight",
        str(args.sacq_gate_loss_weight),
        "--sacq_corr_sign_loss_weight",
        str(args.sacq_corr_sign_loss_weight),
        "--init_checkpoint_setting",
        init_setting,
        "--short_setting",
        "1",
        "--des",
        "s2",
        "--skip_visual",
        "1",
    ]
    run_train_cmd(cmd, args.gpu, os.path.join(out_dir, "logs", "stage2_train.log"))
    return setting


def build_eval_args():
    return SimpleNamespace(
        data="ETTh1",
        root_path="./dataset/",
        data_path="ETTh1_shift.csv",
        features="M",
        target="OT",
        freq="h",
        embed="timeF",
        seq_len=512,
        label_len=48,
        pred_len=720,
        enc_in=7,
        dec_in=7,
        c_out=7,
        model="PatchTST",
        e_layers=3,
        n_heads=4,
        d_model=16,
        d_ff=128,
        dropout=0.2,
        fc_dropout=0.2,
        head_dropout=0.0,
        patch_len=16,
        stride=8,
        padding_patch="end",
        revin=1,
        affine=0,
        subtract_last=0,
        individual=0,
        decomposition=0,
        kernel_size=25,
        cross_attn_layers=2,
        head_type="sacq_cross_attention",
        sacq_overlap_avg=0,
        sacq_alpha_mode="learnable",
        sacq_alpha_value=0.5,
        dsacq_enable=0,
        dsacq_mem_gate_enable=0,
        dsacq_drift_gate_enable=0,
        dsacq_mem_gate_hidden=32,
        dsacq_mem_gate_mode="stats",
        mem_shift_aug_std=0.15,
        mem_shift_aug_prob=0.5,
        align_ablate_shift=0,
        align_ablate_dual_bank=0,
        align_ablate_token_gate=0,
        align_ablate_shift_drift=0,
        batch_size=128,
        num_workers=4,
    )


def evaluate_twostage(ckpt_path, shift_idx, out_csv, device):
    args = build_eval_args()
    _, loader = data_provider(args, "test")
    model = PatchTST.Model(args).float().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    preds, trues, inputx = [], [], []
    test_border1 = 12 * 30 * 24 + 4 * 30 * 24 - args.seq_len
    err_map = defaultdict(list)
    alpha_map = defaultdict(list)
    base = 0
    with torch.no_grad():
        for bx, by, _, _ in loader:
            bx = bx.float().to(device)
            by = by.float().to(device)
            out = model(bx)
            outputs, aux = out if isinstance(out, tuple) else (out, {})
            outputs = outputs[:, -args.pred_len :, :]
            by = by[:, -args.pred_len :, :]
            p = outputs.detach().cpu().numpy()
            y = by.detach().cpu().numpy()
            x = bx.detach().cpu().numpy()
            preds.append(p)
            trues.append(y)
            inputx.append(x)
            alpha_seq = aux.get("sacq_alpha_seq", None)
            alpha_np = alpha_seq.detach().cpu().numpy() if torch.is_tensor(alpha_seq) else None
            bsz = p.shape[0]
            for j in range(bsz):
                gi = base + j
                mse_h = ((p[j] - y[j]) ** 2).mean(axis=1)
                for h in range(args.pred_len):
                    t_abs = test_border1 + args.seq_len + gi + h
                    err_map[t_abs].append(float(mse_h[h]))
                    if alpha_np is not None:
                        alpha_map[t_abs].append(float(alpha_np[j, :, h].mean()))
            base += bsz

    preds = np.array(preds).reshape(-1, preds[0].shape[-2], preds[0].shape[-1])
    trues = np.array(trues).reshape(-1, trues[0].shape[-2], trues[0].shape[-1])
    inputx = np.array(inputx).reshape(-1, inputx[0].shape[-2], inputx[0].shape[-1])
    mae, mse, rmse, mape, mspe, rse, corr, smape, mase, ql50, pwmae = metric(preds, trues, inputx=inputx)

    t_sorted = sorted(err_map.keys())
    mse_t = np.array([np.mean(err_map[t]) for t in t_sorted], dtype=np.float64)
    alpha_t = np.array([np.mean(alpha_map[t]) if t in alpha_map else np.nan for t in t_sorted], dtype=np.float64)
    t_arr = np.array(t_sorted, dtype=np.int64)
    pre = t_arr < shift_idx
    post = t_arr >= shift_idx
    baseline = float(np.mean(mse_t[pre])) if np.any(pre) else float(np.mean(mse_t))
    threshold = baseline * 1.2
    recovery = -1
    if np.any(post):
        for idx in np.where(post)[0]:
            if mse_t[idx] < threshold:
                recovery = int(t_arr[idx] - shift_idx)
                break

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_abs", "mse_t", "alpha_t"])
        for t, m, a in zip(t_arr, mse_t, alpha_t):
            w.writerow([int(t), float(m), float(a)])
    return {
        "mse": float(mse),
        "mae": float(mae),
        "rse": float(rse),
        "rmse": float(rmse),
        "mape": float(mape),
        "mspe": float(mspe),
        "corr": float(corr),
        "smape": float(smape),
        "mase": float(mase),
        "ql50": float(ql50),
        "pwmae": float(pwmae),
        "baseline_mse_t": baseline,
        "threshold": threshold,
        "recovery_steps": recovery,
        "alpha_pre_mean": float(np.nanmean(alpha_t[pre])),
        "alpha_post300_mean": float(np.nanmean(alpha_t[(t_arr >= shift_idx) & (t_arr < shift_idx + 300)])),
        "alpha_post300_max": float(np.nanmax(alpha_t[(t_arr >= shift_idx) & (t_arr < shift_idx + 300)])),
    }, t_arr, mse_t, alpha_t


def plot_curve(out_path, shift_idx, t_arr, mse_t, alpha_t):
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(t_arr, mse_t, linewidth=1.4, label="MSE_t")
    ax1.axvline(shift_idx, color="red", linestyle="--", linewidth=1.2, label=f"Shift @ {shift_idx}")
    ax1.set_xlabel("Absolute Time Index")
    ax1.set_ylabel("Per-time MSE")
    ax1.grid(alpha=0.2)
    ax2 = ax1.twinx()
    ax2.plot(t_arr, alpha_t, color="purple", linewidth=1.2, alpha=0.7, label="alpha_t")
    ax2.set_ylabel("Gate alpha")
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="upper right")
    plt.title("Two-stage SACQ training: shift recovery and gate dynamics")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Two-stage SACQ training on ETTh1 shift experiment")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="logs/shift_patchtst_etth1_twostage")
    parser.add_argument("--epochs_stage1", type=int, default=30)
    parser.add_argument("--epochs_stage2", type=int, default=30)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--e_layers", type=int, default=3)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--fc_dropout", type=float, default=0.2)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--sacq_gate_loss_weight", type=float, default=0.2)
    parser.add_argument("--sacq_corr_sign_loss_weight", type=float, default=0.1)
    parser.add_argument("--shock_steps", type=int, default=300)
    parser.add_argument("--recovery_steps", type=int, default=200)
    parser.add_argument("--level_mult_peak", type=float, default=1.5)
    parser.add_argument("--level_mult_steady", type=float, default=1.15)
    parser.add_argument("--slope_peak", type=float, default=-0.2)
    parser.add_argument("--slope_steady", type=float, default=-0.05)
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    ensure_dir(os.path.join(args.out_dir, "logs"))
    ensure_dir(os.path.join(args.out_dir, "artifacts"))
    shift_idx, n_total = make_shifted_etth1(
        "./dataset/ETTh1.csv",
        "./dataset/ETTh1_shift.csv",
        shift_ratio=0.8,
        shock_steps=args.shock_steps,
        recovery_steps=args.recovery_steps,
        level_mult_peak=args.level_mult_peak,
        level_mult_steady=args.level_mult_steady,
        old_slope=0.1,
        slope_peak=args.slope_peak,
        slope_steady=args.slope_steady,
    )
    s1 = train_stage1(args, args.out_dir)
    s2 = train_stage2(args, args.out_dir, s1)
    ckpt = os.path.join("checkpoints", s2, "checkpoint.pth")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    res, t_arr, mse_t, alpha_t = evaluate_twostage(
        ckpt, shift_idx, os.path.join(args.out_dir, "artifacts", "twostage_per_t.csv"), device
    )
    plot_path = os.path.join(args.out_dir, "twostage_shift_curve.png")
    plot_curve(plot_path, shift_idx, t_arr, mse_t, alpha_t)

    summary = {
        "dataset": "ETTh1",
        "seq_len": 512,
        "pred_len": 720,
        "shift_idx": int(shift_idx),
        "n_total": int(n_total),
        "stage1_setting": s1,
        "stage2_setting": s2,
        "sacq_gate_loss_weight": float(args.sacq_gate_loss_weight),
        "sacq_corr_sign_loss_weight": float(args.sacq_corr_sign_loss_weight),
        "result": res,
        "plot_path": plot_path,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

