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
    return shift_idx, n, shock_steps, int(recovery_steps)


def train_if_needed(args, model_id, head_type, des, out_dir):
    setting = f"{model_id}_PatchTST_ETTh1_sl512_pl720_{des}_0"
    ckpt_path = os.path.join("checkpoints", setting, "checkpoint.pth")
    log_path = os.path.join(out_dir, "logs", f"train_{des}.log")
    if os.path.exists(ckpt_path):
        return setting, ckpt_path, True

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
        model_id,
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
        str(args.epochs),
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
        head_type,
        "--short_setting",
        "1",
        "--des",
        des,
        "--skip_visual",
        "1",
    ]
    if head_type == "sacq_cross_attention":
        cmd += [
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
        ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"train failed: {des}, see {log_path}")
    return setting, ckpt_path, False


def build_eval_args(data_path, head_type):
    return SimpleNamespace(
        data="ETTh1",
        root_path="./dataset/",
        data_path=data_path,
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
        head_type=head_type,
        sacq_overlap_avg=0,
        sacq_alpha_mode="learnable",
        sacq_alpha_value=0.5,
        dsacq_enable=0 if head_type == "sacq_cross_attention" else 1,
        dsacq_mem_gate_enable=0 if head_type == "sacq_cross_attention" else 1,
        dsacq_drift_gate_enable=0 if head_type == "sacq_cross_attention" else 1,
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


def evaluate_on_shift(args, ckpt_path, shift_idx, out_prefix, device):
    eval_args = build_eval_args("ETTh1_shift.csv", args["head_type"])
    data_set, loader = data_provider(eval_args, "test")

    model = PatchTST.Model(eval_args).float().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    preds, trues, inputx = [], [], []
    test_border1 = 12 * 30 * 24 + 4 * 30 * 24 - eval_args.seq_len
    err_map = defaultdict(list)
    alpha_map = defaultdict(list)
    sample_base = 0

    with torch.no_grad():
        for batch_x, batch_y, _, _ in loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            out = model(batch_x)
            if isinstance(out, tuple):
                outputs, aux = out
            else:
                outputs, aux = out, {}
            f_dim = 0
            outputs = outputs[:, -eval_args.pred_len :, f_dim:]
            batch_y = batch_y[:, -eval_args.pred_len :, f_dim:]

            pred_np = outputs.detach().cpu().numpy()
            true_np = batch_y.detach().cpu().numpy()
            x_np = batch_x.detach().cpu().numpy()
            preds.append(pred_np)
            trues.append(true_np)
            inputx.append(x_np)

            alpha_seq = aux.get("sacq_alpha_seq", None)
            alpha_np = None
            if torch.is_tensor(alpha_seq):
                alpha_np = alpha_seq.detach().cpu().numpy()  # [B,C,H]

            bsz = pred_np.shape[0]
            for j in range(bsz):
                gi = sample_base + j
                sqe = (pred_np[j] - true_np[j]) ** 2  # [H,C]
                mse_h = sqe.mean(axis=1)  # [H]
                for h in range(eval_args.pred_len):
                    t_abs = test_border1 + eval_args.seq_len + gi + h
                    err_map[t_abs].append(float(mse_h[h]))
                    if alpha_np is not None:
                        alpha_h = float(alpha_np[j, :, h].mean())
                        alpha_map[t_abs].append(alpha_h)
            sample_base += bsz

    preds = np.array(preds).reshape(-1, preds[0].shape[-2], preds[0].shape[-1])
    trues = np.array(trues).reshape(-1, trues[0].shape[-2], trues[0].shape[-1])
    inputx = np.array(inputx).reshape(-1, inputx[0].shape[-2], inputx[0].shape[-1])
    mae, mse, rmse, mape, mspe, rse, corr, smape, mase, ql50, pwmae = metric(preds, trues, inputx=inputx)

    t_sorted = sorted(err_map.keys())
    mse_t = np.array([np.mean(err_map[t]) for t in t_sorted], dtype=np.float64)
    alpha_t = np.array([np.mean(alpha_map[t]) if t in alpha_map else np.nan for t in t_sorted], dtype=np.float64)
    t_arr = np.array(t_sorted, dtype=np.int64)

    pre_mask = t_arr < shift_idx
    post_mask = t_arr >= shift_idx
    baseline = float(np.mean(mse_t[pre_mask])) if np.any(pre_mask) else float(np.mean(mse_t))
    threshold = baseline * 1.2
    rec = -1
    if np.any(post_mask):
        post_idx = np.where(post_mask)[0]
        for idx in post_idx:
            if mse_t[idx] < threshold:
                rec = int(t_arr[idx] - shift_idx)
                break

    per_t_path = out_prefix + "_per_t.csv"
    with open(per_t_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_abs", "mse_t", "alpha_t"])
        for t, m, a in zip(t_arr, mse_t, alpha_t):
            w.writerow([int(t), float(m), "" if np.isnan(a) else float(a)])

    res = {
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
        "recovery_steps": rec,
        "per_t_path": per_t_path,
    }
    return res, t_arr, mse_t, alpha_t


def plot_shift_results(out_dir, shift_idx, f_curve, s_curve, s_alpha):
    t_f, mse_f = f_curve
    t_s, mse_s = s_curve
    t_a, alpha_s = s_alpha
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(t_f, mse_f, label="PatchTST+Flatten MSE_t", linewidth=1.4)
    ax1.plot(t_s, mse_s, label="PatchTST+SACQ MSE_t", linewidth=1.4)
    ax1.axvline(shift_idx, color="red", linestyle="--", linewidth=1.2, label=f"Shift @ {shift_idx}")
    ax1.set_xlabel("Absolute Time Index")
    ax1.set_ylabel("Per-time MSE")
    ax1.grid(alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(t_a, alpha_s, color="purple", alpha=0.65, linewidth=1.2, label="SACQ gate alpha_t")
    ax2.set_ylabel("Gate alpha")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    plt.title("Synthetic Shift: Recovery and SACQ Gate Dynamics")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "shift_recovery_plot.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="PatchTST F/S synthetic shift experiment on ETTh1")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="logs/shift_patchtst_etth1")
    parser.add_argument("--epochs", type=int, default=50)
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

    shift_idx, n_total, shock_steps, recovery_steps = make_shifted_etth1(
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

    gate_tag = int(round(args.sacq_gate_loss_weight * 100))
    sign_tag = int(round(args.sacq_corr_sign_loss_weight * 100))
    tag = f"g{gate_tag}s{sign_tag}"
    f_setting, f_ckpt, _ = train_if_needed(args, f"ShPtfF{tag}", "flatten", f"sF{tag}", args.out_dir)
    s_setting, s_ckpt, _ = train_if_needed(
        args,
        f"ShPtfS{tag}",
        "sacq_cross_attention",
        f"sS{tag}",
        args.out_dir,
    )

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    f_res, t_f, mse_f, _ = evaluate_on_shift(
        {"head_type": "flatten"}, f_ckpt, shift_idx, os.path.join(args.out_dir, "artifacts", "flatten"), device
    )
    s_res, t_s, mse_s, alpha_s = evaluate_on_shift(
        {"head_type": "sacq_cross_attention"},
        s_ckpt,
        shift_idx,
        os.path.join(args.out_dir, "artifacts", "sacq"),
        device,
    )

    plot_path = plot_shift_results(args.out_dir, shift_idx, (t_f, mse_f), (t_s, mse_s), (t_s, alpha_s))

    summary = {
        "dataset": "ETTh1",
        "seq_len": 512,
        "pred_len": 720,
        "shift_idx": int(shift_idx),
        "n_total": int(n_total),
        "shock_steps": int(shock_steps),
        "recovery_steps": int(recovery_steps),
        "level_mult_peak": float(args.level_mult_peak),
        "level_mult_steady": float(args.level_mult_steady),
        "trend_old_slope": 0.1,
        "trend_peak_slope": float(args.slope_peak),
        "trend_steady_slope": float(args.slope_steady),
        "flatten_setting": f_setting,
        "sacq_setting": s_setting,
        "flatten": f_res,
        "sacq": s_res,
        "plot_path": plot_path,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "head",
                "mse",
                "mae",
                "rse",
                "rmse",
                "mape",
                "mspe",
                "corr",
                "smape",
                "mase",
                "ql50",
                "pwmae",
                "baseline_mse_t",
                "threshold",
                "recovery_steps",
            ]
        )
        for name, rec in [("flatten", f_res), ("sacq", s_res)]:
            w.writerow(
                [
                    name,
                    rec["mse"],
                    rec["mae"],
                    rec["rse"],
                    rec["rmse"],
                    rec["mape"],
                    rec["mspe"],
                    rec["corr"],
                    rec["smape"],
                    rec["mase"],
                    rec["ql50"],
                    rec["pwmae"],
                    rec["baseline_mse_t"],
                    rec["threshold"],
                    rec["recovery_steps"],
                ]
            )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

