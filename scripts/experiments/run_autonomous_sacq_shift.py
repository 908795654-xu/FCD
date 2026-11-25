#!/usr/bin/env python3
"""
Autonomous SACQ shift experiment runner.

Trains basic SACQ (DSACQ off), evaluates on synthetic shifts, maintains leaderboard.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_provider.data_factory import data_provider
from scripts.experiments.shift_utils import (
    DATASET_META,
    SHIFT_PRESETS,
    make_shifted_dataset,
    shift_csv_name,
)
from utils.metrics import metric

CONDA_PYTHON = os.environ.get(
    "SACQ_PYTHON",
    "/home/bkai/miniconda3/bin/conda run -n forecast python",
).split()

LEADERBOARD_FIELDS = [
    "run_id",
    "timestamp",
    "status",
    "score",
    "train_dataset",
    "eval_dataset",
    "shift_preset",
    "model",
    "head_type",
    "twostage",
    "mse",
    "mae",
    "recovery_steps",
    "baseline_mse_t",
    "alpha_pre_mean",
    "alpha_post_mean",
    "alpha_post300_mean",
    "alpha_post300_max",
    "alpha_post_delta",
    "alpha_post_max_delta",
    "sacq_gate_loss_weight",
    "sacq_corr_sign_loss_weight",
    "sacq_shift_boost_loss_weight",
    "sacq_alpha_temperature",
    "sacq_drift_alpha_enable",
    "setting",
    "ckpt_path",
    "log_path",
    "plot_path",
]

EXPERIMENT_GRID: List[Dict[str, Any]] = [
  dict(name="pt_base", model="PatchTST", d_model=16, gate=0.2, sign=0.1, boost=0.0, tau=0.5, twostage=0),
  dict(name="pt_drift_v1", model="PatchTST", d_model=16, gate=0.3, sign=0.15, boost=0.1, tau=0.35, twostage=0),
  dict(name="pt_drift_v2", model="PatchTST", d_model=16, gate=0.5, sign=0.2, boost=0.15, tau=0.3, twostage=0),
  dict(name="pt_drift_v1_ts", model="PatchTST", d_model=16, gate=0.3, sign=0.15, boost=0.1, tau=0.35, twostage=1),
  dict(name="lin_drift_v1", model="Linear", d_model=128, gate=0.3, sign=0.15, boost=0.1, tau=0.35, twostage=0),
  dict(name="arima_drift_v1", model="ARIMA", d_model=64, gate=0.3, sign=0.15, boost=0.1, tau=0.35, twostage=0),
  dict(name="nbeats_drift_v1", model="NBEATS", d_model=64, gate=0.3, sign=0.15, boost=0.1, tau=0.35, twostage=0),
]

EVAL_MATRIX: List[Tuple[str, str, str]] = [
    ("ETTh1", "ETTh1", "stage_combo"),
    ("ETTh1", "ETTh1", "mild_stage"),
    ("ETTh1", "ETTh1", "level_only"),
    ("ETTh1", "ETTh1", "trend_only"),
    ("ETTh1", "ETTh1", "spike_revert"),
    ("ETTh1", "ETTh2", "stage_combo"),
    ("ETTh1", "ETTm2", "mild_stage"),
    ("ETTm2", "ETTm2", "stage_combo"),
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_id_for(cfg: Dict[str, Any], train_ds: str, eval_ds: str, shift_preset: str) -> str:
    raw = json.dumps(
        {
            "cfg": cfg,
            "train": train_ds,
            "eval": eval_ds,
            "shift": shift_preset,
        },
        sort_keys=True,
    )
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def load_leaderboard(path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(path):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", newline="") as f:
        for row in csv.DictReader(f):
            rid = row.get("run_id", "")
            if rid:
                out[rid] = row
    return out


def append_leaderboard(path: str, row: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEADERBOARD_FIELDS, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LEADERBOARD_FIELDS})


def composite_score(row: Dict[str, Any]) -> float:
    mse = float(row.get("mse", 1e9))
    rec = int(float(row.get("recovery_steps", -1)))
    rec_pen = 0.0 if rec >= 0 else 80.0
    alpha_delta = float(row.get("alpha_post_delta", 0.0) or 0.0)
    alpha_max_delta = float(row.get("alpha_post_max_delta", 0.0) or 0.0)
    return mse + rec_pen - 2.0 * max(0.0, alpha_delta) - 0.5 * max(0.0, alpha_max_delta)


def update_best_config(out_dir: str, row: Dict[str, Any]) -> None:
    best_path = os.path.join(out_dir, "best_config.json")
    prev: Optional[Dict[str, Any]] = None
    if os.path.exists(best_path):
        with open(best_path, "r") as f:
            prev = json.load(f)
    score = float(row.get("score", composite_score(row)))
    if prev is None or score < float(prev.get("score", 1e18)):
        payload = dict(row)
        payload["score"] = score
        payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
        with open(best_path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)


def build_train_cmd(
    cfg: Dict[str, Any],
    train_ds: str,
    model_id: str,
    stage: int,
    init_setting: str = "",
) -> List[str]:
    meta = DATASET_META[train_ds]
    alpha_mode = "fixed" if stage == 1 and cfg.get("twostage") else "learnable"
    cmd = CONDA_PYTHON + [
        "-u",
        "run_longExp.py",
        "--is_training",
        "1",
        "--random_seed",
        "2021",
        "--root_path",
        "./dataset/",
        "--data_path",
        meta["csv"],
        "--model_id",
        model_id,
        "--model",
        cfg["model"],
        "--data",
        meta["data"],
        "--features",
        "M",
        "--freq",
        meta["freq"],
        "--seq_len",
        "512",
        "--pred_len",
        "720",
        "--enc_in",
        str(meta["enc_in"]),
        "--dec_in",
        str(meta["enc_in"]),
        "--c_out",
        str(meta["enc_in"]),
        "--e_layers",
        "3" if cfg["model"] == "PatchTST" else "1",
        "--n_heads",
        "4",
        "--d_model",
        str(cfg.get("d_model", 16)),
        "--d_ff",
        str(max(128, int(cfg.get("d_model", 16)) * 4)),
        "--dropout",
        "0.2",
        "--fc_dropout",
        "0.2",
        "--patch_len",
        "16",
        "--stride",
        "8",
        "--head_type",
        "sacq_cross_attention",
        "--cross_attn_layers",
        "2",
        "--padding_patch",
        "end",
        "--revin",
        "1",
        "--affine",
        "0",
        "--subtract_last",
        "0",
        "--sacq_overlap_avg",
        "0",
        "--sacq_alpha_mode",
        alpha_mode,
        "--sacq_alpha_value",
        "0.5",
        "--sacq_drift_alpha_enable",
        "1",
        "--sacq_alpha_temperature",
        str(cfg.get("tau", 0.35)),
        "--dsacq_enable",
        "0",
        "--dsacq_mem_gate_enable",
        "0",
        "--dsacq_drift_gate_enable",
        "0",
        "--sacq_gate_loss_weight",
        str(cfg.get("gate", 0.3)),
        "--sacq_corr_sign_loss_weight",
        str(cfg.get("sign", 0.15)),
        "--sacq_shift_boost_loss_weight",
        str(cfg.get("boost", 0.1) if stage >= 2 else 0.0),
        "--batch_size",
        "128",
        "--num_workers",
        "4",
        "--learning_rate",
        "1e-4",
        "--train_epochs",
        "40",
        "--patience",
        "12",
        "--des",
        f"s{stage}",
        "--itr",
        "1",
        "--skip_visual",
        "1",
        "--short_setting",
        "1",
    ]
    if init_setting:
        cmd += ["--init_checkpoint_setting", init_setting]
    return cmd


def train_model(cfg: Dict[str, Any], train_ds: str, gpu: int, log_dir: str) -> str:
    ensure_dir(log_dir)
    train_key = f"{cfg['name']}_{train_ds}"
    twostage = int(cfg.get("twostage", 0))
    meta = DATASET_META[train_ds]
    data_name = meta["data"]
    if twostage:
        s1_id = f"Au{train_key}S1"
        s1_setting = f"{s1_id}_{cfg['model']}_{data_name}_sl512_pl720_s1_0"
        s1_ckpt = os.path.join("checkpoints", s1_setting, "checkpoint.pth")
        s1_log = os.path.join(log_dir, f"train_{train_key}_s1.log")
        if not os.path.exists(s1_ckpt):
            cmd = build_train_cmd(cfg, train_ds, s1_id, stage=1)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            with open(s1_log, "w") as f:
                proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
            if proc.returncode != 0:
                raise RuntimeError(f"stage1 failed: {s1_log}")
        s2_id = f"Au{train_key}S2"
        s2_setting = f"{s2_id}_{cfg['model']}_{data_name}_sl512_pl720_s2_0"
        s2_ckpt = os.path.join("checkpoints", s2_setting, "checkpoint.pth")
        s2_log = os.path.join(log_dir, f"train_{train_key}_s2.log")
        if not os.path.exists(s2_ckpt):
            cmd = build_train_cmd(cfg, train_ds, s2_id, stage=2, init_setting=s1_setting)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            with open(s2_log, "w") as f:
                proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
            if proc.returncode != 0:
                raise RuntimeError(f"stage2 failed: {s2_log}")
        return s2_setting
    model_id = f"Au{train_key}"
    setting = f"{model_id}_{cfg['model']}_{data_name}_sl512_pl720_s1_0"
    ckpt = os.path.join("checkpoints", setting, "checkpoint.pth")
    log_path = os.path.join(log_dir, f"train_{train_key}.log")
    if not os.path.exists(ckpt):
        cmd = build_train_cmd(cfg, train_ds, model_id, stage=1)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        with open(log_path, "w") as f:
            proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            raise RuntimeError(f"train failed: {log_path}")
    return setting


def build_eval_args(
    eval_ds: str,
    shift_csv: str,
    model_cfg: Dict[str, Any],
) -> SimpleNamespace:
    model = model_cfg["model"]
    meta = DATASET_META[eval_ds]
    enc_in = meta["enc_in"]
    freq = meta["freq"]
    d_model = int(model_cfg.get("d_model", 16))
    return SimpleNamespace(
        random_seed=2021,
        seq_len=512,
        label_len=48,
        pred_len=720,
        features="M",
        target="OT",
        embed="timeF",
        freq=freq,
        data=DATASET_META[eval_ds]["data"],
        root_path="./dataset/",
        data_path=shift_csv,
        batch_size=128,
        num_workers=4,
        enc_in=enc_in,
        dec_in=enc_in,
        c_out=enc_in,
        d_model=d_model,
        n_heads=4,
        e_layers=3 if model == "PatchTST" else 1,
        d_layers=1,
        d_ff=max(128, d_model * 4),
        moving_avg=25,
        factor=1,
        distil=True,
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
        sacq_drift_alpha_enable=1,
        sacq_alpha_temperature=float(model_cfg.get("tau", 0.35)),
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
    )


def evaluate_shift(
    cfg: Dict[str, Any],
    ckpt_path: str,
    eval_ds: str,
    shift_preset: str,
    shift_idx: int,
    device: torch.device,
    out_prefix: str,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    shift_csv = shift_csv_name(eval_ds, shift_preset)
    eval_args = build_eval_args(eval_ds, shift_csv, cfg)
    _, loader = data_provider(eval_args, "test")

    mod = importlib.import_module(f"models.{cfg['model']}")
    model = mod.Model(eval_args).float().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    preds, trues, inputx = [], [], []
    err_map: Dict[int, List[float]] = {}
    alpha_map: Dict[int, List[float]] = {}
    sample_base = 0
    test_border1 = 12 * 30 * 24 + 4 * 30 * 24 - eval_args.seq_len
    if eval_ds == "ETTm2":
        test_border1 = 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - eval_args.seq_len
    elif eval_ds == "weather":
        test_border1 = int(len(pd.read_csv(os.path.join("./dataset/", shift_csv))) * 0.7) - eval_args.seq_len

    with torch.no_grad():
        for batch_x, batch_y, _, _ in loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            out = model(batch_x)
            outputs, aux = out if isinstance(out, tuple) else (out, {})
            outputs = outputs[:, -eval_args.pred_len :, :]
            batch_y = batch_y[:, -eval_args.pred_len :, :]

            pred_np = outputs.detach().cpu().numpy()
            true_np = batch_y.detach().cpu().numpy()
            x_np = batch_x.detach().cpu().numpy()
            preds.append(pred_np)
            trues.append(true_np)
            inputx.append(x_np)

            alpha_seq = aux.get("sacq_alpha_seq")
            alpha_np = alpha_seq.detach().cpu().numpy() if torch.is_tensor(alpha_seq) else None
            bsz = pred_np.shape[0]
            for j in range(bsz):
                gi = sample_base + j
                mse_h = ((pred_np[j] - true_np[j]) ** 2).mean(axis=1)
                for h in range(eval_args.pred_len):
                    t_abs = test_border1 + eval_args.seq_len + gi + h
                    err_map.setdefault(t_abs, []).append(float(mse_h[h]))
                    if alpha_np is not None:
                        alpha_map.setdefault(t_abs, []).append(float(alpha_np[j, :, h].mean()))
            sample_base += bsz

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    inputx = np.concatenate(inputx, axis=0)
    mae, mse, rmse, mape, mspe, rse, corr, smape, mase, ql50, pwmae = metric(preds, trues, inputx=inputx)

    t_sorted = sorted(err_map.keys())
    mse_t = np.array([np.mean(err_map[t]) for t in t_sorted], dtype=np.float64)
    alpha_t = np.array(
        [np.mean(alpha_map[t]) if t in alpha_map else np.nan for t in t_sorted],
        dtype=np.float64,
    )
    t_arr = np.array(t_sorted, dtype=np.int64)

    pre_mask = t_arr < shift_idx
    post_mask = t_arr >= shift_idx
    baseline = float(np.mean(mse_t[pre_mask])) if np.any(pre_mask) else float(np.mean(mse_t))
    threshold = baseline * 1.2
    rec = -1
    if np.any(post_mask):
        for idx in np.where(post_mask)[0]:
            if mse_t[idx] < threshold:
                rec = int(t_arr[idx] - shift_idx)
                break

    alpha_pre = float(np.nanmean(alpha_t[pre_mask])) if np.any(pre_mask) else float("nan")
    alpha_post = float(np.nanmean(alpha_t[post_mask])) if np.any(post_mask) else float("nan")
    post300_mask = (t_arr >= shift_idx) & (t_arr < shift_idx + 300)
    alpha_post300 = float(np.nanmean(alpha_t[post300_mask])) if np.any(post300_mask) else float("nan")
    alpha_post300_max = float(np.nanmax(alpha_t[post300_mask])) if np.any(post300_mask) else float("nan")

    per_t_path = out_prefix + "_per_t.csv"
    with open(per_t_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_abs", "mse_t", "alpha_t"])
        for t, m, a in zip(t_arr, mse_t, alpha_t):
            w.writerow([int(t), float(m), "" if np.isnan(a) else float(a)])

    res = {
        "mse": float(mse),
        "mae": float(mae),
        "recovery_steps": rec,
        "baseline_mse_t": baseline,
        "threshold": threshold,
        "alpha_pre_mean": alpha_pre,
        "alpha_post_mean": alpha_post,
        "alpha_post300_mean": alpha_post300,
        "alpha_post300_max": alpha_post300_max,
        "alpha_post_delta": alpha_post - alpha_pre if not (np.isnan(alpha_post) or np.isnan(alpha_pre)) else 0.0,
        "alpha_post_max_delta": alpha_post300_max - alpha_pre if not np.isnan(alpha_pre) else 0.0,
        "per_t_path": per_t_path,
    }
    return res, t_arr, mse_t, alpha_t


def plot_curve(out_path: str, shift_idx: int, t_arr, mse_t, alpha_t, title: str) -> str:
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(t_arr, mse_t, label="MSE_t", linewidth=1.3, color="tab:blue")
    ax1.axvline(shift_idx, color="red", linestyle="--", linewidth=1.1, label=f"Shift @ {shift_idx}")
    ax1.set_xlabel("Absolute time index")
    ax1.set_ylabel("Per-time MSE")
    ax1.grid(alpha=0.2)
    ax2 = ax1.twinx()
    ax2.plot(t_arr, alpha_t, color="purple", alpha=0.65, linewidth=1.1, label="alpha_t")
    ax2.set_ylabel("Gate alpha")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def worker_loop(job_q: queue.Queue, gpu_id: int, out_dir: str, lb_path: str, device_map: Dict[int, str]) -> None:
    device = torch.device(device_map.get(gpu_id, "cuda:0"))
    while True:
        try:
            job = job_q.get(timeout=2)
        except queue.Empty:
            return
        if job is None:
            job_q.task_done()
            return
        rid, cfg, train_ds, eval_ds, shift_preset = job
        row: Dict[str, Any] = {
            "run_id": rid,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "train_dataset": train_ds,
            "eval_dataset": eval_ds,
            "shift_preset": shift_preset,
            "model": cfg["model"],
            "head_type": "sacq_cross_attention",
            "twostage": int(cfg.get("twostage", 0)),
            "sacq_gate_loss_weight": cfg.get("gate"),
            "sacq_corr_sign_loss_weight": cfg.get("sign"),
            "sacq_shift_boost_loss_weight": cfg.get("boost"),
            "sacq_alpha_temperature": cfg.get("tau"),
            "sacq_drift_alpha_enable": 1,
            "status": "running",
        }
        run_tag = f"{cfg['name']}_{train_ds}_{eval_ds}_{shift_preset}"
        log_dir = os.path.join(out_dir, "logs", run_tag)
        art_dir = os.path.join(out_dir, "artifacts", run_tag)
        ensure_dir(log_dir)
        ensure_dir(art_dir)
        try:
            if train_ds != eval_ds and cfg["model"] in ("Linear", "ARIMA", "NBEATS"):
                if DATASET_META[train_ds]["enc_in"] != DATASET_META[eval_ds]["enc_in"]:
                    row["status"] = "skipped_cross_enc"
                    row["score"] = 1e9
                    append_leaderboard(lb_path, row)
                    job_q.task_done()
                    continue
            shift_idx, _, _, shift_info = make_shifted_dataset(eval_ds, shift_preset)
            with open(os.path.join(art_dir, "shift_info.json"), "w") as f:
                json.dump(shift_info, f, indent=2)
            train_log_dir = os.path.join(out_dir, "logs", "train", f"{cfg['name']}_{train_ds}")
            setting = train_model(cfg, train_ds, gpu_id, train_log_dir)
            ckpt_path = os.path.join("checkpoints", setting, "checkpoint.pth")
            row["setting"] = setting
            row["ckpt_path"] = ckpt_path
            row["log_path"] = log_dir
            out_prefix = os.path.join(art_dir, "eval")
            metrics, t_arr, mse_t, alpha_t = evaluate_shift(
                cfg, ckpt_path, eval_ds, shift_preset, shift_idx, device, out_prefix
            )
            row.update(metrics)
            plot_path = plot_curve(
                os.path.join(art_dir, "shift_curve.png"),
                shift_idx,
                t_arr,
                mse_t,
                alpha_t,
                title=f"{cfg['name']} {train_ds}->{eval_ds} [{shift_preset}]",
            )
            row["plot_path"] = plot_path
            row["status"] = "ok"
            row["score"] = composite_score(row)
            append_leaderboard(lb_path, row)
            update_best_config(out_dir, row)
            with open(os.path.join(out_dir, "last_ok.json"), "w") as f:
                json.dump(row, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            row["status"] = f"error:{type(exc).__name__}"
            row["score"] = 1e9
            row["log_path"] = log_dir
            append_leaderboard(lb_path, row)
        finally:
            job_q.task_done()


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous SACQ shift sweep")
    parser.add_argument("--out_dir", default="logs/autonomous_sacq")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--max_jobs", type=int, default=0, help="0 = all pending")
    parser.add_argument("--configs", default="", help="comma names subset of EXPERIMENT_GRID")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    ensure_dir(os.path.join(args.out_dir, "logs"))
    ensure_dir(os.path.join(args.out_dir, "artifacts"))
    lb_path = os.path.join(args.out_dir, "leaderboard.csv")
    done = load_leaderboard(lb_path)

    cfg_list = EXPERIMENT_GRID
    if args.configs.strip():
        names = {x.strip() for x in args.configs.split(",") if x.strip()}
        cfg_list = [c for c in EXPERIMENT_GRID if c["name"] in names]

    jobs: List[Tuple[str, Dict[str, Any], str, str, str]] = []
    seen_rids = set()
    for cfg in cfg_list:
        for train_ds, eval_ds, shift_preset in EVAL_MATRIX:
            if train_ds != eval_ds and cfg["model"] == "PatchTST" and shift_preset not in ("stage_combo", "mild_stage"):
                continue
            if train_ds != eval_ds and train_ds not in ("ETTh1", "ETTm2"):
                continue
            rid = run_id_for(cfg, train_ds, eval_ds, shift_preset)
            if rid in seen_rids:
                continue
            seen_rids.add(rid)
            prev = done.get(rid)
            if prev and prev.get("status") == "ok":
                continue
            jobs.append((rid, cfg, train_ds, eval_ds, shift_preset))

    if args.max_jobs > 0:
        jobs = jobs[: args.max_jobs]

    if not jobs:
        print("No pending jobs.")
        return

    gpu_ids = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    job_q: queue.Queue = queue.Queue()
    for job in jobs:
        job_q.put(job)
    for _ in gpu_ids:
        job_q.put(None)

    device_map = {gid: "cuda:0" for gid in gpu_ids}
    threads = []
    for gid in gpu_ids:
        t = threading.Thread(
            target=worker_loop,
            args=(job_q, gid, args.out_dir, lb_path, device_map),
            daemon=True,
        )
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    print(f"Finished batch. Leaderboard: {lb_path}")


if __name__ == "__main__":
    main()
