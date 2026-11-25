#!/usr/bin/env python3
"""ETTh1 pl=720: seq_len x cross_attn_layers grid (SAR + logcosh_scaled).

Protocol aligned with logs/etth1_sar_sl512_logcosh_scaled/logs/H1s512p720SARslc.log:
  SAR, sacq_autoregressive=1, logcosh_scaled, logcosh_beta=1.0, auto_mad,
  d_model=16, n_heads=4, epochs=50, patience=20, short_setting=1.

Grid: sl in {96,336,720,1440,1920} x L in {1,2,3} -> 15 runs.
Reuses equivalent checkpoints from prior seqlen / Lk sweeps when present.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from queue import Empty, Queue

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_PYTHON = "/home/bkai/miniconda3/envs/forecast/bin/python"

DATA = "ETTh1"
DATA_PATH = "ETTh1.csv"
ENC_IN = 7
FREQ = "h"
PRED_LEN = 720
SEQ_LENS = (96, 336, 720, 1440, 1920)
L_VALUES = (1, 2, 3)
LOGCOSH_BETA = 1.0


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def task_model_id(seq_len: int, L: int) -> str:
    return f"H1s{seq_len}p{PRED_LEN}SAR_L{L}slc"


def task_des(L: int) -> str:
    return f"hSAR_L{L}slc"


def task_setting(seq_len: int, L: int) -> str:
    mid = task_model_id(seq_len, L)
    return f"{mid}_PatchTST_{DATA}_sl{seq_len}_pl{PRED_LEN}_{task_des(L)}_0"


def equiv_settings(seq_len: int, L: int) -> list[str]:
    """Prior runs with identical hyperparameters but different model_id."""
    out: list[str] = []
    if L == 2:
        out.append(f"H1s{seq_len}p{PRED_LEN}SARslc_PatchTST_{DATA}_sl{seq_len}_pl{PRED_LEN}_hSARslc_0")
    if seq_len == 512:
        out.append(
            f"H1s512p{PRED_LEN}SAR_L{L}k1_PatchTST_{DATA}_sl512_pl{PRED_LEN}_hSAR_L{L}k1_0"
        )
    return out


def checkpoint_for_setting(setting: str) -> str:
    return os.path.join(PROJECT_ROOT, "checkpoints", setting, "checkpoint.pth")


def metrics_for_setting(setting: str) -> str:
    return os.path.join(PROJECT_ROOT, "results", setting, "metrics.json")


def find_reusable_setting(seq_len: int, L: int, force: bool) -> tuple[str | None, str]:
    """Return (setting, source) where source is 'own' | 'equiv' | ''."""
    if force:
        return None, ""
    own = task_setting(seq_len, L)
    if os.path.isfile(checkpoint_for_setting(own)):
        return own, "own"
    for setting in equiv_settings(seq_len, L):
        if os.path.isfile(checkpoint_for_setting(setting)):
            return setting, "equiv"
    return None, ""


def build_tasks(l_values: tuple[int, ...], seq_lens: tuple[int, ...]) -> list[dict]:
    tasks = []
    for sl in seq_lens:
        for L in l_values:
            tasks.append(
                {
                    "task_id": task_model_id(sl, L),
                    "model_id": task_model_id(sl, L),
                    "seq_len": sl,
                    "pred_len": PRED_LEN,
                    "cross_attn_layers": L,
                    "des": task_des(L),
                    "setting": task_setting(sl, L),
                }
            )
    return tasks


def build_cmd(task: dict, args) -> list[str]:
    py = os.environ.get(
        "PYTHON",
        DEFAULT_PYTHON if os.path.isfile(DEFAULT_PYTHON) else sys.executable,
    )
    return [
        py,
        "-u",
        os.path.join(PROJECT_ROOT, "run_longExp.py"),
        "--is_training",
        "1",
        "--random_seed",
        str(args.seed),
        "--root_path",
        "./dataset/",
        "--data_path",
        DATA_PATH,
        "--model_id",
        task["model_id"],
        "--model",
        "PatchTST",
        "--data",
        DATA,
        "--features",
        "M",
        "--freq",
        FREQ,
        "--seq_len",
        str(task["seq_len"]),
        "--pred_len",
        str(task["pred_len"]),
        "--enc_in",
        str(ENC_IN),
        "--dec_in",
        str(ENC_IN),
        "--c_out",
        str(ENC_IN),
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
        "--cross_attn_layers",
        str(task["cross_attn_layers"]),
        "--train_epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch_size",
        str(args.batch_size),
        "--learning_rate",
        str(args.lr),
        "--lradj",
        "TST",
        "--pct_start",
        "0.2",
        "--itr",
        "1",
        "--num_workers",
        str(args.num_workers),
        "--head_type",
        "sacq_cross_attention",
        "--sacq_autoregressive",
        "1",
        "--sacq_alpha_mode",
        "learnable",
        "--short_setting",
        "1",
        "--des",
        task["des"],
        "--skip_visual",
        "1",
        "--dsacq_enable",
        "0",
        "--dsacq_mem_gate_enable",
        "0",
        "--dsacq_drift_gate_enable",
        "0",
        "--loss",
        "logcosh_scaled",
        "--logcosh_beta",
        str(LOGCOSH_BETA),
        "--logcosh_beta_mode",
        "auto_mad",
    ]


def write_csv_row(path: str, header: list, row: dict) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_metrics(setting: str) -> dict:
    path = metrics_for_setting(setting)
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def run_one(task: dict, gpu_id: int, args, out_dir: str) -> dict:
    log_path = os.path.join(out_dir, "logs", f"{task['task_id']}.log")
    ckpt = checkpoint_for_setting(task["setting"])
    reuse_setting, reuse_src = find_reusable_setting(
        task["seq_len"], task["cross_attn_layers"], args.force
    )

    if reuse_setting is not None:
        metrics = load_metrics(reuse_setting)
        status = "reused" if reuse_src == "equiv" else "skipped"
        row = {
            "task_id": task["task_id"],
            "status": status,
            "return_code": 0,
            "gpu": gpu_id,
            "elapsed_sec": 0.0,
            "dataset": DATA,
            "seq_len": task["seq_len"],
            "pred_len": task["pred_len"],
            "cross_attn_layers": task["cross_attn_layers"],
            "logcosh_beta": LOGCOSH_BETA,
            "logcosh_beta_mode": "auto_mad",
            "loss": "logcosh_scaled",
            "setting": task["setting"],
            "reuse_setting": reuse_setting,
            "reuse_source": reuse_src,
            "log_path": log_path,
            "checkpoint": checkpoint_for_setting(reuse_setting),
        }
        for key in ("mse", "mae", "rse", "rmse", "corr", "mape", "mspe", "smape", "mase", "ql50", "pwmae"):
            row[key] = metrics.get(key)
        with open(log_path, "w") as lf:
            lf.write(f"task_id={task['task_id']}\n")
            lf.write(f"status={status}\n")
            lf.write(f"reuse_setting={reuse_setting}\n")
            lf.write(f"reuse_source={reuse_src}\n")
            lf.write(f"checkpoint={row['checkpoint']}\n")
            if metrics:
                lf.write(f"metrics={json.dumps(metrics)}\n")
        return row

    cmd = build_cmd(task, args)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    t0 = time.time()
    with open(log_path, "w") as lf:
        lf.write(f"task_id={task['task_id']}\n")
        lf.write(f"gpu={gpu_id}\n")
        lf.write(f"setting={task['setting']}\n")
        lf.write(
            "ref=logs/etth1_sar_sl512_logcosh_scaled/logs/H1s512p720SARslc.log\n"
        )
        lf.write("cmd=" + " ".join(cmd) + "\n\n")
        lf.flush()
        proc = subprocess.run(
            cmd, cwd=PROJECT_ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT
        )
    elapsed = time.time() - t0

    row = {
        "task_id": task["task_id"],
        "status": "ok" if proc.returncode == 0 else "fail",
        "return_code": proc.returncode,
        "gpu": gpu_id,
        "elapsed_sec": round(elapsed, 2),
        "dataset": DATA,
        "seq_len": task["seq_len"],
        "pred_len": task["pred_len"],
        "cross_attn_layers": task["cross_attn_layers"],
        "logcosh_beta": LOGCOSH_BETA,
        "logcosh_beta_mode": "auto_mad",
        "loss": "logcosh_scaled",
        "setting": task["setting"],
        "reuse_setting": "",
        "reuse_source": "",
        "log_path": log_path,
        "checkpoint": ckpt,
    }
    if proc.returncode == 0:
        metrics = load_metrics(task["setting"])
        for key in ("mse", "mae", "rse", "rmse", "corr", "mape", "mspe", "smape", "mase", "ql50", "pwmae"):
            row[key] = metrics.get(key)
    return row


def worker(
    gpu_id: int,
    q: Queue,
    args,
    out_dir: str,
    csv_path: str,
    header: list,
    lock: threading.Lock,
) -> None:
    while True:
        try:
            task = q.get_nowait()
        except Empty:
            break
        row = run_one(task, gpu_id, args, out_dir)
        with lock:
            write_csv_row(csv_path, header, row)
            print(
                f"[GPU {gpu_id}] {row['task_id']} sl={row['seq_len']} L={row['cross_attn_layers']} "
                f"-> {row['status']} mse={row.get('mse', 'n/a')} ({row['elapsed_sec']}s)",
                flush=True,
            )
        q.task_done()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ETTh1 pl720: seq_len x SACQ layers grid (SAR+logcosh_scaled)"
    )
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--out_dir", type=str, default="logs/etth1_sl_L_grid_pl720")
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
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--fc_dropout", type=float, default=0.3)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="re-train even if checkpoint exists")
    parser.add_argument(
        "--seq_lens",
        type=str,
        default="96,336,720,1440,1920",
        help="comma-separated seq_len values",
    )
    parser.add_argument(
        "--L",
        type=str,
        default="1,2,3",
        help="cross_attn_layers values, comma-separated",
    )
    args = parser.parse_args()

    seq_lens = tuple(int(x.strip()) for x in args.seq_lens.split(",") if x.strip())
    l_values = tuple(int(x.strip()) for x in args.L.split(",") if x.strip())

    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "logs"))
    csv_path = os.path.join(out_dir, "summary.csv")

    if args.force and os.path.isfile(csv_path):
        os.remove(csv_path)

    tasks = build_tasks(l_values, seq_lens)
    q: Queue = Queue()
    for t in tasks:
        q.put(t)

    header = [
        "task_id",
        "status",
        "return_code",
        "gpu",
        "elapsed_sec",
        "dataset",
        "seq_len",
        "pred_len",
        "cross_attn_layers",
        "logcosh_beta",
        "logcosh_beta_mode",
        "loss",
        "setting",
        "reuse_setting",
        "reuse_source",
        "log_path",
        "checkpoint",
        "mse",
        "mae",
        "rse",
        "rmse",
        "corr",
        "mape",
        "mspe",
        "smape",
        "mase",
        "ql50",
        "pwmae",
    ]

    print(
        f"total_tasks={len(tasks)} sl={seq_lens} L={l_values} pl={PRED_LEN} gpus={args.gpus}",
        flush=True,
    )

    lock = threading.Lock()
    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    threads = [
        threading.Thread(
            target=worker,
            args=(gid, q, args, out_dir, csv_path, header, lock),
            daemon=False,
        )
        for gid in gpu_ids
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    print(f"done. summary -> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
