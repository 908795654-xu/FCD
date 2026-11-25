#!/usr/bin/env python3
"""Traffic + Electricity: PatchTST sl512 × pl∈{96,192,336,720}, SAR + logcosh_scaled (auto beta).

Hyperparams match scripts/PatchTST/traffic.sh & electricity.sh; head/loss per user request.
4-GPU queue: one training job per GPU at a time.
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
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SEQ_LEN = 512
PRED_LENS = [96, 192, 336, 720]

DATASETS = [
    {
        "data": "traffic",
        "data_path": "traffic.csv",
        "enc_in": 862,
        "freq": "h",
        "model_id_prefix": "traffic",
        "des": "tSARslc",
        "out_subdir": "traffic_sar_sl512_lc_scaled",
        "batch_size": 8,
    },
    {
        "data": "electricity",
        "data_path": "electricity.csv",
        "enc_in": 321,
        "freq": "h",
        "model_id_prefix": "Electricity",
        "des": "eSARslc",
        "out_subdir": "electricity_sar_sl512_lc_scaled",
        "batch_size": 8,
    },
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_tasks(datasets: list[dict]) -> list[dict]:
    tasks = []
    for ds in datasets:
        for pred_len in PRED_LENS:
            model_id = f"{ds['model_id_prefix']}_{SEQ_LEN}_{pred_len}_SARslc"
            tasks.append(
                {
                    "task_id": model_id,
                    "model_id": model_id,
                    "data": ds["data"],
                    "data_path": ds["data_path"],
                    "enc_in": ds["enc_in"],
                    "freq": ds["freq"],
                    "pred_len": pred_len,
                    "des": ds["des"],
                    "out_subdir": ds["out_subdir"],
                    "batch_size": ds["batch_size"],
                    "setting": (
                        f"{model_id}_PatchTST_{ds['data']}_sl{SEQ_LEN}_pl{pred_len}_{ds['des']}_0"
                    ),
                }
            )
    return tasks


def write_csv_row(path: str, header: list, row: dict) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_one(task: dict, gpu_id: int, args) -> dict:
    out_dir = os.path.join(PROJECT_ROOT, "logs", task["out_subdir"])
    ensure_dir(os.path.join(out_dir, "logs"))
    log_path = os.path.join(out_dir, "logs", f"{task['task_id']}.log")
    metrics_path = os.path.join(PROJECT_ROOT, "results", task["setting"], "metrics.json")
    py = os.environ.get("PYTHON", sys.executable)
    cmd = [
        py,
        "-u",
        os.path.join(PROJECT_ROOT, "run_longExp.py"),
        "--is_training",
        "1",
        "--random_seed",
        str(args.seed),
        "--root_path",
        os.path.join(PROJECT_ROOT, "dataset/"),
        "--data_path",
        task["data_path"],
        "--model_id",
        task["model_id"],
        "--model",
        "PatchTST",
        "--data",
        task["data"],
        "--features",
        "M",
        "--freq",
        task["freq"],
        "--seq_len",
        str(SEQ_LEN),
        "--pred_len",
        str(task["pred_len"]),
        "--enc_in",
        str(task["enc_in"]),
        "--dec_in",
        str(task["enc_in"]),
        "--c_out",
        str(task["enc_in"]),
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
        "2",
        "--train_epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch_size",
        str(task["batch_size"]),
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
        "1.0",
        "--logcosh_beta_mode",
        "auto_mad",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    t0 = time.time()
    with open(log_path, "w") as lf:
        lf.write(f"task_id={task['task_id']}\n")
        lf.write(f"gpu={gpu_id}\n")
        lf.write(f"setting={task['setting']}\n")
        lf.write("cmd=" + " ".join(cmd) + "\n\n")
        lf.flush()
        proc = subprocess.run(
            cmd, cwd=PROJECT_ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT, check=False
        )
    elapsed = time.time() - t0

    row = {
        "task_id": task["task_id"],
        "status": "ok" if proc.returncode == 0 else "fail",
        "return_code": proc.returncode,
        "gpu": gpu_id,
        "elapsed_sec": round(elapsed, 2),
        "dataset": task["data"],
        "seq_len": SEQ_LEN,
        "pred_len": task["pred_len"],
        "des": task["des"],
        "head_type": "sacq_cross_attention",
        "sacq_autoregressive": 1,
        "loss": "logcosh_scaled",
        "batch_size": task["batch_size"],
        "setting": task["setting"],
        "log_path": log_path,
    }
    if proc.returncode == 0 and os.path.exists(metrics_path):
        with open(metrics_path) as f:
            m = json.load(f)
        for k in (
            "mse", "mae", "rse", "rmse", "mape", "mspe", "corr",
            "smape", "mase", "ql50", "pwmae",
        ):
            row[k] = m.get(k)
    return row


def worker(gpu_id: int, q: Queue, args, csv_path: str, header: list, lock: threading.Lock) -> None:
    while True:
        try:
            task = q.get_nowait()
        except Empty:
            break
        row = run_one(task, gpu_id, args)
        with lock:
            write_csv_row(csv_path, header, row)
            per_ds = os.path.join(PROJECT_ROOT, "logs", task["out_subdir"], "summary.csv")
            write_csv_row(per_ds, header, row)
            print(
                f"[GPU {gpu_id}] {row['task_id']} -> {row['status']} "
                f"mse={row.get('mse', 'n/a')} ({row['elapsed_sec']}s)",
                flush=True,
            )
        q.task_done()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Traffic+Electricity SAR+logcosh_scaled (PatchTST.sh hyperparams)"
    )
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument(
        "--datasets",
        type=str,
        default="traffic,electricity",
        help="comma-separated: traffic, electricity",
    )
    parser.add_argument("--out_dir", type=str, default="logs/traffic_electricity_sar_lc_scaled")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--e_layers", type=int, default=3)
    parser.add_argument("--n_heads", type=int, default=16)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--fc_dropout", type=float, default=0.2)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    args = parser.parse_args()

    names = {x.strip() for x in args.datasets.split(",") if x.strip()}
    datasets = [d for d in DATASETS if d["data"] in names]
    if not datasets:
        raise SystemExit(f"No datasets matched: {args.datasets}")

    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    ensure_dir(out_dir)
    for ds in datasets:
        ensure_dir(os.path.join(PROJECT_ROOT, "logs", ds["out_subdir"], "logs"))

    csv_path = os.path.join(out_dir, "summary.csv")
    tasks = build_tasks(datasets)
    q: Queue = Queue()
    for t in tasks:
        q.put(t)

    header = [
        "task_id", "status", "return_code", "gpu", "elapsed_sec",
        "dataset", "seq_len", "pred_len", "des", "head_type",
        "sacq_autoregressive", "loss", "batch_size", "setting", "log_path",
        "mse", "mae", "rse", "rmse", "mape", "mspe", "corr",
        "smape", "mase", "ql50", "pwmae",
    ]
    print(f"total_tasks={len(tasks)} gpus={args.gpus}", flush=True)
    lock = threading.Lock()
    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    threads = [
        threading.Thread(target=worker, args=(gid, q, args, csv_path, header, lock), daemon=False)
        for gid in gpu_ids
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
