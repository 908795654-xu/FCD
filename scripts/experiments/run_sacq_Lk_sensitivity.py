#!/usr/bin/env python3
"""ETTh1 + Weather: SAR + logcosh_scaled sensitivity over L (cross_attn_layers) and k (logcosh_beta).

Defaults: pl=720, sl=512, logcosh_beta_mode=auto_mad, sacq_autoregressive=1.
Sweep: L in {1,2,3}, k in {0.5,1,2,4} -> 12 configs x 2 datasets = 24 runs.
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

SEQ_LEN = 512
PRED_LEN = 720
L_VALUES = (1, 2, 3)
K_VALUES = (0.5, 1.0, 2.0, 4.0)

DATASETS = {
    "ETTh1": dict(
        data="ETTh1",
        data_path="ETTh1.csv",
        id_prefix="H1",
        enc_in=7,
        freq="h",
        d_model=16,
        n_heads=4,
        d_ff=128,
        dropout=0.3,
        fc_dropout=0.3,
        epochs=50,
        patience=20,
    ),
    "weather": dict(
        data="weather",
        data_path="weather.csv",
        id_prefix="W",
        enc_in=21,
        freq="h",
        d_model=128,
        n_heads=16,
        d_ff=256,
        dropout=0.2,
        fc_dropout=0.2,
        epochs=30,
        patience=12,
    ),
}


def k_tag(k: float) -> str:
    if k == int(k):
        return f"k{int(k)}"
    s = f"{k:.4f}".rstrip("0").rstrip(".")
    return "k" + s.replace(".", "")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_tasks(
    datasets: list[str],
    l_values: tuple[int, ...],
    k_values: tuple[float, ...],
    pred_len: int,
) -> list[dict]:
    tasks = []
    for ds_name in datasets:
        cfg = DATASETS[ds_name]
        prefix = cfg["id_prefix"]
        for L in l_values:
            for k in k_values:
                kt = k_tag(k)
                model_id = f"{prefix}s{SEQ_LEN}p{pred_len}SAR_L{L}{kt}"
                des = f"hSAR_L{L}{kt}"
                setting = (
                    f"{model_id}_PatchTST_{cfg['data']}_sl{SEQ_LEN}_pl{pred_len}_{des}_0"
                )
                tasks.append(
                    {
                        "task_id": model_id,
                        "model_id": model_id,
                        "dataset": ds_name,
                        "des": des,
                        "setting": setting,
                        "pred_len": pred_len,
                        "cross_attn_layers": L,
                        "logcosh_beta": k,
                        **cfg,
                    }
                )
    return tasks


def checkpoint_path(task: dict) -> str:
    return os.path.join(PROJECT_ROOT, "checkpoints", task["setting"], "checkpoint.pth")


def build_cmd(task: dict, args) -> list[str]:
    py = os.environ.get(
        "PYTHON",
        DEFAULT_PYTHON if os.path.isfile(DEFAULT_PYTHON) else sys.executable,
    )
    cfg = task
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
        cfg["data_path"],
        "--model_id",
        task["model_id"],
        "--model",
        "PatchTST",
        "--data",
        cfg["data"],
        "--features",
        "M",
        "--freq",
        cfg["freq"],
        "--seq_len",
        str(SEQ_LEN),
        "--pred_len",
        str(task["pred_len"]),
        "--enc_in",
        str(cfg["enc_in"]),
        "--dec_in",
        str(cfg["enc_in"]),
        "--c_out",
        str(cfg["enc_in"]),
        "--e_layers",
        str(args.e_layers),
        "--n_heads",
        str(cfg["n_heads"]),
        "--d_model",
        str(cfg["d_model"]),
        "--d_ff",
        str(cfg["d_ff"]),
        "--dropout",
        str(cfg["dropout"]),
        "--fc_dropout",
        str(cfg["fc_dropout"]),
        "--head_dropout",
        "0",
        "--patch_len",
        str(args.patch_len),
        "--stride",
        str(args.stride),
        "--cross_attn_layers",
        str(task["cross_attn_layers"]),
        "--train_epochs",
        str(task["epochs"] if args.epochs <= 0 else args.epochs),
        "--patience",
        str(task["patience"] if args.patience <= 0 else args.patience),
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
        str(task["logcosh_beta"]),
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


def run_one(task: dict, gpu_id: int, args, out_dir: str) -> dict:
    log_path = os.path.join(out_dir, "logs", f"{task['task_id']}.log")
    metrics_path = os.path.join(PROJECT_ROOT, "results", task["setting"], "metrics.json")
    ckpt = checkpoint_path(task)

    if not args.force and os.path.isfile(ckpt):
        row = {
            "task_id": task["task_id"],
            "status": "skipped",
            "return_code": 0,
            "gpu": gpu_id,
            "elapsed_sec": 0.0,
            "dataset": task["dataset"],
            "seq_len": SEQ_LEN,
            "pred_len": task["pred_len"],
            "cross_attn_layers": task["cross_attn_layers"],
            "logcosh_beta": task["logcosh_beta"],
            "logcosh_beta_mode": "auto_mad",
            "loss": "logcosh_scaled",
            "setting": task["setting"],
            "log_path": log_path,
            "checkpoint": ckpt,
        }
        if os.path.isfile(metrics_path):
            with open(metrics_path) as f:
                m = json.load(f)
            for key in ("mse", "mae", "rse", "rmse", "corr"):
                row[key] = m.get(key)
        with open(log_path, "w") as lf:
            lf.write(f"task_id={task['task_id']}\nstatus=skipped (checkpoint exists)\n{ckpt}\n")
        return row

    cmd = build_cmd(task, args)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    t0 = time.time()
    with open(log_path, "w") as lf:
        lf.write(f"task_id={task['task_id']}\n")
        lf.write(f"gpu={gpu_id}\n")
        lf.write(f"L={task['cross_attn_layers']} k={task['logcosh_beta']}\n")
        lf.write(f"setting={task['setting']}\n")
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
        "dataset": task["dataset"],
        "seq_len": SEQ_LEN,
        "pred_len": task["pred_len"],
        "cross_attn_layers": task["cross_attn_layers"],
        "logcosh_beta": task["logcosh_beta"],
        "logcosh_beta_mode": "auto_mad",
        "loss": "logcosh_scaled",
        "setting": task["setting"],
        "log_path": log_path,
        "checkpoint": ckpt,
    }
    if proc.returncode == 0 and os.path.isfile(metrics_path):
        with open(metrics_path) as f:
            m = json.load(f)
        for key in ("mse", "mae", "rse", "rmse", "corr"):
            row[key] = m.get(key)
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
                f"[GPU {gpu_id}] {row['task_id']} L={row['cross_attn_layers']} "
                f"k={row['logcosh_beta']} -> {row['status']} "
                f"mse={row.get('mse', 'n/a')} ({row['elapsed_sec']}s)",
                flush=True,
            )
        q.task_done()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SACQ L x k sensitivity: ETTh1 + Weather, pl=720, SAR+logcosh_scaled"
    )
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument(
        "--datasets",
        type=str,
        default="ETTh1,weather",
        help="comma-separated: ETTh1, weather",
    )
    parser.add_argument("--out_dir", type=str, default="logs/sacq_Lk_sensitivity")
    parser.add_argument("--pred_len", type=int, default=PRED_LEN)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="override train_epochs for all datasets (0 = per-dataset default)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="override patience for all datasets (0 = per-dataset default)",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--e_layers", type=int, default=3)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="re-train even if checkpoint exists")
    parser.add_argument(
        "--L",
        type=str,
        default="1,2,3",
        help="cross_attn_layers values, comma-separated",
    )
    parser.add_argument(
        "--k",
        type=str,
        default="0.5,1,2,4",
        help="logcosh_beta multipliers (auto_mad), comma-separated",
    )
    args = parser.parse_args()

    ds_names = [x.strip() for x in args.datasets.split(",") if x.strip()]
    unknown = [d for d in ds_names if d not in DATASETS]
    if unknown:
        raise SystemExit(f"Unknown datasets: {unknown}. Choose from {list(DATASETS)}")

    l_values = tuple(int(x.strip()) for x in args.L.split(",") if x.strip())
    k_values = tuple(float(x.strip()) for x in args.k.split(",") if x.strip())

    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "logs"))
    csv_path = os.path.join(out_dir, "summary.csv")

    if args.force and os.path.isfile(csv_path):
        os.remove(csv_path)

    tasks = build_tasks(ds_names, l_values, k_values, args.pred_len)
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
        "log_path",
        "checkpoint",
        "mse",
        "mae",
        "rse",
        "rmse",
        "corr",
    ]

    print(
        f"total_tasks={len(tasks)} datasets={ds_names} L={l_values} k={k_values} "
        f"pl={args.pred_len} gpus={args.gpus}",
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
