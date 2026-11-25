#!/usr/bin/env python3
"""ETTh1 + Weather: Flatten+MSE vs SACQ+ScaledLogCosh with gradient blocking probe."""
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

DATASETS = [
    {
        "data": "ETTh1",
        "data_path": "ETTh1.csv",
        "enc_in": 7,
        "freq": "h",
        "d_model": 16,
        "n_heads": 4,
        "d_ff": 128,
        "dropout": 0.3,
        "fc_dropout": 0.3,
        "batch_size": 128,
        "lr": 1e-4,
        "tag": "H1",
    },
    {
        "data": "weather",
        "data_path": "weather.csv",
        "enc_in": 21,
        "freq": "h",
        "d_model": 128,
        "n_heads": 16,
        "d_ff": 256,
        "dropout": 0.2,
        "fc_dropout": 0.2,
        "batch_size": 128,
        "lr": 1e-4,
        "tag": "W",
    },
]

VARIANTS = [
    {
        "name": "Flatten+MSE",
        "suffix": "Flatmse",
        "head_type": "flatten",
        "loss": "mse",
        "sacq_ar": 0,
        "des": "hFlatmse_probe",
    },
    {
        "name": "SACQ+ScaledLogCosh",
        "suffix": "SACQslc",
        "head_type": "sacq_cross_attention",
        "loss": "logcosh_scaled",
        "sacq_ar": 0,
        "des": "hSACQslc_probe",
    },
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_row(path: str, header: list[str], row: dict) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def build_tasks(pred_len: int) -> list[dict]:
    tasks = []
    for ds in DATASETS:
        for v in VARIANTS:
            model_id = f"{ds['tag']}s{SEQ_LEN}p{pred_len}{v['suffix']}gprobe"
            setting = (
                f"{model_id}_PatchTST_{ds['data']}_"
                f"sl{SEQ_LEN}_pl{pred_len}_{v['des']}_0"
            )
            tasks.append(
                {
                    "task_id": model_id,
                    "model_id": model_id,
                    "setting": setting,
                    "variant": v["name"],
                    "head_type": v["head_type"],
                    "loss": v["loss"],
                    "sacq_ar": v["sacq_ar"],
                    "des": v["des"],
                    **ds,
                }
            )
    return tasks


def run_one(task: dict, args, gpu_id: int, out_dir: str) -> dict:
    log_path = os.path.join(out_dir, "logs", f"{task['task_id']}.log")
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
        str(args.pred_len),
        "--enc_in",
        str(task["enc_in"]),
        "--dec_in",
        str(task["enc_in"]),
        "--c_out",
        str(task["enc_in"]),
        "--e_layers",
        "3",
        "--n_heads",
        str(task["n_heads"]),
        "--d_model",
        str(task["d_model"]),
        "--d_ff",
        str(task["d_ff"]),
        "--dropout",
        str(task["dropout"]),
        "--fc_dropout",
        str(task["fc_dropout"]),
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
        str(task["lr"]),
        "--lradj",
        "TST",
        "--pct_start",
        "0.2",
        "--itr",
        "1",
        "--num_workers",
        str(args.num_workers),
        "--head_type",
        task["head_type"],
        "--sacq_autoregressive",
        str(task["sacq_ar"]),
        "--sacq_scheme",
        "default",
        "--sacq_pe_mode",
        "default",
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
        task["loss"],
        "--logcosh_beta",
        "1.0",
        "--logcosh_beta_mode",
        "auto_mad",
        "--paired_noise_enable",
        "1",
        "--paired_noise_sigma",
        str(args.noise_sigma),
        "--grad_probe_enable",
        "1",
        "--grad_probe_epoch",
        str(args.probe_epoch),
        "--grad_probe_max_batches",
        str(args.probe_max_batches),
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

    metrics_path = os.path.join(PROJECT_ROOT, "results", task["setting"], "metrics.json")
    probe_path = os.path.join(PROJECT_ROOT, "results", task["setting"], "grad_probe.json")
    row = {
        "task_id": task["task_id"],
        "dataset": task["data"],
        "variant": task["variant"],
        "head_type": task["head_type"],
        "loss": task["loss"],
        "status": "ok" if proc.returncode == 0 else "fail",
        "return_code": proc.returncode,
        "gpu": gpu_id,
        "elapsed_sec": round(elapsed, 2),
        "setting": task["setting"],
        "log_path": log_path,
        "grad_probe_path": probe_path if os.path.exists(probe_path) else "",
    }
    if proc.returncode == 0 and os.path.exists(metrics_path):
        with open(metrics_path) as f:
            met = json.load(f)
        row["mse"] = met.get("mse")
        row["mae"] = met.get("mae")
    if proc.returncode == 0 and os.path.exists(probe_path):
        with open(probe_path) as f:
            probe = json.load(f)
        s = probe.get("summary", {})
        row["probe_epoch"] = s.get("probe_epoch")
        row["backbone_noise_to_total_ratio_mean"] = s.get("backbone_noise_to_total_ratio_mean")
        row["gradient_blocking_rate_mean"] = s.get("gradient_blocking_rate_mean")
        row["noisy_backbone_norm_ratio_mean"] = s.get("noisy_backbone_norm_ratio_mean")
        row["noisy_sacq_alpha_norm_ratio_mean"] = s.get("noisy_sacq_alpha_norm_ratio_mean")
        row["noisy_sacq_cross_norm_ratio_mean"] = s.get("noisy_sacq_cross_norm_ratio_mean")
    return row


def worker(gpu_id: int, q: Queue, args, out_dir: str, csv_path: str, header: list[str], lock: threading.Lock):
    while True:
        try:
            task = q.get_nowait()
        except Empty:
            break
        row = run_one(task, args, gpu_id, out_dir)
        with lock:
            write_row(csv_path, header, row)
            print(
                f"[GPU {gpu_id}] {row['task_id']} -> {row['status']} "
                f"block={row.get('gradient_blocking_rate_mean', 'n/a')} "
                f"mse={row.get('mse', 'n/a')} ({row['elapsed_sec']}s)",
                flush=True,
            )
        q.task_done()


def main():
    parser = argparse.ArgumentParser(
        description="ETTh1/Weather paired-noise gradient blocking probe (Flatten+MSE vs SACQ+SLC)"
    )
    parser.add_argument("--gpus", type=str, default="0,1")
    parser.add_argument("--out_dir", type=str, default="logs/etth1_weather_grad_block_probe")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--pred_len", type=int, default=720)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--noise_sigma", type=float, default=7.0)
    parser.add_argument("--probe_epoch", type=int, default=5)
    parser.add_argument("--probe_max_batches", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "logs"))
    csv_path = os.path.join(out_dir, "summary.csv")

    tasks = build_tasks(args.pred_len)
    if args.skip_existing:
        kept = []
        for task in tasks:
            probe_path = os.path.join(PROJECT_ROOT, "results", task["setting"], "grad_probe.json")
            if os.path.exists(probe_path):
                print(f"skip existing: {task['task_id']}", flush=True)
            else:
                kept.append(task)
        tasks = kept

    header = [
        "task_id",
        "dataset",
        "variant",
        "head_type",
        "loss",
        "status",
        "return_code",
        "gpu",
        "elapsed_sec",
        "setting",
        "log_path",
        "grad_probe_path",
        "probe_epoch",
        "backbone_noise_to_total_ratio_mean",
        "gradient_blocking_rate_mean",
        "noisy_backbone_norm_ratio_mean",
        "noisy_sacq_alpha_norm_ratio_mean",
        "noisy_sacq_cross_norm_ratio_mean",
        "mse",
        "mae",
    ]
    print(
        f"total_tasks={len(tasks)} gpus={args.gpus} pred_len={args.pred_len} "
        f"epochs={args.epochs} probe_epoch={args.probe_epoch}",
        flush=True,
    )
    q = Queue()
    for t in tasks:
        q.put(t)
    lock = threading.Lock()
    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    threads = [
        threading.Thread(
            target=worker, args=(gid, q, args, out_dir, csv_path, header, lock), daemon=False
        )
        for gid in gpu_ids
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
