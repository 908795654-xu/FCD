"""Weather pl=720: MSE (flatten, weather.sh) vs SAR+logcosh_scaled on 2 GPUs."""
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

# Hyperparams from scripts/PatchTST/weather.sh (pl=720 only)
WEATHER_SH_BASE = [
    "--random_seed", "2021",
    "--is_training", "1",
    "--root_path", "./dataset/",
    "--data_path", "weather.csv",
    "--model", "PatchTST",
    "--data", "weather",
    "--features", "M",
    "--seq_len", "512",
    "--pred_len", "720",
    "--enc_in", "21",
    "--e_layers", "3",
    "--n_heads", "16",
    "--d_model", "128",
    "--d_ff", "256",
    "--dropout", "0.2",
    "--fc_dropout", "0.2",
    "--head_dropout", "0",
    "--patch_len", "16",
    "--stride", "8",
    "--des", "Exp",
    "--train_epochs", "30",
    "--patience", "12",
    "--itr", "1",
    "--batch_size", "128",
    "--learning_rate", "0.0001",
]


def build_tasks() -> list[dict]:
    return [
        {
            "task_id": "weather_512_720_mse",
            "model_id": "weather_512_720",
            "variant": "mse_flatten",
            "extra": ["--head_type", "flatten", "--loss", "mse"],
        },
        {
            "task_id": "weather_512_720_sar",
            "model_id": "weather_512_720_SAR",
            "variant": "sar_logcosh_scaled",
            "extra": [
                "--head_type", "sacq_cross_attention",
                "--cross_attn_layers", "2",
                "--sacq_autoregressive", "1",
                "--loss", "logcosh_scaled",
                "--logcosh_beta", "1.0",
                "--logcosh_beta_mode", "auto_mad",
            ],
        },
    ]


def setting_name(model_id: str) -> str:
    return (
        f"{model_id}_PatchTST_weather_ftM_sl512_ll48_pl720_"
        f"dm128_nh16_el3_dl1_df256_fc1_ebtimeF_dtFalse_Exp_0"
    )


def run_one(task: dict, gpu_id: int, py: str, out_dir: str) -> dict:
    setting = setting_name(task["model_id"])
    log_path = os.path.join(out_dir, "logs", f"{task['task_id']}.log")
    metrics_path = os.path.join(PROJECT_ROOT, "results", setting, "metrics.json")

    cmd = [
        py, "-u", os.path.join(PROJECT_ROOT, "run_longExp.py"),
        *WEATHER_SH_BASE,
        "--model_id", task["model_id"],
        *task["extra"],
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    t0 = time.time()
    with open(log_path, "w") as lf:
        lf.write(f"task_id={task['task_id']}\nvariant={task['variant']}\n")
        lf.write(f"gpu={gpu_id}\nsetting={setting}\n")
        lf.write("cmd=" + " ".join(cmd) + "\n\n")
        lf.flush()
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0

    row = {
        "task_id": task["task_id"],
        "variant": task["variant"],
        "model_id": task["model_id"],
        "status": "ok" if proc.returncode == 0 else "fail",
        "return_code": proc.returncode,
        "gpu": gpu_id,
        "elapsed_sec": round(elapsed, 2),
        "setting": setting,
        "log_path": log_path,
    }
    if proc.returncode == 0 and os.path.isfile(metrics_path):
        with open(metrics_path) as f:
            m = json.load(f)
        for k in ("mse", "mae", "rse", "rmse", "corr"):
            row[k] = m.get(k)
    return row


def worker(gpu_id: int, q: Queue, py: str, out_dir: str, csv_path: str, header: list, lock: threading.Lock):
    while True:
        try:
            task = q.get_nowait()
        except Empty:
            break
        row = run_one(task, gpu_id, py, out_dir)
        with lock:
            with open(csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=header)
                if f.tell() == 0:
                    w.writeheader()
                w.writerow(row)
            print(
                f"[GPU {gpu_id}] {row['task_id']} -> {row['status']} "
                f"mse={row.get('mse', 'n/a')} ({row['elapsed_sec']}s)",
                flush=True,
            )
        q.task_done()


def main():
    parser = argparse.ArgumentParser(description="Weather pl720: MSE vs SAR (weather.sh hyperparams)")
    parser.add_argument("--gpus", type=str, default="0,1")
    parser.add_argument("--out_dir", type=str, default="logs/weather_pl720_mse_vs_sar")
    args = parser.parse_args()

    py = os.environ.get("PYTHON", DEFAULT_PYTHON if os.path.isfile(DEFAULT_PYTHON) else sys.executable)
    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    csv_path = os.path.join(out_dir, "summary.csv")
    if os.path.isfile(csv_path):
        os.remove(csv_path)

    tasks = build_tasks()
    q: Queue = Queue()
    for t in tasks:
        q.put(t)

    header = [
        "task_id", "variant", "model_id", "status", "return_code", "gpu",
        "elapsed_sec", "setting", "log_path", "mse", "mae", "rse", "rmse", "corr",
    ]
    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    lock = threading.Lock()
    threads = [
        threading.Thread(target=worker, args=(gid, q, py, out_dir, csv_path, header, lock), daemon=False)
        for gid in gpu_ids[: len(tasks)]
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    print("all tasks finished", flush=True)


if __name__ == "__main__":
    main()
