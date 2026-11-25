"""ETTh2 pl=192 SARslc: seed sweep (parallel), same config as logcosh_scaled baseline."""
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

DEFAULT_SEEDS = [2022, 2023, 42, 3407]
PRED_LEN = 192


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_seeds(spec: str) -> list[int]:
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def build_cmd(args, seed: int) -> list[str]:
    py = os.environ.get("PYTHON", DEFAULT_PYTHON if os.path.isfile(DEFAULT_PYTHON) else sys.executable)
    tag = f"s{seed}"
    model_id = f"E2s512p{PRED_LEN}SARslc_{tag}"
    des = f"hSARslc_{tag}"
    return [
        py,
        "-u",
        os.path.join(PROJECT_ROOT, "run_longExp.py"),
        "--is_training",
        "1",
        "--random_seed",
        str(seed),
        "--root_path",
        "./dataset/",
        "--data_path",
        "ETTh2.csv",
        "--model_id",
        model_id,
        "--model",
        "PatchTST",
        "--data",
        "ETTh2",
        "--features",
        "M",
        "--freq",
        "h",
        "--seq_len",
        "512",
        "--pred_len",
        str(PRED_LEN),
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
        "--cross_attn_layers",
        "2",
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
        des,
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
        str(args.logcosh_beta),
        "--logcosh_beta_mode",
        args.logcosh_beta_mode,
    ]


def run_one(task: dict, gpu_id: int) -> dict:
    setting = (
        f"{task['model_id']}_PatchTST_ETTh2_sl512_pl{PRED_LEN}_{task['des']}_0"
    )
    metrics_path = os.path.join(PROJECT_ROOT, "results", setting, "metrics.json")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    t0 = time.time()
    with open(task["log_path"], "w") as lf:
        lf.write(f"random_seed={task['seed']}\n")
        lf.write("cmd=" + " ".join(task["cmd"]) + "\n\n")
        lf.flush()
        proc = subprocess.run(
            task["cmd"], cwd=PROJECT_ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT
        )
    row = {
        "task_id": task["task_id"],
        "seed": task["seed"],
        "status": "ok" if proc.returncode == 0 else "fail",
        "return_code": proc.returncode,
        "gpu": gpu_id,
        "elapsed_sec": round(time.time() - t0, 2),
        "setting": setting,
        "log_path": task["log_path"],
    }
    if proc.returncode == 0 and os.path.isfile(metrics_path):
        with open(metrics_path) as f:
            m = json.load(f)
        for k in ("mse", "mae", "rse", "rmse", "corr"):
            row[k] = m.get(k)
    return row


def worker(gpu_id: int, q: Queue, csv_path: str, header: list, lock: threading.Lock) -> None:
    while True:
        try:
            task = q.get_nowait()
        except Empty:
            break
        row = run_one(task, gpu_id)
        with lock:
            exists = os.path.isfile(csv_path)
            with open(csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=header)
                if not exists:
                    w.writeheader()
                w.writerow(row)
            print(
                f"[GPU {gpu_id}] seed={row['seed']} -> {row['status']} "
                f"mse={row.get('mse', 'n/a')} ({row['elapsed_sec']}s)",
                flush=True,
            )
        q.task_done()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="logs/etth2_sar_sl512_pl192_seed_sweep",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="",
        help="comma-separated seeds; default 2022,2023,42,3407",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--logcosh_beta", type=float, default=1.0)
    parser.add_argument("--logcosh_beta_mode", type=str, default="auto_mad")
    parser.add_argument("--e_layers", type=int, default=3)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--fc_dropout", type=float, default=0.3)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds) if args.seeds.strip() else DEFAULT_SEEDS

    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    log_dir = os.path.join(out_dir, "logs")
    ensure_dir(log_dir)
    csv_path = os.path.join(out_dir, "summary.csv")
    header = [
        "task_id", "seed", "status", "return_code", "gpu", "elapsed_sec",
        "setting", "log_path", "mse", "mae", "rse", "rmse", "corr",
    ]

    tasks = []
    for seed in seeds:
        tag = f"s{seed}"
        model_id = f"E2s512p{PRED_LEN}SARslc_{tag}"
        des = f"hSARslc_{tag}"
        tasks.append(
            {
                "task_id": f"pl192_{tag}",
                "seed": seed,
                "des": des,
                "model_id": model_id,
                "cmd": build_cmd(args, seed),
                "log_path": os.path.join(log_dir, f"pl192_{tag}.log"),
            }
        )

    q: Queue = Queue()
    for t in tasks:
        q.put(t)
    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    lock = threading.Lock()
    print(f"total_tasks={len(tasks)} seeds={seeds} gpus={args.gpus}", flush=True)
    threads = [
        threading.Thread(target=worker, args=(gid, q, csv_path, header, lock), daemon=False)
        for gid in gpu_ids
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    print("all tasks finished", flush=True)


if __name__ == "__main__":
    main()
