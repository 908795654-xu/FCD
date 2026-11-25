#!/usr/bin/env python3
"""Scan Weather gradient-firewall aux weights for MSE vs blocking trade-off."""
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


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_row(path: str, header: list[str], row: dict) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parse_weight_pairs(text: str) -> list[tuple[float, float]]:
    pairs = []
    for token in text.split(","):
        t = token.strip()
        if not t:
            continue
        if ":" not in t:
            raise ValueError(f"invalid pair {t!r}, expected sup:sparse")
        sup_s, sparse_s = t.split(":", 1)
        pairs.append((float(sup_s), float(sparse_s)))
    if not pairs:
        raise ValueError("empty weight pairs")
    return pairs


def fmt_weight(x: float) -> str:
    return str(x).replace(".", "p")


def build_tasks(pairs: list[tuple[float, float]], seed: int, pred_len: int) -> list[dict]:
    tasks = []
    for sup_w, sparse_w in pairs:
        sup_tag = fmt_weight(sup_w)
        sparse_tag = fmt_weight(sparse_w)
        model_id = f"Ws512p{pred_len}SACQslcGFWsw{sup_tag}sp{sparse_tag}sd{seed}"
        des = f"hSACQslcGFW_scan_sw{sup_tag}_sp{sparse_tag}_sd{seed}"
        setting = f"{model_id}_PatchTST_weather_sl512_pl{pred_len}_{des}_0"
        tasks.append(
            {
                "task_id": model_id,
                "setting": setting,
                "sup_w": sup_w,
                "sparse_w": sparse_w,
                "seed": seed,
                "des": des,
            }
        )
    return tasks


def run_one(task: dict, args, gpu_id: int, out_dir: str) -> dict:
    py = os.environ.get("PYTHON", "/home/bkai/miniconda3/envs/forecast/bin/python")
    log_path = os.path.join(out_dir, "logs", f"{task['task_id']}.log")
    cmd = [
        py,
        "-u",
        os.path.join(PROJECT_ROOT, "run_longExp.py"),
        "--is_training",
        "1",
        "--random_seed",
        str(task["seed"]),
        "--root_path",
        os.path.join(PROJECT_ROOT, "dataset/"),
        "--data_path",
        "weather.csv",
        "--model_id",
        task["task_id"],
        "--model",
        "PatchTST",
        "--data",
        "weather",
        "--features",
        "M",
        "--freq",
        "h",
        "--seq_len",
        "512",
        "--pred_len",
        str(args.pred_len),
        "--enc_in",
        "21",
        "--dec_in",
        "21",
        "--c_out",
        "21",
        "--e_layers",
        "3",
        "--n_heads",
        "16",
        "--d_model",
        "128",
        "--d_ff",
        "256",
        "--dropout",
        "0.2",
        "--fc_dropout",
        "0.2",
        "--head_dropout",
        "0",
        "--patch_len",
        "16",
        "--stride",
        "8",
        "--cross_attn_layers",
        "2",
        "--train_epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch_size",
        "128",
        "--learning_rate",
        "0.0001",
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
        "0",
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
        "logcosh_scaled",
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
        "--sacq_grad_firewall_enable",
        "1",
        "--sacq_grad_firewall_hidden",
        "64",
        "--sacq_grad_firewall_temp",
        "1.0",
        "--sacq_grad_firewall_sup_weight",
        str(task["sup_w"]),
        "--sacq_grad_firewall_sparse_weight",
        str(task["sparse_w"]),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    t0 = time.time()
    with open(log_path, "w") as lf:
        lf.write("cmd=" + " ".join(cmd) + "\n\n")
        lf.flush()
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT, check=False)
    elapsed = time.time() - t0

    metrics_path = os.path.join(PROJECT_ROOT, "results", task["setting"], "metrics.json")
    probe_path = os.path.join(PROJECT_ROOT, "results", task["setting"], "grad_probe.json")
    row = {
        "task_id": task["task_id"],
        "sup_weight": task["sup_w"],
        "sparse_weight": task["sparse_w"],
        "status": "ok" if proc.returncode == 0 else "fail",
        "return_code": proc.returncode,
        "gpu": gpu_id,
        "elapsed_sec": round(elapsed, 2),
        "setting": task["setting"],
        "log_path": log_path,
        "mse": "",
        "gradient_blocking_rate_mean": "",
    }
    if proc.returncode == 0 and os.path.exists(metrics_path):
        with open(metrics_path) as f:
            row["mse"] = json.load(f).get("mse")
    if proc.returncode == 0 and os.path.exists(probe_path):
        with open(probe_path) as f:
            row["gradient_blocking_rate_mean"] = json.load(f).get("summary", {}).get("gradient_blocking_rate_mean")
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
                f"[GPU {gpu_id}] sup={row['sup_weight']} sparse={row['sparse_weight']} "
                f"-> {row['status']} mse={row.get('mse','n/a')} block={row.get('gradient_blocking_rate_mean','n/a')}",
                flush=True,
            )
        q.task_done()


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan Weather GFW auxiliary weights")
    parser.add_argument(
        "--weight_pairs",
        type=str,
        default="0.02:0,0.05:0,0.1:0.002,0.2:0.01",
        help="comma-separated sup:sparse pairs",
    )
    parser.add_argument("--gpus", type=str, default="0,1")
    parser.add_argument("--out_dir", type=str, default="logs/weather_gfw_weight_scan")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--pred_len", type=int, default=720)
    parser.add_argument("--noise_sigma", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--probe_epoch", type=int, default=5)
    parser.add_argument("--probe_max_batches", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    pairs = parse_weight_pairs(args.weight_pairs)
    tasks = build_tasks(pairs, args.seed, args.pred_len)
    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "logs"))
    csv_path = os.path.join(out_dir, "summary.csv")

    if args.skip_existing:
        kept = []
        for t in tasks:
            met = os.path.join(PROJECT_ROOT, "results", t["setting"], "metrics.json")
            probe = os.path.join(PROJECT_ROOT, "results", t["setting"], "grad_probe.json")
            if os.path.exists(met) and os.path.exists(probe):
                print(f"skip existing: {t['task_id']}", flush=True)
            else:
                kept.append(t)
        tasks = kept

    header = [
        "task_id",
        "sup_weight",
        "sparse_weight",
        "status",
        "return_code",
        "gpu",
        "elapsed_sec",
        "setting",
        "log_path",
        "mse",
        "gradient_blocking_rate_mean",
    ]
    print(
        f"total_tasks={len(tasks)} pairs={pairs} gpus={args.gpus} "
        f"sigma={args.noise_sigma} epochs={args.epochs}",
        flush=True,
    )
    q = Queue()
    for t in tasks:
        q.put(t)
    lock = threading.Lock()
    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    threads = [
        threading.Thread(target=worker, args=(gid, q, args, out_dir, csv_path, header, lock), daemon=False)
        for gid in gpu_ids
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()

