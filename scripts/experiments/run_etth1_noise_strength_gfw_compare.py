#!/usr/bin/env python3
"""ETTh1 paired-noise strength sweep: SACQ baseline vs SACQ+gradient-firewall."""
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
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow(row)


def build_tasks(sigmas: list[float], pred_len: int) -> list[dict]:
    tasks = []
    variants = [
        {
            "name": "sacq_no_gfw",
            "enable_gfw": 0,
            "sup_w": 0.0,
            "sparse_w": 0.0,
            "suffix": "SACQslc",
            "des": "hSACQslc_nsweep",
        },
        {
            "name": "sacq_with_gfw",
            "enable_gfw": 1,
            "sup_w": 0.2,
            "sparse_w": 0.01,
            "suffix": "SACQslcGFW",
            "des": "hSACQslcGFW_nsweep",
        },
    ]
    for sigma in sigmas:
        sigma_tag = str(sigma).replace(".", "p")
        for v in variants:
            model_id = f"H1s512p{pred_len}{v['suffix']}ns{sigma_tag}"
            des = f"{v['des']}_ns{sigma_tag}"
            setting = f"{model_id}_PatchTST_ETTh1_sl512_pl{pred_len}_{des}_0"
            tasks.append(
                {
                    "task_id": model_id,
                    "setting": setting,
                    "variant": v["name"],
                    "enable_gfw": v["enable_gfw"],
                    "sup_w": v["sup_w"],
                    "sparse_w": v["sparse_w"],
                    "noise_sigma": sigma,
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
        str(args.seed),
        "--root_path",
        os.path.join(PROJECT_ROOT, "dataset/"),
        "--data_path",
        "ETTh1.csv",
        "--model_id",
        task["task_id"],
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
        str(args.pred_len),
        "--enc_in",
        "7",
        "--dec_in",
        "7",
        "--c_out",
        "7",
        "--e_layers",
        "3",
        "--n_heads",
        "4",
        "--d_model",
        "16",
        "--d_ff",
        "128",
        "--dropout",
        "0.3",
        "--fc_dropout",
        "0.3",
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
        str(task["noise_sigma"]),
        "--grad_probe_enable",
        "1",
        "--grad_probe_epoch",
        str(args.probe_epoch),
        "--grad_probe_max_batches",
        str(args.probe_max_batches),
        "--sacq_grad_firewall_enable",
        str(task["enable_gfw"]),
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
        "variant": task["variant"],
        "noise_sigma": task["noise_sigma"],
        "status": "ok" if proc.returncode == 0 else "fail",
        "return_code": proc.returncode,
        "gpu": gpu_id,
        "elapsed_sec": round(elapsed, 2),
        "setting": task["setting"],
        "log_path": log_path,
        "mse": "",
        "gradient_blocking_rate_mean": "",
        "backbone_noise_to_total_ratio_mean": "",
    }
    if proc.returncode == 0 and os.path.exists(metrics_path):
        with open(metrics_path) as f:
            met = json.load(f)
        row["mse"] = met.get("mse")
    if proc.returncode == 0 and os.path.exists(probe_path):
        with open(probe_path) as f:
            probe = json.load(f)
        s = probe.get("summary", {})
        row["gradient_blocking_rate_mean"] = s.get("gradient_blocking_rate_mean")
        row["backbone_noise_to_total_ratio_mean"] = s.get("backbone_noise_to_total_ratio_mean")
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
                f"[GPU {gpu_id}] sigma={row['noise_sigma']} {row['variant']} -> {row['status']} "
                f"mse={row.get('mse', 'n/a')} block={row.get('gradient_blocking_rate_mean', 'n/a')}",
                flush=True,
            )
        q.task_done()


def main() -> None:
    parser = argparse.ArgumentParser(description="ETTh1 noise-strength sweep with/without gradient firewall")
    parser.add_argument("--gpus", type=str, default="0,1")
    parser.add_argument("--out_dir", type=str, default="logs/etth1_noise_strength_gfw_compare")
    parser.add_argument("--sigmas", type=str, default="7,10,14")
    parser.add_argument("--pred_len", type=int, default=720)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--probe_epoch", type=int, default=5)
    parser.add_argument("--probe_max_batches", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    sigmas = [float(x.strip()) for x in args.sigmas.split(",") if x.strip()]
    tasks = build_tasks(sigmas, args.pred_len)
    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "logs"))
    csv_path = os.path.join(out_dir, "summary.csv")

    if args.skip_existing:
        kept = []
        for t in tasks:
            probe_path = os.path.join(PROJECT_ROOT, "results", t["setting"], "grad_probe.json")
            met_path = os.path.join(PROJECT_ROOT, "results", t["setting"], "metrics.json")
            if os.path.exists(probe_path) and os.path.exists(met_path):
                print(f"skip existing: {t['task_id']}", flush=True)
            else:
                kept.append(t)
        tasks = kept

    header = [
        "task_id",
        "variant",
        "noise_sigma",
        "status",
        "return_code",
        "gpu",
        "elapsed_sec",
        "setting",
        "log_path",
        "mse",
        "gradient_blocking_rate_mean",
        "backbone_noise_to_total_ratio_mean",
    ]
    print(
        f"total_tasks={len(tasks)} sigmas={sigmas} gpus={args.gpus} "
        f"epochs={args.epochs} probe_epoch={args.probe_epoch}",
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
