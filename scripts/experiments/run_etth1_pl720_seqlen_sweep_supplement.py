"""ETTh1 pl720: supplement seq_len sweep with Flatten+logcosh_scaled & SAR+MSE.

5 seq_lens: 96, 336, 512, 720, 1440 → 10 tasks (4-GPU queue).
Appends to logs/etth1_pl720_seqlen_sweep/summary.csv
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

DATA = "ETTh1"
DATA_PATH = "ETTh1.csv"
ENC_IN = 7
FREQ = "h"
DTAG = "H1"
PRED_LEN = 720
SEQ_LENS = [96, 336, 512, 720, 1440]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_tasks() -> list[dict]:
    tasks = []
    for seq_len in SEQ_LENS:
        flat_id = f"ETTh1_{seq_len}_{PRED_LEN}"
        tasks.append(
            {
                "task_id": f"{flat_id}Flatslc",
                "model_id": flat_id,
                "seq_len": seq_len,
                "pred_len": PRED_LEN,
                "head_type": "flatten",
                "loss": "logcosh_scaled",
                "variant": "flatten_logcosh_scaled",
                "des": "hFlatslc",
                "setting": f"{flat_id}_PatchTST_{DATA}_sl{seq_len}_pl{PRED_LEN}_hFlatslc_0",
            }
        )
        sar_id = f"{DTAG}s{seq_len}p{PRED_LEN}SAR"
        tasks.append(
            {
                "task_id": sar_id,
                "model_id": sar_id,
                "seq_len": seq_len,
                "pred_len": PRED_LEN,
                "head_type": "sacq_cross_attention",
                "loss": "mse",
                "variant": "sar_mse",
                "des": "hSAR",
                "sacq_autoregressive": 1,
                "setting": f"{sar_id}_PatchTST_{DATA}_sl{seq_len}_pl{PRED_LEN}_hSAR_0",
            }
        )
    return tasks


def write_csv_row(path: str, header: list, row: dict) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_one(task: dict, gpu_id: int, args, out_dir: str) -> dict:
    log_path = os.path.join(out_dir, "logs", f"{task['task_id']}.log")
    metrics_path = os.path.join("results", task["setting"], "metrics.json")
    py = os.environ.get("PYTHON", sys.executable)
    cmd = [
        py,
        "-u",
        "run_longExp.py",
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
        task["head_type"],
        "--short_setting",
        "1",
        "--des",
        task["des"],
        "--skip_visual",
        "1",
        "--loss",
        task["loss"],
    ]

    if task["loss"] == "logcosh_scaled":
        cmd.extend(
            [
                "--logcosh_beta",
                str(args.logcosh_beta),
                "--logcosh_beta_mode",
                args.logcosh_beta_mode,
            ]
        )

    if task["head_type"] == "sacq_cross_attention":
        cmd.extend(
            [
                "--cross_attn_layers",
                "2",
                "--sacq_autoregressive",
                "1",
                "--sacq_alpha_mode",
                "learnable",
                "--dsacq_enable",
                "0",
                "--dsacq_mem_gate_enable",
                "0",
                "--dsacq_drift_gate_enable",
                "0",
            ]
        )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    t0 = time.time()
    with open(log_path, "w") as lf:
        lf.write(f"task_id={task['task_id']}\n")
        lf.write(f"gpu={gpu_id}\n")
        lf.write(f"setting={task['setting']}\n")
        lf.write("supplement=flatten_logcosh_scaled + sar_mse (5 seq_lens)\n")
        lf.write("cmd=" + " ".join(cmd) + "\n\n")
        lf.flush()
        proc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, check=False)
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
        "variant": task["variant"],
        "head_type": task["head_type"],
        "loss": task["loss"],
        "setting": task["setting"],
        "log_path": log_path,
    }
    if task.get("sacq_autoregressive"):
        row["sacq_autoregressive"] = 1
    if proc.returncode == 0 and os.path.exists(metrics_path):
        with open(metrics_path) as f:
            m = json.load(f)
        for k in (
            "mse", "mae", "rse", "rmse", "mape", "mspe", "corr",
            "smape", "mase", "ql50", "pwmae",
        ):
            row[k] = m.get(k)
    return row


def worker(gpu_id: int, q: Queue, args, out_dir: str, csv_path: str, header: list, lock: threading.Lock):
    while True:
        try:
            task = q.get_nowait()
        except Empty:
            break
        row = run_one(task, gpu_id, args, out_dir)
        with lock:
            write_csv_row(csv_path, header, row)
            print(
                f"[GPU {gpu_id}] {row['task_id']} -> {row['status']} "
                f"mse={row.get('mse', 'n/a')} ({row['elapsed_sec']}s)",
                flush=True,
            )
        q.task_done()


def main():
    parser = argparse.ArgumentParser(
        description="ETTh1 pl720 seq_len supplement: Flatten+logcosh & SAR+MSE"
    )
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--out_dir", type=str, default="logs/etth1_pl720_seqlen_sweep")
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
    parser.add_argument("--logcosh_beta", type=float, default=1.0)
    parser.add_argument("--logcosh_beta_mode", type=str, default="auto_mad")
    args = parser.parse_args()

    out_dir = args.out_dir
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "logs"))

    tasks = build_tasks()
    q: Queue = Queue()
    for t in tasks:
        q.put(t)

    csv_path = os.path.join(out_dir, "summary.csv")
    header = [
        "task_id", "status", "return_code", "gpu", "elapsed_sec",
        "dataset", "seq_len", "pred_len", "variant", "head_type",
        "sacq_autoregressive", "loss", "setting", "log_path",
        "mse", "mae", "rse", "rmse", "mape", "mspe", "corr",
        "smape", "mase", "ql50", "pwmae",
    ]
    print(
        f"supplement_tasks={len(tasks)} gpus={args.gpus} "
        f"pred_len={PRED_LEN} seq_lens={SEQ_LENS}",
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
    print("all supplement tasks finished", flush=True)


if __name__ == "__main__":
    main()
