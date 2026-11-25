"""ETTm1 sl512: SAR (SACQ+AR) on pred_len 96/192/336/720, MSE and/or logcosh, 4-GPU queue.

Same training setup as logs/etth1_sar_sl512 (SAR, 50 ep, patience 20, dsacq off).
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

DATA = "ETTm1"
DATA_PATH = "ETTm1.csv"
ENC_IN = 7
FREQ = "t"
DTAG = "M1"
SEQ_LEN = 512
PRED_LENS = [96, 192, 336, 720]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_tasks(losses: list[str]) -> list[dict]:
    tasks = []
    for loss in losses:
        if loss == "mse":
            tag, des = "SAR", "hSAR"
        elif loss == "logcosh":
            tag, des = "SARlc", "hSARlc"
        else:
            raise ValueError(f"unsupported loss: {loss}")
        for pred_len in PRED_LENS:
            model_id = f"{DTAG}s{SEQ_LEN}p{pred_len}{tag}"
            tasks.append(
                {
                    "task_id": model_id,
                    "model_id": model_id,
                    "pred_len": pred_len,
                    "des": des,
                    "loss": loss,
                    "setting": f"{model_id}_PatchTST_{DATA}_sl{SEQ_LEN}_pl{pred_len}_{des}_0",
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
    log_path = os.path.join(args.out_dir, "logs", f"{task['task_id']}.log")
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
        str(SEQ_LEN),
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
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    t0 = time.time()
    with open(log_path, "w") as lf:
        lf.write(f"task_id={task['task_id']}\n")
        lf.write(f"gpu={gpu_id}\n")
        lf.write(f"loss={task['loss']}\n")
        lf.write(f"setting={task['setting']}\n")
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
        "seq_len": SEQ_LEN,
        "pred_len": task["pred_len"],
        "variant": task["des"],
        "head_type": "sacq_cross_attention",
        "sacq_autoregressive": 1,
        "loss": task["loss"],
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


def worker(gpu_id: int, q: Queue, args, csv_path: str, header: list, lock: threading.Lock):
    while True:
        try:
            task = q.get_nowait()
        except Empty:
            break
        row = run_one(task, gpu_id, args)
        with lock:
            write_csv_row(csv_path, header, row)
            print(
                f"[GPU {gpu_id}] {row['task_id']} loss={row['loss']} -> {row['status']} "
                f"mse={row.get('mse', 'n/a')} ({row['elapsed_sec']}s)",
                flush=True,
            )
        q.task_done()


def main():
    parser = argparse.ArgumentParser(
        description="ETTm1 SAR sl512: 4 pred_lens × MSE/logcosh (same as etth1_sar_sl512)"
    )
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--out_dir", type=str, default="logs/ettm1_sar_sl512")
    parser.add_argument(
        "--losses",
        type=str,
        default="mse,logcosh",
        help="comma-separated: mse, logcosh",
    )
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
    args = parser.parse_args()

    losses = [x.strip() for x in args.losses.split(",") if x.strip()]
    for loss in losses:
        if loss not in ("mse", "logcosh"):
            raise ValueError(f"unsupported loss in --losses: {loss!r}")

    ensure_dir(args.out_dir)
    ensure_dir(os.path.join(args.out_dir, "logs"))
    csv_path = os.path.join(args.out_dir, "summary.csv")

    tasks = build_tasks(losses)
    q: Queue = Queue()
    for t in tasks:
        q.put(t)

    header = [
        "task_id", "status", "return_code", "gpu", "elapsed_sec",
        "dataset", "seq_len", "pred_len", "variant", "head_type",
        "sacq_autoregressive", "loss", "setting", "log_path",
        "mse", "mae", "rse", "rmse", "mape", "mspe", "corr",
        "smape", "mase", "ql50", "pwmae",
    ]
    print(f"total_tasks={len(tasks)} losses={losses} gpus={args.gpus}", flush=True)
    lock = threading.Lock()
    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    threads = [
        threading.Thread(
            target=worker,
            args=(gid, q, args, csv_path, header, lock),
            daemon=False,
        )
        for gid in gpu_ids
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    print("all tasks finished", flush=True)


if __name__ == "__main__":
    main()
