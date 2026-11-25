#!/usr/bin/env python3
"""Mamba / Linear / DLinear × ETT datasets × pl — Flatten+MSE vs SAR+logcosh_scaled.

4-GPU task queue. Hyperparams: scripts/PatchTST/*.sh (sl=512), DLinear lr/batch from scripts/Linear/*.sh.
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

DEFAULT_PYTHON = "/home/bkai/miniconda3/envs/forecast/bin/python"
SEQ_LEN = 512
DEFAULT_PRED_LENS = [96, 192, 336, 720]

MODELS = [
    ("PatchTSTMamba", "Mmb"),
    ("Linear", "Lin"),
    ("DLinear", "Dln"),
]

VARIANTS = [
    {
        "variant": "flatten_mse",
        "head_type": "flatten",
        "loss": "mse",
        "htag": "F",
        "des": "hFlatslc",
        "sacq_ar": False,
    },
    {
        "variant": "sar_logcosh_scaled",
        "head_type": "sacq_cross_attention",
        "loss": "logcosh_scaled",
        "htag": "S",
        "des": "hSARslc",
        "sacq_ar": True,
    },
]

DATASETS = [
    {
        "data": "ETTh1",
        "data_path": "ETTh1.csv",
        "enc_in": 7,
        "freq": "h",
        "dtag": "H1",
        "e_layers": 3,
        "n_heads": 4,
        "d_model": 16,
        "d_ff": 128,
        "dropout": 0.3,
        "fc_dropout": 0.3,
        "batch_size": 128,
        "lr": 1e-4,
        "lradj": None,
        "pct_start": None,
        "dlinear_batch_size": 32,
        "dlinear_lr": 0.005,
    },
    {
        "data": "ETTh2",
        "data_path": "ETTh2.csv",
        "enc_in": 7,
        "freq": "h",
        "dtag": "H2",
        "e_layers": 3,
        "n_heads": 4,
        "d_model": 16,
        "d_ff": 128,
        "dropout": 0.3,
        "fc_dropout": 0.3,
        "batch_size": 128,
        "lr": 1e-4,
        "lradj": None,
        "pct_start": None,
        "dlinear_batch_size": 32,
        "dlinear_lr": 0.05,
    },
    {
        "data": "ETTm1",
        "data_path": "ETTm1.csv",
        "enc_in": 7,
        "freq": "t",
        "dtag": "M1",
        "e_layers": 3,
        "n_heads": 16,
        "d_model": 128,
        "d_ff": 256,
        "dropout": 0.2,
        "fc_dropout": 0.2,
        "batch_size": 128,
        "lr": 1e-4,
        "lradj": "TST",
        "pct_start": 0.4,
        "dlinear_batch_size": 8,
        "dlinear_lr": 1e-4,
    },
    {
        "data": "ETTm2",
        "data_path": "ETTm2.csv",
        "enc_in": 7,
        "freq": "t",
        "dtag": "M2",
        "e_layers": 3,
        "n_heads": 16,
        "d_model": 128,
        "d_ff": 256,
        "dropout": 0.2,
        "fc_dropout": 0.2,
        "batch_size": 128,
        "lr": 1e-4,
        "lradj": "TST",
        "pct_start": 0.4,
        "dlinear_batch_size": 32,
        "dlinear_lr": 1e-3,
        "dlinear_lr_by_pl": {96: 1e-3, 192: 1e-3, 336: 0.01, 720: 0.1},
    },
    {
        "data": "weather",
        "data_path": "weather.csv",
        "enc_in": 21,
        "freq": "h",
        "dtag": "W",
        "e_layers": 3,
        "n_heads": 16,
        "d_model": 128,
        "d_ff": 256,
        "dropout": 0.2,
        "fc_dropout": 0.2,
        "batch_size": 128,
        "lr": 1e-4,
        "lradj": None,
        "pct_start": None,
        "dlinear_batch_size": 16,
        "dlinear_lr": 1e-4,
    },
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_setting(task: dict) -> str:
    return (
        f"{task['model_id']}_{task['model']}_{task['data']}_"
        f"sl{task['seq_len']}_pl{task['pred_len']}_{task['des']}_0"
    )


def dlinear_lr_for(ds: dict, pred_len: int) -> float:
    by_pl = ds.get("dlinear_lr_by_pl")
    if by_pl and pred_len in by_pl:
        return by_pl[pred_len]
    return ds["dlinear_lr"]


def build_tasks(
    datasets: list[dict],
    models: list[tuple[str, str]],
    pred_lens: list[int],
) -> list[dict]:
    tasks = []
    for ds in datasets:
        for pred_len in pred_lens:
            for model, mtag in models:
                for var in VARIANTS:
                    model_id = f"{ds['dtag']}s{SEQ_LEN}p{pred_len}{mtag}{var['htag']}"
                    dlr = dlinear_lr_for(ds, pred_len) if model == "DLinear" else ds["lr"]
                    task = {
                        "task_id": model_id,
                        "model_id": model_id,
                        "data": ds["data"],
                        "data_path": ds["data_path"],
                        "enc_in": ds["enc_in"],
                        "freq": ds["freq"],
                        "dtag": ds["dtag"],
                        "seq_len": SEQ_LEN,
                        "pred_len": pred_len,
                        "model": model,
                        "mtag": mtag,
                        "variant": var["variant"],
                        "head_type": var["head_type"],
                        "loss": var["loss"],
                        "des": var["des"],
                        "sacq_ar": var["sacq_ar"],
                        "e_layers": ds["e_layers"],
                        "n_heads": ds["n_heads"],
                        "d_model": ds["d_model"],
                        "d_ff": ds["d_ff"],
                        "dropout": ds["dropout"],
                        "fc_dropout": ds["fc_dropout"],
                        "batch_size": ds["dlinear_batch_size"] if model == "DLinear" else ds["batch_size"],
                        "lr": dlr,
                        "lradj": ds["lradj"],
                        "pct_start": ds["pct_start"],
                    }
                    task["setting"] = build_setting(task)
                    tasks.append(task)
    return tasks


def write_csv_row(path: str, header: list, row: dict) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_one(task: dict, gpu_id: int, args) -> dict:
    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    ensure_dir(os.path.join(out_dir, "logs"))
    log_path = os.path.join(out_dir, "logs", f"{task['task_id']}.log")
    metrics_path = os.path.join(PROJECT_ROOT, "results", task["setting"], "metrics.json")
    py = os.environ.get("PYTHON", args.python or sys.executable)

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
        task["model"],
        "--data",
        task["data"],
        "--features",
        "M",
        "--freq",
        task["freq"],
        "--seq_len",
        str(task["seq_len"]),
        "--pred_len",
        str(task["pred_len"]),
        "--enc_in",
        str(task["enc_in"]),
        "--dec_in",
        str(task["enc_in"]),
        "--c_out",
        str(task["enc_in"]),
        "--e_layers",
        str(task["e_layers"]),
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
        "--train_epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch_size",
        str(task["batch_size"]),
        "--learning_rate",
        str(task["lr"]),
        "--itr",
        "1",
        "--num_workers",
        str(args.num_workers),
        "--head_type",
        task["head_type"],
        "--loss",
        task["loss"],
        "--short_setting",
        "1",
        "--des",
        task["des"],
        "--skip_visual",
        "1",
    ]
    if task["lradj"]:
        cmd.extend(["--lradj", task["lradj"]])
    if task["pct_start"] is not None:
        cmd.extend(["--pct_start", str(task["pct_start"])])

    if task["head_type"] == "sacq_cross_attention":
        cmd.extend(
            [
                "--cross_attn_layers",
                "2",
                "--sacq_autoregressive",
                "1" if task["sacq_ar"] else "0",
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
        if task["loss"] == "logcosh_scaled":
            cmd.extend(
                [
                    "--logcosh_beta",
                    "1.0",
                    "--logcosh_beta_mode",
                    "auto_mad",
                ]
            )

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
        "model": task["model"],
        "seq_len": task["seq_len"],
        "pred_len": task["pred_len"],
        "variant": task["variant"],
        "head_type": task["head_type"],
        "loss": task["loss"],
        "batch_size": task["batch_size"],
        "lr": task["lr"],
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
            print(
                f"[GPU {gpu_id}] {row['task_id']} -> {row['status']} "
                f"mse={row.get('mse', 'n/a')} ({row['elapsed_sec']}s)",
                flush=True,
            )
        q.task_done()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backbone×dataset: Flatten+MSE vs SAR+logcosh_scaled (4-GPU queue)"
    )
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument(
        "--datasets",
        type=str,
        default="ETTh1,ETTh2,ETTm1,ETTm2",
        help="comma-separated: ETTh1,ETTh2,ETTm1,ETTm2,weather",
    )
    parser.add_argument(
        "--pred_lens",
        type=str,
        default="96,192,336,720",
        help="comma-separated pred lengths, e.g. 96,720",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="skip tasks whose results/<setting>/metrics.json already exists",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="PatchTSTMamba,Linear,DLinear",
        help="comma-separated: PatchTSTMamba,Linear,DLinear",
    )
    parser.add_argument("--out_dir", type=str, default="logs/backbone_flatten_vs_sacq")
    parser.add_argument("--python", type=str, default=DEFAULT_PYTHON)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    args = parser.parse_args()

    ds_names = {x.strip() for x in args.datasets.split(",") if x.strip()}
    datasets = [d for d in DATASETS if d["data"] in ds_names]
    if not datasets:
        raise SystemExit(f"No datasets matched: {args.datasets}")

    pred_lens = [int(x.strip()) for x in args.pred_lens.split(",") if x.strip()]
    if not pred_lens:
        raise SystemExit("empty --pred_lens")

    model_names = {x.strip() for x in args.models.split(",") if x.strip()}
    models = [m for m in MODELS if m[0] in model_names]
    if not models:
        raise SystemExit(f"No models matched: {args.models}")

    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "logs"))
    csv_path = os.path.join(out_dir, "summary.csv")

    tasks = build_tasks(datasets, models, pred_lens)
    if args.skip_existing:
        kept = []
        for t in tasks:
            mp = os.path.join(PROJECT_ROOT, "results", t["setting"], "metrics.json")
            if os.path.exists(mp):
                print(f"skip existing: {t['task_id']}", flush=True)
            else:
                kept.append(t)
        tasks = kept
    q: Queue = Queue()
    for t in tasks:
        q.put(t)

    header = [
        "task_id", "status", "return_code", "gpu", "elapsed_sec",
        "dataset", "model", "seq_len", "pred_len", "variant",
        "head_type", "loss", "batch_size", "lr", "setting", "log_path",
        "mse", "mae", "rse", "rmse", "mape", "mspe", "corr",
        "smape", "mase", "ql50", "pwmae",
    ]
    print(
        f"total_tasks={len(tasks)} datasets={[d['data'] for d in datasets]} "
        f"pred_lens={pred_lens} models={[m[0] for m in models]} gpus={args.gpus} "
        f"epochs={args.epochs} patience={args.patience}",
        flush=True,
    )
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
