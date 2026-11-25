"""ETTh1 pl=720: train + test CSV bilateral spike outliers (SAR vs SACQ).

Unlike train-only spike CSVs, the test segment (last 4 months) is also corrupted.
Scaler is fit on clean ETTh1 train data; val segment stays clean.

Example::
  python scripts/experiments/run_etth1_spike_bilateral_pl720.py --prepare_only
  python scripts/experiments/run_etth1_spike_bilateral_pl720.py --gpus 0,1 --outlier_pcts 0,5
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

from scripts.experiments.outlier_utils import (
    bilateral_spike_tag,
    ensure_bilateral_spike_dataset,
)

DATA = "ETTh1"
ENC_IN = 7
FREQ = "h"
SEQ_LEN = 512
PRED_LEN = 720

VARIANTS = [
    {
        "tag": "SAR+logcosh_scaled",
        "suffix": "SARslc",
        "des": "hSARslc",
        "head_type": "sacq_cross_attention",
        "sacq_ar": 1,
        "loss": "logcosh_scaled",
    },
    {
        "tag": "SACQ+logcosh_scaled",
        "suffix": "SACQslc",
        "des": "hSACQslc",
        "head_type": "sacq_cross_attention",
        "sacq_ar": 0,
        "loss": "logcosh_scaled",
    },
    {
        "tag": "Flatten+logcosh_scaled",
        "suffix": "Flatslc",
        "des": "hFlatslc",
        "head_type": "flatten",
        "sacq_ar": 0,
        "loss": "logcosh_scaled",
    },
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_tasks(
    outlier_pcts: list[float],
    variants: list[dict],
    test_pcts: list[float] | None,
) -> list[dict]:
    tasks = []
    test_pcts = test_pcts if test_pcts is not None else outlier_pcts
    for train_pct in outlier_pcts:
        for test_pct in test_pcts:
            if train_pct <= 0 and test_pct <= 0:
                spike_tag = "clean"
            else:
                spike_tag = bilateral_spike_tag(train_pct, test_pct)
            for v in variants:
                model_id = f"H1s{SEQ_LEN}p{PRED_LEN}{v['suffix']}_{spike_tag}"
                des = f"{v['des']}_{spike_tag}" if spike_tag != "clean" else v["des"]
                tasks.append(
                    {
                        "task_id": model_id,
                        "model_id": model_id,
                        "pred_len": PRED_LEN,
                        "des": des,
                        "train_outlier_pct": train_pct,
                        "test_outlier_pct": test_pct,
                        "variant_tag": v["tag"],
                        "head_type": v["head_type"],
                        "sacq_ar": v["sacq_ar"],
                        "loss": v["loss"],
                        "setting": (
                            f"{model_id}_PatchTST_{DATA}_sl{SEQ_LEN}_pl{PRED_LEN}_{des}_0"
                        ),
                    }
                )
    return tasks


def write_csv_row(path, header, row):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_one(task, gpu_id, args, out_dir, dataset_dir):
    train_pct = task["train_outlier_pct"]
    test_pct = task["test_outlier_pct"]
    data_path, inj_meta = ensure_bilateral_spike_dataset(
        dataset_dir,
        train_pct,
        test_pct,
        noise_mult=args.noise_mult,
        train_seed=args.train_seed,
        test_seed=args.test_seed,
        force=args.force_data,
    )
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
        data_path,
        "--scaler_data_path",
        "ETTh1.csv" if (train_pct > 0 or test_pct > 0) else "",
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
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    t0 = time.time()
    with open(log_path, "w") as lf:
        lf.write(f"task_id={task['task_id']}\n")
        lf.write(f"gpu={gpu_id}\n")
        lf.write(f"train_outlier_pct={train_pct}\n")
        lf.write(f"test_outlier_pct={test_pct}\n")
        lf.write(f"data_path={data_path}\n")
        lf.write(f"inj_meta={json.dumps(inj_meta)}\n")
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
        "train_outlier_pct": train_pct,
        "test_outlier_pct": test_pct,
        "noise_mult": args.noise_mult,
        "train_seed": args.train_seed,
        "test_seed": args.test_seed,
        "variant_tag": task["variant_tag"],
        "head_type": task["head_type"],
        "loss": task["loss"],
        "data_path": data_path,
        "setting": task["setting"],
        "log_path": log_path,
    }
    if proc.returncode == 0 and os.path.exists(metrics_path):
        with open(metrics_path) as f:
            m = json.load(f)
        for k in ("mse", "mae", "rse", "rmse"):
            row[k] = m.get(k)
    return row


def worker(gpu_id, q, args, out_dir, dataset_dir, csv_path, header, lock):
    while True:
        try:
            task = q.get_nowait()
        except Empty:
            break
        row = run_one(task, gpu_id, args, out_dir, dataset_dir)
        with lock:
            write_csv_row(csv_path, header, row)
            print(
                f"[GPU {gpu_id}] {row['task_id']} "
                f"tr={row['train_outlier_pct']}% te={row['test_outlier_pct']}% "
                f"-> {row['status']} mse={row.get('mse', 'n/a')} ({row['elapsed_sec']}s)",
                flush=True,
            )
        q.task_done()


def main():
    parser = argparse.ArgumentParser(
        description="ETTh1 pl720 bilateral (train+test CSV) spike-outlier sweep"
    )
    parser.add_argument("--gpus", type=str, default="0,1,2")
    parser.add_argument(
        "--out_dir", type=str, default="logs/etth1_spike_bilateral_pl720"
    )
    parser.add_argument(
        "--outlier_pcts",
        type=str,
        default="0,5",
        help="train spike %% (also used for test unless --test_pcts set)",
    )
    parser.add_argument(
        "--test_pcts",
        type=str,
        default="",
        help="optional comma-separated test %% (default: same as train)",
    )
    parser.add_argument("--noise_mult", type=float, default=3.0)
    parser.add_argument("--train_seed", type=int, default=2021)
    parser.add_argument("--test_seed", type=int, default=2022)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2021, help="training random seed")
    parser.add_argument("--e_layers", type=int, default=3)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--fc_dropout", type=float, default=0.3)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--force_data", action="store_true")
    parser.add_argument(
        "--variant_tags",
        type=str,
        default="",
        help="comma-separated variant tags (default: all)",
    )
    args = parser.parse_args()

    dataset_dir = os.path.join(PROJECT_ROOT, "dataset")
    outlier_pcts = [float(x.strip()) for x in args.outlier_pcts.split(",") if x.strip()]
    test_pcts = (
        [float(x.strip()) for x in args.test_pcts.split(",") if x.strip()]
        if args.test_pcts.strip()
        else None
    )
    test_list = test_pcts if test_pcts is not None else outlier_pcts

    for train_pct in outlier_pcts:
        for test_pct in test_list:
            data_path, meta = ensure_bilateral_spike_dataset(
                dataset_dir,
                train_pct,
                test_pct,
                noise_mult=args.noise_mult,
                train_seed=args.train_seed,
                test_seed=args.test_seed,
                force=args.force_data,
            )
            print(
                f"prepared tr={train_pct}% te={test_pct}% -> {data_path} "
                f"({meta.get('mode', 'n/a')})",
                flush=True,
            )

    if args.prepare_only:
        return

    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "logs"))
    csv_path = os.path.join(out_dir, "summary.csv")

    variants = VARIANTS
    if args.variant_tags.strip():
        want = {x.strip() for x in args.variant_tags.split(",") if x.strip()}
        variants = [v for v in VARIANTS if v["tag"] in want]
        if not variants:
            raise SystemExit(f"No variants matched: {args.variant_tags}")

    tasks = build_tasks(outlier_pcts, variants, test_pcts)
    q = Queue()
    for t in tasks:
        q.put(t)

    header = [
        "task_id",
        "status",
        "return_code",
        "gpu",
        "elapsed_sec",
        "train_outlier_pct",
        "test_outlier_pct",
        "noise_mult",
        "train_seed",
        "test_seed",
        "variant_tag",
        "head_type",
        "loss",
        "data_path",
        "setting",
        "log_path",
        "mse",
        "mae",
        "rse",
        "rmse",
    ]
    print(f"total_tasks={len(tasks)} gpus={args.gpus}", flush=True)
    lock = threading.Lock()
    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    threads = [
        threading.Thread(
            target=worker,
            args=(gid, q, args, out_dir, dataset_dir, csv_path, header, lock),
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
