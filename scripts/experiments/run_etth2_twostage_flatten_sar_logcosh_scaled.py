"""ETTh2: two-stage Flatten -> SAR with ScaledLogCosh (auto beta).

Aligned with `logs/etth2_sar_sl512_logcosh_scaled` (single-stage SAR):
  - PatchTST sl=512, pl in {96,192,336,720}
  - loss=logcosh_scaled, logcosh_beta_mode=auto_mad
  - SAR: sacq_autoregressive=1, sacq_alpha_mode=learnable, sacq_pe_mode=default
  - train_epochs=50, patience=20, batch=128, lr=1e-4

Two-stage protocol:
  Stage 1: head_type=flatten (coarse head only)
  Stage 2: head_type=sacq_cross_attention, init from Stage 1,
           bootstrap_flatten_head=1, sacq_coarse_lr_ratio=0.01
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

DATA = "ETTh2"
DATA_PATH = "ETTh2.csv"
ENC_IN = 7
FREQ = "h"
SEQ_LEN = 512
PRED_LENS = [96, 192, 336, 720]
COARSE_LR_RATIO = 0.01


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def short_setting(model_id: str, des: str, pred_len: int, ii: int = 0) -> str:
    return f"{model_id}_PatchTST_{DATA}_sl{SEQ_LEN}_pl{pred_len}_{des}_{ii}"


def stage1_model_id(pred_len: int) -> str:
    return f"E2s{SEQ_LEN}p{pred_len}Ftwslc"


def stage2_model_id(pred_len: int) -> str:
    return f"E2s{SEQ_LEN}p{pred_len}SARslcTw"


def build_base_cmd(args, model_id: str, des: str, pred_len: int) -> list[str]:
    py = os.environ.get("PYTHON", DEFAULT_PYTHON if os.path.isfile(DEFAULT_PYTHON) else sys.executable)
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
        DATA_PATH,
        "--model_id",
        model_id,
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
        str(pred_len),
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


def run_one(cmd: list[str], gpu_id: int, log_path: str) -> dict:
    ensure_dir(os.path.dirname(log_path))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    t0 = time.time()
    with open(log_path, "w") as lf:
        lf.write("cmd=" + " ".join(cmd) + "\n\n")
        lf.flush()
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
    return {"return_code": proc.returncode, "elapsed_sec": round(time.time() - t0, 2)}


def worker(gpu_id: int, q: Queue, csv_path: str, header: list, lock: threading.Lock) -> None:
    while True:
        try:
            task = q.get_nowait()
        except Empty:
            break
        result = run_one(task["cmd"], gpu_id, task["log_path"])
        row = {**task["row_meta"], **result, "gpu": gpu_id, "log_path": task["log_path"]}
        setting = task["row_meta"].get("setting")
        if result["return_code"] == 0 and setting:
            metrics_path = os.path.join(PROJECT_ROOT, "results", setting, "metrics.json")
            if os.path.isfile(metrics_path):
                with open(metrics_path) as f:
                    m = json.load(f)
                for k in ("mse", "mae", "rse", "rmse"):
                    row[k] = m.get(k)
        with lock:
            exists = os.path.isfile(csv_path)
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=header)
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
            print(
                f"[GPU {gpu_id}] {task['row_meta']['task_id']} -> "
                f"{'ok' if result['return_code'] == 0 else 'fail'} "
                f"mse={row.get('mse', 'n/a')} ({result['elapsed_sec']}s)",
                flush=True,
            )
        q.task_done()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="logs/etth2_twostage_sar_logcosh_scaled",
    )
    parser.add_argument("--seed", type=int, default=2021)
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
    parser.add_argument("--skip_stage1", action="store_true")
    parser.add_argument("--skip_stage2", action="store_true")
    args = parser.parse_args()

    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    log_dir = os.path.join(out_dir, "logs")
    ensure_dir(log_dir)
    csv_path = os.path.join(out_dir, "summary.csv")
    header = [
        "phase", "task_id", "return_code", "gpu", "elapsed_sec",
        "pred_len", "model_id", "des", "setting", "coarse_lr_ratio", "log_path",
        "mse", "mae", "rse", "rmse",
    ]

    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    lock = threading.Lock()

    if not args.skip_stage1:
        s1_tasks: list[dict] = []
        for pl in PRED_LENS:
            s1_mid = stage1_model_id(pl)
            s1_des = "hFtwslc"
            cmd = build_base_cmd(args, model_id=s1_mid, des=s1_des, pred_len=pl)
            cmd += ["--head_type", "flatten"]
            setting = short_setting(s1_mid, s1_des, pl)
            s1_tasks.append(
                {
                    "cmd": cmd,
                    "log_path": os.path.join(log_dir, f"s1_pl{pl}.log"),
                    "row_meta": {
                        "phase": "s1_flatten",
                        "task_id": f"s1_pl{pl}",
                        "pred_len": pl,
                        "model_id": s1_mid,
                        "des": s1_des,
                        "setting": setting,
                        "coarse_lr_ratio": "",
                    },
                }
            )
        q: Queue = Queue()
        for t in s1_tasks:
            q.put(t)
        print(f"[Stage1 Flatten] total_tasks={len(s1_tasks)} gpus={args.gpus}", flush=True)
        threads = [
            threading.Thread(target=worker, args=(gid, q, csv_path, header, lock), daemon=False)
            for gid in gpu_ids
        ]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        print("[Stage1] all tasks finished", flush=True)
    else:
        print("[Stage1] skipped", flush=True)

    if not args.skip_stage2:
        s2_tasks: list[dict] = []
        for pl in PRED_LENS:
            s1_mid = stage1_model_id(pl)
            s1_des = "hFtwslc"
            s1_setting = short_setting(s1_mid, s1_des, pl)
            s2_mid = stage2_model_id(pl)
            s2_des = "hSARslcTw"
            cmd = build_base_cmd(args, model_id=s2_mid, des=s2_des, pred_len=pl)
            cmd += [
                "--head_type",
                "sacq_cross_attention",
                "--sacq_autoregressive",
                "1",
                "--sacq_scheme",
                "default",
                "--sacq_pe_mode",
                "default",
                "--sacq_alpha_mode",
                "learnable",
                "--sacq_coarse_lr_ratio",
                str(COARSE_LR_RATIO),
                "--init_checkpoint_setting",
                s1_setting,
                "--bootstrap_flatten_head",
                "1",
            ]
            setting = short_setting(s2_mid, s2_des, pl)
            s2_tasks.append(
                {
                    "cmd": cmd,
                    "log_path": os.path.join(log_dir, f"s2_pl{pl}.log"),
                    "row_meta": {
                        "phase": "s2_sar_joint",
                        "task_id": f"s2_pl{pl}",
                        "pred_len": pl,
                        "model_id": s2_mid,
                        "des": s2_des,
                        "setting": setting,
                        "coarse_lr_ratio": COARSE_LR_RATIO,
                    },
                }
            )
        q2: Queue = Queue()
        for t in s2_tasks:
            q2.put(t)
        print(
            f"[Stage2 SAR joint] total_tasks={len(s2_tasks)} "
            f"coarse_lr_ratio={COARSE_LR_RATIO} gpus={args.gpus}",
            flush=True,
        )
        threads = [
            threading.Thread(target=worker, args=(gid, q2, csv_path, header, lock), daemon=False)
            for gid in gpu_ids
        ]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        print("[Stage2] all tasks finished", flush=True)
    else:
        print("[Stage2] skipped", flush=True)


if __name__ == "__main__":
    main()
