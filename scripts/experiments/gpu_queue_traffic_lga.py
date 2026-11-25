#!/usr/bin/env python3
"""GPU queue: Traffic (SACQ+SAR+logcosh_scaled, traffic.sh hparams) then LGA official defaults.

One training job per GPU at a time. Logs under logs/batch_traffic_lga_queue/.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from queue import Empty, Queue

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LGA_ROOT = os.path.abspath(os.path.expanduser(os.environ.get("FCD_LGA_ROOT", os.path.join(ROOT, "external", "LGA"))))
PY = os.environ.get("PYTHON", sys.executable)
LOG_ROOT = os.path.join(ROOT, "logs", "batch_traffic_lga_queue")

TRAFFIC_PRED_LENS = [96, 192, 336, 720]

LGA_JOBS = [
    # ETTm1.sh defaults
    *[
        {
            "group": "lga",
            "task_id": f"ETTm1_512_{pl}",
            "cwd": LGA_ROOT,
            "log": f"lga/ETTm1_512_{pl}.log",
            "args": [
                "--is_training", "1",
                "--root_path", "./dataset/ETT-small/",
                "--data_path", "ETTm1.csv",
                "--model_id", f"ETTm1_512_{pl}",
                "--model", "PatchTST",
                "--data", "ETTm1",
                "--features", "M",
                "--seq_len", "512",
                "--pred_len", str(pl),
                "--e_layers", "3",
                "--enc_in", "7",
                "--des", "Exp",
                "--lga",
                "--itr", "1",
                "--eps", "1e-2",
                "--d_G", "64",
                "--batch_size", "128",
                "--train_epochs", "100",
                "--patience", "10",
                "--pct_start", "0.4",
                "--learning_rate", "1e-4",
            ],
        }
        for pl in [96, 192, 336, 720]
    ],
    # ETTm2.sh defaults
    *[
        {
            "group": "lga",
            "task_id": f"ETTm2_512_{pl}",
            "cwd": LGA_ROOT,
            "log": f"lga/ETTm2_512_{pl}.log",
            "args": [
                "--is_training", "1",
                "--root_path", "./dataset/ETT-small/",
                "--data_path", "ETTm2.csv",
                "--model_id", f"ETTm2_512_{pl}",
                "--model", "PatchTST",
                "--data", "ETTm2",
                "--features", "M",
                "--seq_len", "512",
                "--pred_len", str(pl),
                "--e_layers", "3",
                "--enc_in", "7",
                "--des", "Exp",
                "--itr", "1",
                "--lga",
                "--eps", "1e-2",
                "--d_G", "64",
                "--batch_size", "128",
                "--train_epochs", "100",
                "--patience", "10",
                "--pct_start", "0.4",
                "--learning_rate", "1e-4",
            ],
        }
        for pl in [96, 192, 336, 720]
    ],
    # weather.sh defaults (pl720 uses eps=1)
    *[
        {
            "group": "lga",
            "task_id": f"weather_512_{pl}",
            "cwd": LGA_ROOT,
            "log": f"lga/weather_512_{pl}.log",
            "args": [
                "--is_training", "1",
                "--root_path", "./dataset/weather/",
                "--data_path", "weather.csv",
                "--model_id", f"weather_512_{pl}",
                "--model", "PatchTST",
                "--data", "custom",
                "--features", "M",
                "--seq_len", "512",
                "--pred_len", str(pl),
                "--e_layers", "3",
                "--enc_in", "21",
                "--des", "Exp",
                "--itr", "1",
                "--lga",
                "--eps", "1" if pl == 720 else "1e-2",
                "--d_G", "64",
                "--batch_size", "128",
                "--train_epochs", "100",
                "--patience", "10",
                "--schedulefree",
                "--learning_rate", "1e-3",
            ],
        }
        for pl in [96, 192, 336, 720]
    ],
    # electricity.sh defaults (pl720 uses eps=1)
    *[
        {
            "group": "lga",
            "task_id": f"electricity_512_{pl}",
            "cwd": LGA_ROOT,
            "log": f"lga/electricity_512_{pl}.log",
            "args": [
                "--is_training", "1",
                "--root_path", "./dataset/electricity/",
                "--data_path", "electricity.csv",
                "--model_id", f"electricity_512_{pl}",
                "--model", "PatchTST",
                "--data", "custom",
                "--features", "M",
                "--seq_len", "512",
                "--pred_len", str(pl),
                "--e_layers", "3",
                "--enc_in", "321",
                "--des", "Exp",
                "--itr", "1",
                "--lga",
                "--eps", "1" if pl == 720 else "1e-2",
                "--d_G", "64",
                "--batch_size", "32",
                "--train_epochs", "100",
                "--patience", "10",
                "--learning_rate", "1e-4",
            ],
        }
        for pl in [96, 192, 336, 720]
    ],
]


def traffic_cmd(pred_len: int) -> list[str]:
    """traffic.sh hyperparams + SACQ(SAR) + logcosh_scaled auto_mad."""
    model_id = f"traffic_512_{pred_len}_SARslc"
    return [
        PY, "-u", os.path.join(ROOT, "run_longExp.py"),
        "--random_seed", "2021",
        "--is_training", "1",
        "--root_path", os.path.join(ROOT, "dataset/"),
        "--data_path", "traffic.csv",
        "--model_id", model_id,
        "--model", "PatchTST",
        "--data", "traffic",
        "--features", "M",
        "--seq_len", "512",
        "--pred_len", str(pred_len),
        "--enc_in", "862",
        "--dec_in", "862",
        "--c_out", "862",
        "--e_layers", "3",
        "--n_heads", "16",
        "--d_model", "128",
        "--d_ff", "256",
        "--dropout", "0.2",
        "--fc_dropout", "0.2",
        "--head_dropout", "0.05",
        "--patch_len", "16",
        "--stride", "8",
        "--cross_attn_layers", "1",
        "--sacq_coarse_lr_ratio", "1.0",
        "--sacq_decoder_lr_ratio", "1.0",
        "--sacq_alpha_lr_ratio", "1.0",
        "--sacq_backbone_lr_ratio", "0.5",
        "--train_epochs", "50",
        "--patience", "15",
        "--lradj", "TST",
        "--pct_start", "0.2",
        "--itr", "1",
        "--batch_size", "4",
        "--learning_rate", "0.00005",
        "--num_workers", "4",
        "--head_type", "sacq_cross_attention",
        "--sacq_autoregressive", "1",
        "--sacq_alpha_mode", "learnable",
        "--short_setting", "1",
        "--des", "Exp",
        "--skip_visual", "1",
        "--dsacq_enable", "0",
        "--dsacq_mem_gate_enable", "0",
        "--dsacq_drift_gate_enable", "0",
        "--loss", "logcosh_scaled",
        "--logcosh_beta", "1.0",
        "--logcosh_beta_mode", "auto_mad",
    ]


def build_tasks(traffic: bool, lga: bool) -> list[dict]:
    tasks = []
    if traffic:
        for pl in TRAFFIC_PRED_LENS:
            tasks.append({
                "group": "traffic",
                "task_id": f"traffic_512_{pl}_SARslc",
                "cwd": ROOT,
                "log": f"traffic/traffic_512_{pl}_SARslc.log",
                "cmd": traffic_cmd(pl),
            })
    if lga:
        tasks.extend(LGA_JOBS)
    return tasks


def run_one(task: dict, gpu_id: int) -> dict:
    os.makedirs(os.path.dirname(os.path.join(LOG_ROOT, task["log"])), exist_ok=True)
    log_path = os.path.join(LOG_ROOT, task["log"])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if task["group"] == "lga":
        cmd = [PY, "-u", "run.py", *task["args"]]
    else:
        cmd = task["cmd"]

    t0 = time.time()
    with open(log_path, "w") as lf:
        lf.write(f"task_id={task['task_id']}\n")
        lf.write(f"gpu={gpu_id}\n")
        lf.write(f"group={task['group']}\n")
        lf.write(f"cwd={task['cwd']}\n")
        lf.write("cmd=" + " ".join(cmd) + "\n\n")
        lf.flush()
        proc = subprocess.run(
            cmd, cwd=task["cwd"], env=env, stdout=lf, stderr=subprocess.STDOUT, check=False
        )
    elapsed = time.time() - t0
    status = "ok" if proc.returncode == 0 else "fail"
    print(
        f"[GPU {gpu_id}] {task['task_id']} ({task['group']}) -> {status} "
        f"rc={proc.returncode} ({elapsed:.1f}s)",
        flush=True,
    )
    return {"task_id": task["task_id"], "group": task["group"], "gpu": gpu_id,
            "status": status, "return_code": proc.returncode, "elapsed_sec": round(elapsed, 2),
            "log_path": log_path}


def worker(gpu_id: int, q: Queue) -> None:
    while True:
        try:
            task = q.get_nowait()
        except Empty:
            break
        run_one(task, gpu_id)
        q.task_done()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--traffic", action="store_true", default=True)
    parser.add_argument("--no-traffic", action="store_true")
    parser.add_argument("--lga", action="store_true", default=True)
    parser.add_argument("--no-lga", action="store_true")
    args = parser.parse_args()
    do_traffic = args.traffic and not args.no_traffic
    do_lga = args.lga and not args.no_lga
    if do_lga and not os.path.isfile(os.path.join(LGA_ROOT, "run.py")):
        parser.error("LGA is not bundled. Set FCD_LGA_ROOT to your authorized external checkout; see docs/EXTERNAL_BASELINES.md, or pass --no-lga.")

    os.makedirs(LOG_ROOT, exist_ok=True)
    tasks = build_tasks(do_traffic, do_lga)
    q: Queue = Queue()
    for t in tasks:
        q.put(t)

    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    print(
        f"LOG_ROOT={LOG_ROOT}\n"
        f"total_tasks={len(tasks)} (traffic={sum(1 for t in tasks if t['group']=='traffic')}, "
        f"lga={sum(1 for t in tasks if t['group']=='lga')}) gpus={gpu_ids}",
        flush=True,
    )
    threads = [
        threading.Thread(target=worker, args=(gid, q), daemon=False) for gid in gpu_ids
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
