"""ETTm1: two-stage Flatten -> SAR (SACQ) with coarse-LR sweep.

Goal: rerun original `M1*SARlc` experiments, but with:
  Stage 1: train `head_type=flatten` for each pred_len (loss=logcosh).
  Stage 2: init from Stage 1 checkpoint, set `bootstrap_flatten_head=1`,
           train `head_type=sacq_cross_attention` (SAR: autoregressive=1)
           while sweeping `sacq_coarse_lr_ratio` in {0.1, 0.5, 1, 0.01}.

All other core configurations are kept aligned with `logs/ettm1_sar_sl512`
(i.e., PatchTST, sl=512, batch=128, epochs=50, patience=20, lr=1e-4, etc).
"""

from __future__ import annotations

import argparse
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

DATA = "ETTm1"
DATA_PATH = "ETTm1.csv"
ENC_IN = 7
FREQ = "t"
SEQ_LEN = 512

# Original experiments: pl=336, 720 with M1s512p{pl}SARlc.
DEFAULT_PRED_LENS = [336, 720]

# LR multipliers for the coarse (Flatten) branch in Stage 2.
COARSE_LR_VARIANTS = [
    ("lr01", 0.1),
    ("lr05", 0.5),
    ("lr1", 1.0),
    ("lr001", 0.01),
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def short_setting(model_id: str, des: str, pred_len: int, ii: int = 0) -> str:
    # Must match run_longExp when --short_setting=1:
    # f'{args.model_id}_{args.model}_{args.data}_sl{args.seq_len}_pl{args.pred_len}_{args.des}_{ii}'
    return f"{model_id}_PatchTST_{DATA}_sl{SEQ_LEN}_pl{pred_len}_{des}_{ii}"


def stage1_model_id(pred_len: int) -> str:
    # Unique naming to avoid collisions with other experiments.
    return f"M1s{SEQ_LEN}p{pred_len}Ftw"


def stage2_model_id(pred_len: int, ratio_tag: str) -> str:
    return f"M1s{SEQ_LEN}p{pred_len}SARlcTw_{ratio_tag}"


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
        "logcosh",
    ]


def run_one(cmd: list[str], gpu_id: int, log_path: str) -> int:
    ensure_dir(os.path.dirname(log_path))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    t0 = time.time()
    with open(log_path, "w") as lf:
        lf.write("cmd=" + " ".join(cmd) + "\n\n")
        lf.flush()
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    print(f"{os.path.basename(log_path)} done rc={proc.returncode} elapsed_sec={elapsed:.1f}", flush=True)
    return proc.returncode


def worker(gpu_id: int, q: Queue, out_dir: str, lock: threading.Lock) -> None:
    while True:
        try:
            task = q.get_nowait()
        except Empty:
            break

        rc = run_one(task["cmd"], gpu_id, task["log_path"])
        with lock:
            # Minimal marker to quickly check completion.
            with open(task["marker_path"], "w") as f:
                f.write(f"rc={rc}\n")

        q.task_done()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--out_dir", type=str, default="logs/ettm1_twostage_sar_logcosh_lr_sweep")
    parser.add_argument("--seed", type=int, default=2021)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--e_layers", type=int, default=3)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--fc_dropout", type=float, default=0.3)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)

    parser.add_argument("--pred_lens", type=str, default="336,720", help="comma-separated")
    args = parser.parse_args()

    pred_lens = [int(x) for x in args.pred_lens.split(",") if x.strip()]

    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    log_dir = os.path.join(out_dir, "logs")
    ensure_dir(log_dir)

    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    lock = threading.Lock()

    # -------------------------
    # Phase 1: Stage1 Flatten
    # -------------------------
    s1_tasks: list[dict] = []
    for pl in pred_lens:
        s1_mid = stage1_model_id(pl)
        s1_des = "hFtw"
        cmd = build_base_cmd(args, model_id=s1_mid, des=s1_des, pred_len=pl)
        cmd += ["--head_type", "flatten"]

        task_id = f"s1_pl{pl}"
        log_path = os.path.join(log_dir, f"{task_id}.log")
        marker_path = os.path.join(log_dir, f"{task_id}.done")
        s1_tasks.append({"task_id": task_id, "cmd": cmd, "log_path": log_path, "marker_path": marker_path})

    print(f"[Stage1] total_tasks={len(s1_tasks)} gpus={args.gpus}", flush=True)
    q: Queue = Queue()
    for t in s1_tasks:
        q.put(t)
    threads = [
        threading.Thread(target=worker, args=(gid, q, out_dir, lock), daemon=False) for gid in gpu_ids
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    print("[Stage1] all tasks finished", flush=True)

    # -------------------------
    # Phase 2: Stage2 SACQ SAR
    # -------------------------
    s2_tasks: list[dict] = []
    for pl in pred_lens:
        s1_mid = stage1_model_id(pl)
        s1_des = "hFtw"
        s1_setting = short_setting(s1_mid, s1_des, pl)

        for ratio_tag, ratio in COARSE_LR_VARIANTS:
            s2_mid = stage2_model_id(pl, ratio_tag)
            s2_des = "hSARlc"

            cmd = build_base_cmd(args, model_id=s2_mid, des=s2_des, pred_len=pl)
            cmd += [
                "--head_type",
                "sacq_cross_attention",
                "--cross_attn_layers",
                "2",
                "--sacq_autoregressive",
                "1",
                "--sacq_scheme",
                "default",
                "--sacq_pe_mode",
                "default",
                "--sacq_alpha_mode",
                "learnable",
                "--sacq_coarse_lr_ratio",
                str(ratio),
                "--init_checkpoint_setting",
                s1_setting,
                "--bootstrap_flatten_head",
                "1",
            ]

            # task_id name for log/marker
            task_id = f"s2_pl{pl}_{ratio_tag}"
            log_path = os.path.join(log_dir, f"{task_id}.log")
            marker_path = os.path.join(log_dir, f"{task_id}.done")
            s2_tasks.append(
                {
                    "task_id": task_id,
                    "cmd": cmd,
                    "log_path": log_path,
                    "marker_path": marker_path,
                }
            )

    q2: Queue = Queue()
    for t in s2_tasks:
        q2.put(t)
    print(f"[Stage2] total_tasks={len(s2_tasks)} gpus={args.gpus}", flush=True)
    threads = [
        threading.Thread(target=worker, args=(gid, q2, out_dir, lock), daemon=False) for gid in gpu_ids
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    print("[Stage2] all tasks finished", flush=True)


if __name__ == "__main__":
    main()

