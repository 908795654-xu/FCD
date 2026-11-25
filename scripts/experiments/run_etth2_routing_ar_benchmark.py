"""ETTh2 sl512: SACQ S2 baseline vs routing coarse / causal+AR / full stack."""
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


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_tasks():
    variants = [
        {
            "tag": "S2",
            "des": "hS2",
            "sacq_coarse_init": "flatten",
            "sacq_future_causal": 0,
            "sacq_ar_verify": 0,
        },
        {
            "tag": "S2R",
            "des": "hS2R",
            "sacq_coarse_init": "routing_blend",
            "sacq_future_causal": 0,
            "sacq_ar_verify": 0,
        },
        {
            "tag": "S2CA",
            "des": "hS2CA",
            "sacq_coarse_init": "flatten",
            "sacq_future_causal": 1,
            "sacq_ar_verify": 1,
        },
        {
            "tag": "S2RCA",
            "des": "hS2RCA",
            "sacq_coarse_init": "routing_blend",
            "sacq_future_causal": 1,
            "sacq_ar_verify": 1,
        },
    ]
    tasks = []
    for pred_len in (96, 720):
        for v in variants:
            model_id = f"E2s512p{pred_len}{v['tag']}"
            setting = f"{model_id}_PatchTST_ETTh2_sl512_pl{pred_len}_{v['des']}_0"
            tasks.append(
                {
                    "task_id": f"E2s512p{pred_len}{v['tag']}",
                    "model_id": model_id,
                    "pred_len": pred_len,
                    "setting": setting,
                    **v,
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
        "ETTh2.csv",
        "--model_id",
        task["model_id"],
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
        str(task["pred_len"]),
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
        "--padding_patch",
        "end",
        "--revin",
        "1",
        "--affine",
        "0",
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
        "--sacq_scheme",
        "default",
        "--sacq_pe_mode",
        "relative_bias",
        "--sacq_pe_rel_buckets",
        "32",
        "--sacq_coarse_init",
        task["sacq_coarse_init"],
        "--sacq_routing_topk",
        str(args.routing_topk),
        "--sacq_routing_window",
        str(args.routing_window),
        "--sacq_future_causal",
        str(task["sacq_future_causal"]),
        "--sacq_ar_verify",
        str(task["sacq_ar_verify"]),
        "--sacq_alpha_mode",
        "learnable",
        "--sacq_alpha_value",
        "0.5",
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
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    t0 = time.time()
    with open(log_path, "w") as lf:
        lf.write(f"task_id={task['task_id']}\n")
        lf.write(f"gpu={gpu_id}\n")
        lf.write(f"setting={task['setting']}\n")
        lf.write("cmd=" + " ".join(cmd) + "\n\n")
        lf.flush()
        proc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, check=False)
    elapsed = time.time() - t0

    row = {
        "task_id": task["task_id"],
        "tag": task["tag"],
        "status": "ok" if proc.returncode == 0 else "fail",
        "return_code": proc.returncode,
        "gpu": gpu_id,
        "elapsed_sec": round(elapsed, 2),
        "pred_len": task["pred_len"],
        "sacq_coarse_init": task["sacq_coarse_init"],
        "sacq_future_causal": task["sacq_future_causal"],
        "sacq_ar_verify": task["sacq_ar_verify"],
        "setting": task["setting"],
        "log_path": log_path,
    }
    if proc.returncode == 0 and os.path.exists(metrics_path):
        with open(metrics_path) as f:
            m = json.load(f)
        for k in ("mse", "mae", "rmse"):
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


def summarize(csv_path: str, out_dir: str) -> None:
    if not os.path.exists(csv_path):
        return
    rows = list(csv.DictReader(open(csv_path)))
    ok = [r for r in rows if r.get("status") == "ok" and r.get("mse")]
    if not ok:
        return
    by_pl: dict = {}
    for r in ok:
        pl = int(r["pred_len"])
        by_pl.setdefault(pl, []).append(r)
    lines = ["# ETTh2 routing / causal+AR benchmark\n", "| pred_len | variant | MSE | MAE | vs S2 | |\n"]
    lines.append("|---|---|---:|---:|---:|:---:|\n")
    for pl in sorted(by_pl):
        base_mse = None
        for r in sorted(by_pl[pl], key=lambda x: x["tag"]):
            mse = float(r["mse"])
            mae = float(r["mae"])
            if r["tag"] == "S2":
                base_mse = mse
                delta = "—"
                win = "ref"
            elif base_mse is not None:
                pct = (mse - base_mse) / base_mse * 100
                delta = f"{pct:+.2f}%"
                win = "✓" if mse < base_mse else ""
            else:
                delta = "n/a"
                win = ""
            lines.append(
                f"| {pl} | {r['tag']} | {mse:.6f} | {mae:.6f} | {delta} | {win} |\n"
            )
    md_path = os.path.join(out_dir, "RESULTS.md")
    with open(md_path, "w") as f:
        f.writelines(lines)
    print(f"Wrote {md_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="ETTh2 SACQ routing + AR verify benchmark")
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--out_dir", type=str, default="logs/etth2_routing_ar_benchmark")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
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
    parser.add_argument("--routing_topk", type=int, default=16)
    parser.add_argument("--routing_window", type=int, default=8)
    parser.add_argument("--pred_lens", type=str, default="96,720", help="comma-separated pred lengths")
    parser.add_argument("--tags", type=str, default="", help="comma-separated tags to run (default all)")
    args = parser.parse_args()

    out_dir = args.out_dir
    ensure_dir(os.path.join(out_dir, "logs"))
    csv_path = os.path.join(out_dir, "summary.csv")
    header = [
        "task_id", "tag", "status", "return_code", "gpu", "elapsed_sec", "pred_len",
        "sacq_coarse_init", "sacq_future_causal", "sacq_ar_verify",
        "mse", "mae", "rmse", "setting", "log_path",
    ]

    want_pl = {int(x) for x in args.pred_lens.split(",") if x.strip()}
    want_tags = {x.strip() for x in args.tags.split(",") if x.strip()}
    tasks = [
        t for t in build_tasks()
        if t["pred_len"] in want_pl and (not want_tags or t["tag"] in want_tags)
    ]
    if not tasks:
        print("No tasks selected.", file=sys.stderr)
        sys.exit(1)

    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    q: Queue = Queue()
    for t in tasks:
        q.put(t)

    lock = threading.Lock()
    threads = [
        threading.Thread(target=worker, args=(gid, q, args, out_dir, csv_path, header, lock), daemon=True)
        for gid in gpus
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    summarize(csv_path, out_dir)


if __name__ == "__main__":
    main()
