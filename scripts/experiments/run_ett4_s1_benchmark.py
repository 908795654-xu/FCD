"""ETTh1/ETTh2/ETTm1/ETTm2: PatchTST + S1 (dual_residual v0), 4 pred_lens, 4-GPU queue."""
import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from queue import Empty, Queue


DATASETS = [
    ("ETTh1", "ETTh1.csv", 7, "h", "H1"),
    ("ETTh2", "ETTh2.csv", 7, "h", "H2"),
    ("ETTm1", "ETTm1.csv", 7, "t", "M1"),
    ("ETTm2", "ETTm2.csv", 7, "t", "M2"),
]
PRED_LENS = [96, 192, 336, 720]
SEQ_LEN = 512


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def build_tasks():
    tasks = []
    for data, data_path, enc_in, freq, dtag in DATASETS:
        for pred_len in PRED_LENS:
            model_id = f"{dtag}s512p{pred_len}S1"
            des = "hS1"
            tasks.append(
                {
                    "task_id": f"{dtag}s512p{pred_len}S1",
                    "data": data,
                    "data_path": data_path,
                    "enc_in": enc_in,
                    "freq": freq,
                    "pred_len": pred_len,
                    "model_id": model_id,
                    "des": des,
                    "setting": (
                        f"{model_id}_PatchTST_{data}_sl{SEQ_LEN}_pl{pred_len}_{des}_0"
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


def run_one(task, gpu_id, args, out_dir):
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
        task["data_path"],
        "--model_id",
        task["model_id"],
        "--model",
        "PatchTST",
        "--data",
        task["data"],
        "--features",
        "M",
        "--freq",
        task["freq"],
        "--seq_len",
        str(SEQ_LEN),
        "--pred_len",
        str(task["pred_len"]),
        "--enc_in",
        str(task["enc_in"]),
        "--dec_in",
        str(task["enc_in"]),
        "--c_out",
        str(task["enc_in"]),
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
        "--itr",
        "1",
        "--num_workers",
        str(args.num_workers),
        "--head_type",
        "sacq_cross_attention",
        "--sacq_scheme",
        "dual_residual",
        "--sacq_dual_struct_gate",
        "0",
        "--sacq_dual_res_fuse",
        "0",
        "--sacq_dual_detach",
        "1",
        "--sacq_dual_scaffold_ln",
        "0",
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
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    st = time.time()
    with open(log_path, "w") as lf:
        lf.write(f"task_id={task['task_id']}\n")
        lf.write(f"setting={task['setting']}\n")
        lf.write("cmd=" + " ".join(cmd) + "\n\n")
        lf.flush()
        proc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, check=False)
    elapsed = time.time() - st
    row = {
        "task_id": task["task_id"],
        "status": "ok" if proc.returncode == 0 else "fail",
        "return_code": proc.returncode,
        "gpu": gpu_id,
        "elapsed_sec": round(elapsed, 2),
        "data": task["data"],
        "pred_len": task["pred_len"],
        "setting": task["setting"],
        "log_path": log_path,
    }
    if proc.returncode == 0 and os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            m = json.load(f)
        for k in ("mse", "mae", "rse", "rmse", "mape", "mspe", "corr"):
            row[k] = m.get(k)
    return row


def worker(gpu_id, q, args, out_dir, csv_path, header, lock):
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


def write_matrix(out_dir, summary_csv):
    if not os.path.exists(summary_csv):
        return
    rows = list(csv.DictReader(open(summary_csv)))
    path = os.path.join(out_dir, "RESULTS.md")
    with open(path, "w") as f:
        f.write("# ETT×4 S1 benchmark (sl=512, S1-v0 dual_residual)\n\n")
        for data, _, _, _, dtag in DATASETS:
            f.write(f"## {data}\n\n")
            f.write("| pred_len | MSE | MAE |\n|----------|-----|-----|\n")
            for pl in PRED_LENS:
                tid = f"{dtag}s512p{pl}S1"
                hit = next((r for r in rows if r.get("task_id") == tid and r.get("mse")), None)
                if hit:
                    f.write(f"| {pl} | {float(hit['mse']):.6f} | {float(hit['mae']):.6f} |\n")
                else:
                    f.write(f"| {pl} | — | — |\n")
            f.write("\n")
    print(f"wrote {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="ETT4 S1 benchmark queue")
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--out_dir", type=str, default="logs/ett4_s1_benchmark")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0001)
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

    out_dir = args.out_dir
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "logs"))
    csv_path = os.path.join(out_dir, "summary.csv")
    tasks = build_tasks()
    q = Queue()
    for t in tasks:
        q.put(t)
    header = [
        "task_id", "status", "return_code", "gpu", "elapsed_sec",
        "data", "pred_len", "setting", "log_path",
        "mse", "mae", "rse", "rmse", "mape", "mspe", "corr",
    ]
    print(f"total_tasks={len(tasks)} gpus={args.gpus}", flush=True)
    lock = threading.Lock()
    threads = []
    for gid in [int(x) for x in args.gpus.split(",") if x.strip()]:
        th = threading.Thread(
            target=worker,
            kwargs={
                "gpu_id": gid,
                "q": q,
                "args": args,
                "out_dir": out_dir,
                "csv_path": csv_path,
                "header": header,
                "lock": lock,
            },
        )
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    write_matrix(out_dir, csv_path)
    print("all tasks finished", flush=True)


if __name__ == "__main__":
    main()
