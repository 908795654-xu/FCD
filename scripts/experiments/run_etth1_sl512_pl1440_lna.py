import argparse
import csv
import json
import os
import subprocess
import threading
import time
from queue import Empty, Queue


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def build_tasks():
    models = [("Linear", "Lin"), ("NBEATS", "Nbt"), ("ARIMA", "Ari")]
    heads = [("flatten", "F"), ("sacq_cross_attention", "S")]
    tasks = []
    for model, mtag in models:
        for head_type, htag in heads:
            model_id = f"H15121440{mtag}{htag}"
            des = f"h{htag}"
            task_id = f"H1s512p1440m{mtag}h{htag}"
            setting = f"{model_id}_{model}_ETTh1_sl512_pl1440_{des}_0"
            tasks.append(
                {
                    "task_id": task_id,
                    "model": model,
                    "head_type": head_type,
                    "model_id": model_id,
                    "des": des,
                    "setting": setting,
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
    cmd = [
        "python",
        "-u",
        "run_longExp.py",
        "--is_training",
        "1",
        "--random_seed",
        str(args.seed),
        "--root_path",
        "./dataset/",
        "--data_path",
        "ETTh1.csv",
        "--model_id",
        task["model_id"],
        "--model",
        task["model"],
        "--data",
        "ETTh1",
        "--features",
        "M",
        "--freq",
        "h",
        "--seq_len",
        "512",
        "--pred_len",
        "1440",
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
        task["head_type"],
        "--short_setting",
        "1",
        "--des",
        task["des"],
        "--skip_visual",
        "1",
    ]
    if task["head_type"] == "sacq_cross_attention":
        cmd += ["--dsacq_enable", "0", "--dsacq_mem_gate_enable", "0", "--dsacq_drift_gate_enable", "0"]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    st = time.time()
    with open(log_path, "w") as lf:
        lf.write(f"task_id={task['task_id']}\n")
        lf.write(f"gpu={gpu_id}\n")
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
        "dataset": "ETTh1",
        "seq_len": 512,
        "pred_len": 1440,
        "model": task["model"],
        "head_type": task["head_type"],
        "setting": task["setting"],
        "log_path": log_path,
    }
    if proc.returncode == 0 and os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            m = json.load(f)
        for k in ["mse", "mae", "rse", "rmse", "mape", "mspe", "corr", "smape", "mase", "ql50", "pwmae"]:
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
                f"(rc={row['return_code']}, {row['elapsed_sec']}s)",
                flush=True,
            )
        q.task_done()


def main():
    parser = argparse.ArgumentParser(description="ETTh1 sl512 pl1440 run for Linear/NBEATS/ARIMA + F/S")
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--out_dir", type=str, default="logs/sacq_target_1440")
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
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--fc_dropout", type=float, default=0.2)
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
        "task_id",
        "status",
        "return_code",
        "gpu",
        "elapsed_sec",
        "dataset",
        "seq_len",
        "pred_len",
        "model",
        "head_type",
        "setting",
        "log_path",
        "mse",
        "mae",
        "rse",
        "rmse",
        "mape",
        "mspe",
        "corr",
        "smape",
        "mase",
        "ql50",
        "pwmae",
    ]
    print(f"total_tasks={len(tasks)}", flush=True)
    lock = threading.Lock()
    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    threads = []
    for gid in gpu_ids:
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
            daemon=False,
        )
        threads.append(th)
        th.start()
    for th in threads:
        th.join()
    print("all tasks finished", flush=True)


if __name__ == "__main__":
    main()

