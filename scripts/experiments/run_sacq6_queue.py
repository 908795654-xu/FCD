import argparse
import csv
import json
import os
import subprocess
import threading
import time
from queue import Empty, Queue


PRED_LENS = [96, 192, 336, 720]
SEQ_LENS = [96, 512]
MODELS = [
    ("Linear", "Lin"),
    ("ARIMA", "Ari"),
    ("ETS", "Ets"),
    ("Theta", "The"),
    ("Prophet", "Pro"),
    ("NBEATS", "Nbt"),
]
HEADS = [
    ("flatten", "F"),
    ("sacq_cross_attention", "S"),
]
DATASETS = [
    # (data, data_path, enc_in, freq, short_tag)
    ("ETTh1", "ETTh1.csv", 7, "h", "H1"),
    ("ETTm2", "ETTm2.csv", 7, "t", "M2"),
    ("weather", "weather.csv", 21, "h", "W"),
]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def build_setting(task):
    # must match short_setting format in run_longExp.py
    return (
        f"{task['model_id']}_{task['model']}_{task['data']}_"
        f"sl{task['seq_len']}_pl{task['pred_len']}_{task['des']}_0"
    )


def task_id(task):
    return (
        f"{task['dtag']}s{task['seq_len']}p{task['pred_len']}"
        f"m{task['mtag']}h{task['htag']}"
    )


def write_csv_row(csv_path, header, row):
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def build_tasks():
    tasks = []
    for data, data_path, enc_in, freq, dtag in DATASETS:
        for seq_len in SEQ_LENS:
            for pred_len in PRED_LENS:
                for model, mtag in MODELS:
                    for head_type, htag in HEADS:
                        des = f"h{htag}"
                        model_id = f"{dtag}{seq_len}p{pred_len}{mtag}{htag}"
                        tasks.append(
                            {
                                "data": data,
                                "data_path": data_path,
                                "enc_in": enc_in,
                                "freq": freq,
                                "dtag": dtag,
                                "mtag": mtag,
                                "htag": htag,
                                "seq_len": seq_len,
                                "pred_len": pred_len,
                                "model": model,
                                "head_type": head_type,
                                "des": des,
                                "model_id": model_id,
                            }
                        )
    return tasks


def run_one_task(task, gpu_id, args, out_root):
    tid = task_id(task)
    log_path = os.path.join(out_root, "logs", f"{tid}.log")
    setting = build_setting(task)
    metrics_path = os.path.join("results", setting, "metrics.json")

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
    start = time.time()
    with open(log_path, "w") as lf:
        lf.write(f"task_id={tid}\n")
        lf.write(f"gpu={gpu_id}\n")
        lf.write(f"setting={setting}\n")
        lf.write("cmd=" + " ".join(cmd) + "\n\n")
        lf.flush()
        proc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, check=False)
    elapsed = time.time() - start

    result = {
        "task_id": tid,
        "status": "ok" if proc.returncode == 0 else "fail",
        "return_code": proc.returncode,
        "gpu": gpu_id,
        "elapsed_sec": round(elapsed, 2),
        "dataset": task["data"],
        "seq_len": task["seq_len"],
        "pred_len": task["pred_len"],
        "model": task["model"],
        "head_type": task["head_type"],
        "setting": setting,
        "log_path": log_path,
    }

    if proc.returncode == 0 and os.path.exists(metrics_path):
        with open(metrics_path, "r") as mf:
            metrics = json.load(mf)
        for k in [
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
        ]:
            result[k] = metrics.get(k)
    return result


def worker(gpu_id, task_q, csv_path, csv_header, args, out_root, lock):
    while True:
        try:
            task = task_q.get_nowait()
        except Empty:
            break
        res = run_one_task(task, gpu_id=gpu_id, args=args, out_root=out_root)
        with lock:
            write_csv_row(csv_path, csv_header, res)
            print(
                f"[GPU {gpu_id}] {res['task_id']} -> {res['status']} "
                f"(rc={res['return_code']}, {res['elapsed_sec']}s)",
                flush=True,
            )
        task_q.task_done()


def main():
    parser = argparse.ArgumentParser(description="4-GPU queue runner for SACQ baseline suite")
    parser.add_argument("--gpus", type=str, default="0,1,2,3", help="comma-separated gpu ids")
    parser.add_argument("--out_dir", type=str, default="logs/sacq6_suite", help="base output folder")
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

    out_root = args.out_dir
    ensure_dir(out_root)
    ensure_dir(os.path.join(out_root, "logs"))
    csv_path = os.path.join(out_root, "summary.csv")

    tasks = build_tasks()
    task_q = Queue()
    for t in tasks:
        task_q.put(t)

    csv_header = [
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
    lock = threading.Lock()

    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    print(f"total_tasks={len(tasks)} gpus={gpu_ids}", flush=True)
    threads = []
    for gid in gpu_ids:
        th = threading.Thread(
            target=worker,
            kwargs={
                "gpu_id": gid,
                "task_q": task_q,
                "csv_path": csv_path,
                "csv_header": csv_header,
                "args": args,
                "out_root": out_root,
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

