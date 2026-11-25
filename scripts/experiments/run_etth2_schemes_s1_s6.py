"""ETTh2: SACQ scheme1 (dual_residual) & scheme6 (template_ar); compare with F/S/SAR baselines."""
import argparse
import csv
import json
import os
import subprocess
import sys
import threading
from queue import Empty, Queue


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def build_tasks():
    tasks = []
    for pred_len in (96, 720):
        for variant, des, scheme in (
            ("S1", "hS1", "dual_residual"),
            ("S6", "hS6", "template_ar"),
        ):
            model_id = f"E2s512p{pred_len}{variant}"
            tasks.append(
                {
                    "task_id": f"E2s512p{pred_len}mPtf{variant}",
                    "model_id": model_id,
                    "pred_len": pred_len,
                    "des": des,
                    "sacq_scheme": scheme,
                    "setting": (
                        f"{model_id}_PatchTST_ETTh2_sl512_pl{pred_len}_{des}_0"
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
        task["sacq_scheme"],
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
    import time

    st = time.time()
    with open(log_path, "w") as lf:
        lf.write(f"task_id={task['task_id']}\nscheme={task['sacq_scheme']}\n")
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
        "pred_len": task["pred_len"],
        "variant": task["des"],
        "sacq_scheme": task["sacq_scheme"],
        "setting": task["setting"],
        "log_path": log_path,
    }
    if proc.returncode == 0 and os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            m = json.load(f)
        row["mse"] = m.get("mse")
        row["mae"] = m.get("mae")
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
                f"[GPU {gpu_id}] {row['task_id']} {row['sacq_scheme']} -> "
                f"{row['status']} mse={row.get('mse', 'n/a')}",
                flush=True,
            )
        q.task_done()


def write_comparison(out_dir, baseline_csv):
    new_csv = os.path.join(out_dir, "summary.csv")
    cmp_path = os.path.join(out_dir, "comparison.md")
    rows = []
    if os.path.exists(baseline_csv):
        with open(baseline_csv, newline="") as f:
            for r in csv.DictReader(f):
                pl = r.get("pred_len")
                var = r.get("variant") or r.get("des", "")
                if var.startswith("h"):
                    var = var[1:]
                if var == "F":
                    label = "F"
                elif var == "S":
                    label = "S"
                elif var == "SAR":
                    label = "SAR"
                else:
                    label = var
                rows.append(
                    {
                        "pred_len": pl,
                        "label": label,
                        "mse": float(r["mse"]) if r.get("mse") else None,
                        "mae": float(r["mae"]) if r.get("mae") else None,
                        "source": "baseline",
                    }
                )
    if os.path.exists(new_csv):
        with open(new_csv, newline="") as f:
            for r in csv.DictReader(f):
                label = "S1" if r["sacq_scheme"] == "dual_residual" else "S6"
                rows.append(
                    {
                        "pred_len": r["pred_len"],
                        "label": label,
                        "mse": float(r["mse"]) if r.get("mse") else None,
                        "mae": float(r["mae"]) if r.get("mae") else None,
                        "source": "new",
                    }
                )
    with open(cmp_path, "w") as f:
        f.write("# ETTh2 sl512 comparison (MSE / MAE)\n\n")
        f.write(f"Baseline: `{baseline_csv}`\n\n")
        for pl in ("96", "720"):
            f.write(f"## pred_len={pl}\n\n")
            f.write("| Method | MSE | MAE |\n|--------|-----|-----|\n")
            sub = [x for x in rows if str(x["pred_len"]) == pl and x["mse"] is not None]
            sub.sort(key=lambda x: x["mse"])
            for x in sub:
                f.write(f"| {x['label']} | {x['mse']:.6f} | {x['mae']:.6f} |\n")
            f.write("\n")
    print(f"wrote {cmp_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--out_dir", type=str, default="logs/etth2_schemes_s1_s6")
    parser.add_argument(
        "--baseline_csv",
        type=str,
        default="logs/etth2_f_s_ar_sl512/summary.csv",
    )
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
    header = [
        "task_id",
        "status",
        "return_code",
        "gpu",
        "elapsed_sec",
        "pred_len",
        "variant",
        "sacq_scheme",
        "setting",
        "log_path",
        "mse",
        "mae",
    ]
    tasks = build_tasks()
    q = Queue()
    for t in tasks:
        q.put(t)
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
    write_comparison(out_dir, args.baseline_csv)


if __name__ == "__main__":
    main()
