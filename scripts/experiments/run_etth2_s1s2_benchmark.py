"""ETTh2: S1 (dual_residual) + S2 (relative_bias), alone and combined."""
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

DATA = "ETTh2"
DATA_PATH = "ETTh2.csv"
ENC_IN = 7
FREQ = "h"
SEQ_LEN = 512
PRED_LENS = [96, 192, 336, 720]

# tag, des, scheme, pe_mode, extra dual_residual flags
VARIANTS = [
    ("S1", "hS1", "dual_residual", "default", True),
    ("S2", "hS2", "default", "relative_bias", False),
    ("S12", "hS12", "dual_residual", "relative_bias", True),
]

S1_DUAL_FLAGS = [
    "--sacq_dual_struct_gate", "0",
    "--sacq_dual_res_fuse", "0",
    "--sacq_dual_detach", "1",
    "--sacq_dual_scaffold_ln", "0",
]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def build_tasks():
    tasks = []
    for pred_len in PRED_LENS:
        for tag, des, scheme, pe_mode, use_dual_flags in VARIANTS:
            model_id = f"H2s512p{pred_len}{tag}"
            tasks.append(
                {
                    "task_id": f"H2s512p{pred_len}{tag}",
                    "pred_len": pred_len,
                    "tag": tag,
                    "des": des,
                    "scheme": scheme,
                    "pe_mode": pe_mode,
                    "use_dual_flags": use_dual_flags,
                    "model_id": model_id,
                    "setting": f"{model_id}_PatchTST_{DATA}_sl{SEQ_LEN}_pl{pred_len}_{des}_0",
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
    metrics_path = os.path.join(PROJECT_ROOT, "results", task["setting"], "metrics.json")
    py = os.environ.get("PYTHON", sys.executable)
    cmd = [
        py, "-u", os.path.join(PROJECT_ROOT, "run_longExp.py"),
        "--is_training", "1",
        "--random_seed", str(args.seed),
        "--root_path", "./dataset/",
        "--data_path", DATA_PATH,
        "--model_id", task["model_id"],
        "--model", "PatchTST",
        "--data", DATA,
        "--features", "M",
        "--freq", FREQ,
        "--seq_len", str(SEQ_LEN),
        "--pred_len", str(task["pred_len"]),
        "--enc_in", str(ENC_IN),
        "--dec_in", str(ENC_IN),
        "--c_out", str(ENC_IN),
        "--e_layers", str(args.e_layers),
        "--n_heads", str(args.n_heads),
        "--d_model", str(args.d_model),
        "--d_ff", str(args.d_ff),
        "--dropout", str(args.dropout),
        "--fc_dropout", str(args.fc_dropout),
        "--head_dropout", "0",
        "--patch_len", str(args.patch_len),
        "--stride", str(args.stride),
        "--cross_attn_layers", "2",
        "--train_epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.lr),
        "--itr", "1",
        "--num_workers", str(args.num_workers),
        "--head_type", "sacq_cross_attention",
        "--sacq_scheme", task["scheme"],
        "--sacq_pe_mode", task["pe_mode"],
        "--sacq_alpha_mode", "learnable",
        "--short_setting", "1",
        "--des", task["des"],
        "--skip_visual", "1",
        "--dsacq_enable", "0",
        "--dsacq_mem_gate_enable", "0",
        "--dsacq_drift_gate_enable", "0",
    ]
    if task["use_dual_flags"]:
        cmd.extend(S1_DUAL_FLAGS)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    st = time.time()
    with open(log_path, "w") as lf:
        lf.write(f"task_id={task['task_id']}\nscheme={task['scheme']}\npe_mode={task['pe_mode']}\n")
        lf.write(f"setting={task['setting']}\n")
        lf.write("cmd=" + " ".join(cmd) + "\n\n")
        lf.flush()
        proc = subprocess.run(
            cmd, cwd=PROJECT_ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT, check=False,
        )
    elapsed = time.time() - st
    row = {
        "task_id": task["task_id"],
        "tag": task["tag"],
        "scheme": task["scheme"],
        "pe_mode": task["pe_mode"],
        "status": "ok" if proc.returncode == 0 else "fail",
        "return_code": proc.returncode,
        "gpu": gpu_id,
        "elapsed_sec": round(elapsed, 2),
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
                f"[GPU {gpu_id}] {row['task_id']} {row['tag']} -> {row['status']} "
                f"mse={row.get('mse', 'n/a')} ({row['elapsed_sec']}s)",
                flush=True,
            )
        q.task_done()


def write_results_md(out_dir, summary_csv):
    if not os.path.exists(summary_csv):
        return
    rows = list(csv.DictReader(open(summary_csv)))
    path = os.path.join(out_dir, "RESULTS.md")
    ref_s1 = {96: 0.273774, 192: 0.351023, 336: 0.334216, 720: 0.378919}
    ref_s2 = {96: 0.272716, 192: 0.339761, 336: 0.328972, 720: 0.378936}
    with open(path, "w") as f:
        f.write("# ETTh2 S1 + S2 benchmark (sl=512)\n\n")
        f.write("| Tag | scheme | pe_mode | 说明 |\n|-----|--------|---------|------|\n")
        f.write("| S1 | dual_residual | default | 双分支结构 |\n")
        f.write("| S2 | default | relative_bias | 相对时间偏置 |\n")
        f.write("| **S12** | dual_residual | relative_bias | **S1+S2 组合** |\n\n")
        f.write("历史参考（前次单独实验）: S1-v0 / S2-PE benchmark\n\n")
        for pl in PRED_LENS:
            f.write(f"## pred_len={pl}\n\n")
            f.write("| variant | MSE | MAE | Δ vs 历史 S1 | Δ vs 历史 S2 |\n")
            f.write("|---------|-----|-----|--------------|--------------|\n")
            for tag, _, _, _, _ in VARIANTS:
                hit = next((r for r in rows if r.get("task_id") == f"H2s512p{pl}{tag}"), None)
                if hit and hit.get("mse"):
                    mse = float(hit["mse"])
                    mae = float(hit["mae"])
                    d1 = (mse / ref_s1[pl] - 1) * 100
                    d2 = (mse / ref_s2[pl] - 1) * 100
                    f.write(
                        f"| {tag} | {mse:.6f} | {mae:.6f} | {d1:+.2f}% | {d2:+.2f}% |\n"
                    )
                else:
                    f.write(f"| {tag} | — | — | — | — |\n")
            sub = [
                r for r in rows
                if str(r.get("pred_len")) == str(pl) and r.get("mse")
            ]
            if sub:
                best = min(sub, key=lambda r: float(r["mse"]))
                f.write(f"\n**Best**: {best['tag']} MSE={float(best['mse']):.6f}\n\n")
    print(f"wrote {path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--out_dir", type=str, default="logs/etth2_s1s2_benchmark")
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

    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "logs"))
    csv_path = os.path.join(out_dir, "summary.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)
    tasks = build_tasks()
    q = Queue()
    for t in tasks:
        q.put(t)
    header = [
        "task_id", "tag", "scheme", "pe_mode", "status", "return_code", "gpu",
        "elapsed_sec", "pred_len", "setting", "log_path",
        "mse", "mae", "rse", "rmse", "mape", "mspe", "corr",
    ]
    print(f"total_tasks={len(tasks)} gpus={args.gpus}", flush=True)
    lock = threading.Lock()
    threads = []
    for gid in [int(x) for x in args.gpus.split(",") if x.strip()]:
        th = threading.Thread(
            target=worker,
            kwargs={
                "gpu_id": gid, "q": q, "args": args, "out_dir": out_dir,
                "csv_path": csv_path, "header": header, "lock": lock,
            },
        )
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    write_results_md(out_dir, csv_path)
    print("all tasks finished", flush=True)


if __name__ == "__main__":
    main()
