"""ETTh2 S1 (dual_residual) micro-tuning sweep; 4-GPU queue + comparison report."""
import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from queue import Empty, Queue


# name -> extra CLI args (beyond shared SACQ dual_residual settings)
TUNE_VARIANTS = [
    {
        "tag": "v0",
        "des": "hS1v0",
        "dual": {"struct_gate": 0, "res_fuse": 0, "detach": 1, "scaffold_ln": 0},
        "extra": [],
    },
    {
        "tag": "v2",
        "des": "hS1v2",
        "dual": {"struct_gate": 1, "res_fuse": 1, "detach": 1, "scaffold_ln": 1},
        "extra": [],
    },
    {
        "tag": "v2b",
        "des": "hS1v2b",
        "dual": {"struct_gate": 1, "res_fuse": 1, "detach": 1, "scaffold_ln": 0},
        "extra": [],
    },
    {
        "tag": "v2sg",
        "des": "hS1v2sg",
        "dual": {"struct_gate": 1, "res_fuse": 0, "detach": 1, "scaffold_ln": 0},
        "extra": [],
    },
    {
        "tag": "v2nd",
        "des": "hS1v2nd",
        "dual": {"struct_gate": 1, "res_fuse": 1, "detach": 0, "scaffold_ln": 1},
        "extra": [],
    },
    {
        "tag": "v2d02",
        "des": "hS1v2d02",
        "dual": {"struct_gate": 1, "res_fuse": 1, "detach": 1, "scaffold_ln": 1},
        "extra": ["--dropout", "0.2", "--fc_dropout", "0.2"],
    },
    {
        "tag": "v2ca3",
        "des": "hS1v2ca3",
        "dual": {"struct_gate": 1, "res_fuse": 1, "detach": 1, "scaffold_ln": 1},
        "extra": ["--cross_attn_layers", "3"],
    },
    {
        "tag": "v2lr15",
        "des": "hS1v2lr15",
        "dual": {"struct_gate": 1, "res_fuse": 1, "detach": 1, "scaffold_ln": 1},
        "extra": ["--learning_rate", "0.00015"],
    },
]

BASELINE_ROWS = [
    ("F", 96, 0.272946834564209, 0.33634108304977417),
    ("S", 96, 0.27324235439300537, 0.33637535572052),
    ("SAR", 96, 0.27326735854148865, 0.3364093005657196),
    ("S1", 96, 0.27377378940582275, 0.33729827404022217),
    ("F", 720, 0.38041192293167114, 0.4240547716617584),
    ("S", 720, 0.38146206736564636, 0.4243292808532715),
    ("SAR", 720, 0.38158443570137024, 0.42439937591552734),
    ("S1", 720, 0.3789190351963043, 0.42178112268447876),
]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def build_tasks():
    tasks = []
    for pred_len in (96, 720):
        for var in TUNE_VARIANTS:
            model_id = f"E2s512p{pred_len}S1{var['tag']}"
            tasks.append(
                {
                    "task_id": f"E2s512p{pred_len}S1{var['tag']}",
                    "model_id": model_id,
                    "pred_len": pred_len,
                    "des": var["des"],
                    "tag": var["tag"],
                    "dual": var["dual"],
                    "extra": list(var["extra"]),
                    "setting": (
                        f"{model_id}_PatchTST_ETTh2_sl512_pl{pred_len}_{var['des']}_0"
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
    d = task["dual"]
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
        "dual_residual",
        "--sacq_dual_struct_gate",
        str(d["struct_gate"]),
        "--sacq_dual_res_fuse",
        str(d["res_fuse"]),
        "--sacq_dual_detach",
        str(d["detach"]),
        "--sacq_dual_scaffold_ln",
        str(d["scaffold_ln"]),
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
    cmd.extend(task["extra"])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    st = time.time()
    with open(log_path, "w") as lf:
        lf.write(f"task_id={task['task_id']}\n")
        lf.write(f"tag={task['tag']}\n")
        lf.write(f"dual={task['dual']}\n")
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
        "tag": task["tag"],
        "setting": task["setting"],
        "log_path": log_path,
        "struct_gate": d["struct_gate"],
        "res_fuse": d["res_fuse"],
        "detach": d["detach"],
        "scaffold_ln": d["scaffold_ln"],
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
                f"[GPU {gpu_id}] {row['task_id']} tag={row['tag']} -> "
                f"{row['status']} mse={row.get('mse', 'n/a')}",
                flush=True,
            )
        q.task_done()


def write_report(out_dir, summary_csv):
    rows = []
    for label, pl, mse, mae in BASELINE_ROWS:
        rows.append({"pred_len": int(pl), "tag": label, "mse": mse, "mae": mae, "kind": "baseline"})
    if os.path.exists(summary_csv):
        with open(summary_csv, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("mse"):
                    rows.append(
                        {
                            "pred_len": int(r["pred_len"]),
                            "tag": r["tag"],
                            "mse": float(r["mse"]),
                            "mae": float(r["mae"]),
                            "kind": "tune",
                        }
                    )
    best_path = os.path.join(out_dir, "best.json")
    report_path = os.path.join(out_dir, "RESULTS.md")
    best = {}
    lines = [
        "# ETTh2 S1 微调结果",
        "",
        "条件: sl=512, epoch=50, patience=20, PatchTST+SACQ dual_residual",
        "",
    ]
    for pl in (96, 720):
        sub = [x for x in rows if x["pred_len"] == pl]
        sub.sort(key=lambda x: x["mse"])
        best[pl] = {"tag": sub[0]["tag"], "mse": sub[0]["mse"], "mae": sub[0]["mae"]}
        lines.append(f"## pred_len={pl}")
        lines.append("")
        lines.append("| Rank | Method | MSE | MAE |")
        lines.append("|------|--------|-----|-----|")
        for i, x in enumerate(sub, 1):
            lines.append(f"| {i} | {x['tag']} | {x['mse']:.6f} | {x['mae']:.6f} |")
        lines.append("")
        lines.append(
            f"**Best**: `{sub[0]['tag']}` MSE={sub[0]['mse']:.6f} "
            f"(vs S1-v0 baseline {next((b['mse'] for b in BASELINE_ROWS if b[0]=='S1' and b[1]==pl), 0):.6f})"
        )
        lines.append("")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    with open(best_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"wrote {report_path} and {best_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--out_dir", type=str, default="logs/etth2_s1_tune")
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
        "tag",
        "setting",
        "log_path",
        "struct_gate",
        "res_fuse",
        "detach",
        "scaffold_ln",
        "mse",
        "mae",
    ]
    tasks = build_tasks()
    print(f"total_tasks={len(tasks)}", flush=True)
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
    write_report(out_dir, csv_path)
    print("all tasks finished", flush=True)


if __name__ == "__main__":
    main()
