"""Per pred_len best freeze-SACQ config + higher alpha LR ratios (4-GPU queue)."""
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
SEQ_LEN = 512
PRED_LENS = [96, 192, 336, 720]

S2_EPOCHS = 20
S2_PATIENCE = 10
S2_ALPHA_INIT = 0.0001
ALPHA_LR_RATIOS = [2.0, 5.0, 10.0]

# Best SACQ from logs/etth2_twostage_freeze/RESULTS.md
BEST_BY_PL = {
    96: {
        "base_tag": "Zc001",
        "base_des": "hZc001",
        "coarse_ratio": 0.01,
        "freeze_coarse": 0,
        "prev_setting": "TwS2Zc001s512p96_PatchTST_ETTh2_sl512_pl96_hZc001_0",
        "prev_mse": 0.274579,
    },
    192: {
        "base_tag": "Zc01",
        "base_des": "hZc01",
        "coarse_ratio": 0.1,
        "freeze_coarse": 0,
        "prev_setting": "TwS2Zc01s512p192_PatchTST_ETTh2_sl512_pl192_hZc01_0",
        "prev_mse": 0.343163,
    },
    336: {
        "base_tag": "Zc05",
        "base_des": "hZc05",
        "coarse_ratio": 0.5,
        "freeze_coarse": 0,
        "prev_setting": "TwS2Zc05s512p336_PatchTST_ETTh2_sl512_pl336_hZc05_0",
        "prev_mse": 0.330988,
    },
    720: {
        "base_tag": "Zc01",
        "base_des": "hZc01",
        "coarse_ratio": 0.1,
        "freeze_coarse": 0,
        "prev_setting": "TwS2Zc01s512p720_PatchTST_ETTh2_sl512_pl720_hZc01_0",
        "prev_mse": 0.382145,
    },
}

FLATTEN_MSE = {96: 0.274737, 192: 0.343185, 336: 0.331616, 720: 0.383078}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def s1_setting(pred_len):
    return f"TwF2s512p{pred_len}_PatchTST_{DATA}_sl{SEQ_LEN}_pl{pred_len}_hF_0"


def build_tasks(alpha_ratios):
    tasks = []
    for pred_len, cfg in BEST_BY_PL.items():
        for ar in alpha_ratios:
            ar_tag = f"A{str(ar).replace('.', '')}"
            mid = f"TwS2{cfg['base_tag']}{ar_tag}s512p{pred_len}"
            des = f"h{ar_tag}"
            tasks.append(
                {
                    "task_id": f"alpha_{ar_tag}_pl{pred_len}",
                    "pred_len": pred_len,
                    "base_tag": cfg["base_tag"],
                    "alpha_lr_ratio": ar,
                    "ar_tag": ar_tag,
                    "model_id": mid,
                    "des": des,
                    "setting": f"{mid}_PatchTST_{DATA}_sl{SEQ_LEN}_pl{pred_len}_{des}_0",
                    "init_setting": cfg["prev_setting"],
                    "coarse_ratio": cfg["coarse_ratio"],
                    "freeze_coarse": cfg["freeze_coarse"],
                    "prev_mse": cfg["prev_mse"],
                }
            )
    return tasks


def base_cmd(args, task):
    return [
        "python",
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
        task["model_id"],
        "--model",
        "PatchTST",
        "--data",
        DATA,
        "--features",
        "M",
        "--freq",
        "h",
        "--seq_len",
        str(SEQ_LEN),
        "--pred_len",
        str(task["pred_len"]),
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
        "--train_epochs",
        str(S2_EPOCHS),
        "--patience",
        str(S2_PATIENCE),
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
        task["des"],
        "--skip_visual",
        "1",
        "--head_type",
        "sacq_cross_attention",
        "--cross_attn_layers",
        "2",
        "--sacq_scheme",
        "default",
        "--sacq_pe_mode",
        "relative_bias",
        "--sacq_alpha_mode",
        "learnable",
        "--sacq_alpha_value",
        str(S2_ALPHA_INIT),
        "--sacq_coarse_lr_ratio",
        str(task["coarse_ratio"]),
        "--sacq_backbone_lr_ratio",
        "0",
        "--sacq_alpha_lr_ratio",
        str(task["alpha_lr_ratio"]),
        "--freeze_backbone",
        "1",
        "--freeze_coarse_head",
        str(task["freeze_coarse"]),
        "--init_checkpoint_setting",
        task["init_setting"],
        "--bootstrap_flatten_head",
        "0",
        "--dsacq_enable",
        "0",
        "--dsacq_mem_gate_enable",
        "0",
        "--dsacq_drift_gate_enable",
        "0",
    ]


def run_one(task, gpu_id, args, out_dir):
    log_path = os.path.join(out_dir, "logs", f"{task['task_id']}.log")
    metrics_path = os.path.join(PROJECT_ROOT, "results", task["setting"], "metrics.json")
    py = os.environ.get("PYTHON", DEFAULT_PYTHON if os.path.isfile(DEFAULT_PYTHON) else sys.executable)
    cmd = base_cmd(args, task)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if cmd[0] == "python":
        cmd[0] = py
    ensure_dir(os.path.dirname(log_path))
    t0 = time.time()
    with open(log_path, "w") as lf:
        lf.write(f"init={task['init_setting']}\nalpha_lr_ratio={task['alpha_lr_ratio']}\n")
        lf.write("cmd=" + " ".join(cmd) + "\n\n")
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
    row = {
        "task_id": task["task_id"],
        "pred_len": task["pred_len"],
        "base_tag": task["base_tag"],
        "alpha_lr_ratio": task["alpha_lr_ratio"],
        "status": "ok" if proc.returncode == 0 else "fail",
        "return_code": proc.returncode,
        "gpu": gpu_id,
        "elapsed_sec": round(time.time() - t0, 1),
        "setting": task["setting"],
        "prev_mse": task["prev_mse"],
    }
    if proc.returncode == 0 and os.path.isfile(metrics_path):
        with open(metrics_path) as f:
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
            write_csv(csv_path, header, row)
            print(
                f"[GPU {gpu_id}] {row['task_id']} a×{row['alpha_lr_ratio']} -> {row['status']} "
                f"mse={row.get('mse', 'n/a')}",
                flush=True,
            )
        q.task_done()


def write_csv(path, header, row):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(row)


def write_results_md(out_dir, csv_path):
    rows = list(csv.DictReader(open(csv_path))) if os.path.isfile(csv_path) else []
    ok = [r for r in rows if r.get("status") == "ok" and r.get("mse")]
    path = os.path.join(out_dir, "RESULTS.md")
    with open(path, "w") as f:
        f.write("# ETTh2 alpha LR boost (best freeze-SACQ config per pl)\n\n")
        f.write(
            f"Init from previous best SACQ ckpt; train e={S2_EPOCHS} p={S2_PATIENCE}; "
            f"alpha_init={S2_ALPHA_INIT}; freeze encoder; sweep alpha_lr_ratio ∈ {ALPHA_LR_RATIOS}.\n\n"
        )
        f.write("| pl | base | Flatten | SACQ α×1 (prev) |")
        for ar in ALPHA_LR_RATIOS:
            f.write(f" α×{ar:g} |")
        f.write("\n|----|------|---------|----------------|")
        for ar in ALPHA_LR_RATIOS:
            f.write("--------|")
        f.write("\n")
        for pl in PRED_LENS:
            cfg = BEST_BY_PL[pl]
            sub = [r for r in ok if str(r.get("pred_len")) == str(pl)]
            sub.sort(key=lambda r: float(r["alpha_lr_ratio"]))
            f.write(f"| {pl} | {cfg['base_tag']} | {FLATTEN_MSE[pl]:.6f} | {cfg['prev_mse']:.6f} |")
            for ar in ALPHA_LR_RATIOS:
                hit = next((r for r in sub if float(r["alpha_lr_ratio"]) == ar), None)
                if hit:
                    mse = float(hit["mse"])
                    dm = (mse - FLATTEN_MSE[pl]) / FLATTEN_MSE[pl] * 100
                    f.write(f" {mse:.6f} ({dm:+.2f}%) |")
                else:
                    f.write(" — |")
            f.write("\n")
        f.write("\n## Detail (MSE / MAE)\n\n")
        for pl in PRED_LENS:
            f.write(f"### pred_len={pl} ({BEST_BY_PL[pl]['base_tag']})\n\n")
            f.write("| alpha_lr× | MSE | MAE | Δ vs Flatten |\n|-----------|-----|-----|-------------|\n")
            f.write(
                f"| 1 (prev) | {BEST_BY_PL[pl]['prev_mse']:.6f} | — | "
                f"{(BEST_BY_PL[pl]['prev_mse'] - FLATTEN_MSE[pl]) / FLATTEN_MSE[pl] * 100:+.2f}% |\n"
            )
            sub = sorted(
                [r for r in ok if str(r.get("pred_len")) == str(pl)],
                key=lambda r: float(r["alpha_lr_ratio"]),
            )
            for r in sub:
                mse = float(r["mse"])
                dm = (mse - FLATTEN_MSE[pl]) / FLATTEN_MSE[pl] * 100
                f.write(
                    f"| {float(r['alpha_lr_ratio']):g} | {mse:.6f} | {float(r['mae']):.6f} | {dm:+.2f}% |\n"
                )
            if sub:
                best = min(sub, key=lambda r: float(r["mse"]))
                f.write(
                    f"\nBest boost: **α×{float(best['alpha_lr_ratio']):g}** "
                    f"MSE={float(best['mse']):.6f}\n\n"
                )
    print(f"wrote {path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--out_dir", type=str, default="logs/etth2_alpha_lr_boost")
    parser.add_argument("--alpha_ratios", type=str, default="2,5,10")
    parser.add_argument("--seed", type=int, default=2021)
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
    args = parser.parse_args()

    global ALPHA_LR_RATIOS
    ALPHA_LR_RATIOS = [float(x) for x in args.alpha_ratios.split(",") if x.strip()]

    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "logs"))
    os.environ.setdefault("PYTHON", DEFAULT_PYTHON if os.path.isfile(DEFAULT_PYTHON) else sys.executable)

    csv_path = os.path.join(out_dir, "summary.csv")
    if os.path.isfile(csv_path):
        os.remove(csv_path)

    tasks = build_tasks(ALPHA_LR_RATIOS)
    print(f"tasks={len(tasks)} alpha_ratios={ALPHA_LR_RATIOS}", flush=True)

    q = Queue()
    for t in tasks:
        q.put(t)

    header = [
        "task_id", "pred_len", "base_tag", "alpha_lr_ratio", "status", "return_code",
        "gpu", "elapsed_sec", "setting", "prev_mse", "mse", "mae",
    ]
    lock = threading.Lock()
    threads = []
    for gpu in [int(x) for x in args.gpus.split(",") if x.strip()]:
        th = threading.Thread(target=worker, args=(gpu, q, args, out_dir, csv_path, header, lock))
        th.start()
        threads.append(th)
    for th in threads:
        th.join()

    write_results_md(out_dir, csv_path)


if __name__ == "__main__":
    main()
