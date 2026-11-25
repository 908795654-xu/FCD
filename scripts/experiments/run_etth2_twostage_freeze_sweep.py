"""ETTh2 two-stage sweep: S1 Flatten; S2 SACQ-S2 with freeze/LR variants, 4-GPU queue."""
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

S1_EPOCHS = 30
S1_PATIENCE = 15
S2_EPOCHS = 20
S2_PATIENCE = 10
S2_ALPHA = 0.0001

# tag, des, freeze_backbone, freeze_coarse, coarse_lr_ratio
S2_VARIANTS = [
    ("Z0", "hZ0", 1, 1, 0.1),
    ("Zc05", "hZc05", 1, 0, 0.5),
    ("Zc01", "hZc01", 1, 0, 0.1),
    ("Zc001", "hZc001", 1, 0, 0.01),
]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def setting_name(model_id, des, pred_len, ii=0):
    return f"{model_id}_PatchTST_{DATA}_sl{SEQ_LEN}_pl{pred_len}_{des}_{ii}"


def s1_setting(pred_len):
    return setting_name(f"TwF2s512p{pred_len}", "hF", pred_len)


def ckpt_exists(setting):
    return os.path.isfile(os.path.join(PROJECT_ROOT, "checkpoints", setting, "checkpoint.pth"))


def build_tasks(skip_s1_if_ckpt=True, only_stage=""):
    tasks = []
    for pred_len in PRED_LENS:
        s1_set = s1_setting(pred_len)
        if only_stage in ("", "1"):
            if not (skip_s1_if_ckpt and ckpt_exists(s1_set)):
                tasks.append(
                    {
                        "phase": "s1",
                        "task_id": f"s1_pl{pred_len}",
                        "pred_len": pred_len,
                        "model_id": f"TwF2s512p{pred_len}",
                        "des": "hF",
                        "setting": s1_set,
                    }
                )
        if only_stage in ("", "2"):
            for tag, des, fbb, fco, cr in S2_VARIANTS:
                mid = f"TwS2{tag}s512p{pred_len}"
                tasks.append(
                    {
                        "phase": "s2",
                        "task_id": f"s2_{tag}_pl{pred_len}",
                        "pred_len": pred_len,
                        "tag": tag,
                        "des": des,
                        "model_id": mid,
                        "setting": setting_name(mid, des, pred_len),
                        "init_setting": s1_set,
                        "freeze_backbone": fbb,
                        "freeze_coarse": fco,
                        "coarse_lr_ratio": cr,
                    }
                )
    return tasks


def base_cmd(args, task, epochs, patience):
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
        str(epochs),
        "--patience",
        str(patience),
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
    if task["phase"] == "s1":
        cmd = base_cmd(args, task, S1_EPOCHS, S1_PATIENCE)
        cmd += ["--head_type", "flatten"]
    else:
        cmd = base_cmd(args, task, S2_EPOCHS, S2_PATIENCE)
        cmd += [
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
            str(S2_ALPHA),
            "--sacq_coarse_lr_ratio",
            str(task["coarse_lr_ratio"]),
            "--sacq_backbone_lr_ratio",
            "0",
            "--freeze_backbone",
            str(task["freeze_backbone"]),
            "--freeze_coarse_head",
            str(task["freeze_coarse"]),
            "--init_checkpoint_setting",
            task["init_setting"],
            "--bootstrap_flatten_head",
            "1",
        ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if cmd[0] == "python":
        cmd[0] = py
    ensure_dir(os.path.dirname(log_path))
    t0 = time.time()
    with open(log_path, "w") as lf:
        lf.write(f"phase={task['phase']}\nsetting={task['setting']}\n")
        lf.write("cmd=" + " ".join(cmd) + "\n\n")
        lf.flush()
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
    row = {
        "task_id": task["task_id"],
        "phase": task["phase"],
        "pred_len": task["pred_len"],
        "status": "ok" if proc.returncode == 0 else "fail",
        "return_code": proc.returncode,
        "gpu": gpu_id,
        "elapsed_sec": round(time.time() - t0, 1),
        "setting": task["setting"],
        "log_path": log_path,
    }
    if task["phase"] == "s2":
        row["tag"] = task["tag"]
        row["variant"] = task["tag"]
        row["freeze_backbone"] = task["freeze_backbone"]
        row["freeze_coarse"] = task["freeze_coarse"]
        row["coarse_lr_ratio"] = task["coarse_lr_ratio"]
    else:
        row["tag"] = "F"
        row["variant"] = "Flatten"
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
            write_csv_row(csv_path, header, row)
            print(
                f"[GPU {gpu_id}] {row['task_id']} -> {row['status']} "
                f"mse={row.get('mse', 'n/a')} ({row['elapsed_sec']}s)",
                flush=True,
            )
        q.task_done()


def write_csv_row(path, header, row):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(row)


def flatten_metrics_from_disk(pred_len):
    setting = s1_setting(pred_len)
    path = os.path.join(PROJECT_ROOT, "results", setting, "metrics.json")
    if os.path.isfile(path):
        with open(path) as f:
            m = json.load(f)
        return m.get("mse"), m.get("mae")
    return None, None


def write_results_md(out_dir, csv_path):
    if not os.path.isfile(csv_path):
        return
    rows = list(csv.DictReader(open(csv_path)))
    ok = [r for r in rows if r.get("status") == "ok" and r.get("mse")]
    for pl in PRED_LENS:
        if any(str(r.get("pred_len")) == str(pl) and r.get("tag") == "F" for r in ok):
            continue
        fm, fa = flatten_metrics_from_disk(pl)
        if fm is not None:
            ok.append(
                {
                    "pred_len": pl,
                    "tag": "F",
                    "mse": fm,
                    "mae": fa,
                }
            )
    path = os.path.join(out_dir, "RESULTS.md")
    with open(path, "w") as f:
        f.write("# ETTh2 two-stage freeze / coarse-LR sweep (SACQ-S2)\n\n")
        f.write(
            f"Stage1 Flatten: e={S1_EPOCHS}, p={S1_PATIENCE}. "
            f"Stage2 SACQ-S2: e={S2_EPOCHS}, p={S2_PATIENCE}, "
            f"alpha_init={S2_ALPHA}, freeze backbone in S2.\n\n"
        )
        f.write("S2 variants: **Z0** freeze enc+coarse; **Zc05/Zc01/Zc001** freeze enc, coarse LR 0.5/0.1/0.01×.\n\n")
        for pl in PRED_LENS:
            f.write(f"## pred_len={pl}\n\n")
            f.write("| Variant | Config | MSE | MAE |\n|---------|--------|-----|-----|\n")
            sub = [r for r in ok if str(r.get("pred_len")) == str(pl)]
            order = ["F", "Z0", "Zc05", "Zc01", "Zc001"]
            labels = {
                "F": "Flatten (S1)",
                "Z0": "SACQ freeze enc+coarse",
                "Zc05": "SACQ freeze enc, coarse×0.5",
                "Zc01": "SACQ freeze enc, coarse×0.1",
                "Zc001": "SACQ freeze enc, coarse×0.01",
            }
            for tag in order:
                hit = next((r for r in sub if r.get("tag") == tag), None)
                if hit:
                    cfg = labels.get(tag, tag)
                    if tag != "F":
                        cfg = labels[tag]
                    f.write(
                        f"| {tag} | {cfg} | {float(hit['mse']):.6f} | {float(hit['mae']):.6f} |\n"
                    )
                else:
                    f.write(f"| {tag} | {labels.get(tag, tag)} | — | — |\n")
            f_flat = next((r for r in sub if r.get("tag") == "F"), None)
            sacq = [r for r in sub if r.get("tag") != "F"]
            if f_flat and sacq:
                best = min(sacq, key=lambda r: float(r["mse"]))
                dm = (float(best["mse"]) - float(f_flat["mse"])) / float(f_flat["mse"]) * 100
                f.write(
                    f"\nBest SACQ: **{best['tag']}** MSE={float(best['mse']):.6f} "
                    f"(Δ vs Flatten {dm:+.2f}%)\n\n"
                )
            else:
                f.write("\n")
        f.write("## Full matrix (MSE)\n\n")
        f.write("| pred_len | Flatten | Z0 | Zc05 | Zc01 | Zc001 |\n")
        f.write("|----------|---------|----|------|------|-------|\n")
        for pl in PRED_LENS:
            sub = {r.get("tag"): r for r in ok if str(r.get("pred_len")) == str(pl)}
            cells = []
            for tag in order:
                r = sub.get(tag)
                cells.append(f"{float(r['mse']):.6f}" if r and r.get("mse") else "—")
            f.write(f"| {pl} | " + " | ".join(cells) + " |\n")
    print(f"wrote {path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--out_dir", type=str, default="logs/etth2_twostage_freeze")
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
    parser.add_argument("--no_skip_s1_ckpt", action="store_true", help="re-run stage1 even if ckpt exists")
    parser.add_argument("--only_stage", type=str, default="", choices=["", "1", "2"])
    args = parser.parse_args()

    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "logs"))
    os.environ.setdefault("PYTHON", DEFAULT_PYTHON if os.path.isfile(DEFAULT_PYTHON) else sys.executable)

    csv_path = os.path.join(out_dir, "summary.csv")
    if os.path.isfile(csv_path) and args.only_stage == "":
        os.remove(csv_path)

    all_tasks = build_tasks(skip_s1_if_ckpt=not args.no_skip_s1_ckpt, only_stage=args.only_stage)
    s1_tasks = [t for t in all_tasks if t["phase"] == "s1"]
    s2_tasks = [t for t in all_tasks if t["phase"] == "s2"]
    print(f"s1_tasks={len(s1_tasks)} s2_tasks={len(s2_tasks)} gpus={args.gpus}", flush=True)

    header = [
        "task_id", "phase", "tag", "variant", "status", "return_code", "gpu",
        "elapsed_sec", "pred_len", "setting", "log_path",
        "freeze_backbone", "freeze_coarse", "coarse_lr_ratio", "mse", "mae",
    ]
    lock = threading.Lock()

    def run_queue(task_list):
        if not task_list:
            return
        q = Queue()
        for t in task_list:
            q.put(t)
        threads = []
        for gpu in [int(x) for x in args.gpus.split(",") if x.strip() != ""]:
            th = threading.Thread(
                target=worker, args=(gpu, q, args, out_dir, csv_path, header, lock)
            )
            th.start()
            threads.append(th)
        for th in threads:
            th.join()

    run_queue(s1_tasks)
    run_queue(s2_tasks)

    write_results_md(out_dir, csv_path)


if __name__ == "__main__":
    main()
