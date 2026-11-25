"""ETTh2 two-stage: (1) Flatten e30/p15, (2) SACQ-S2 joint e20/p10 with coarse LR 0.1x."""
import argparse
import json
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_PYTHON = "/home/bkai/miniconda3/envs/forecast/bin/python"

DATA = "ETTh2"
DATA_PATH = "ETTh2.csv"
ENC_IN = 7
SEQ_LEN = 512
PRED_LENS = [96, 720]

# Stage 1: Flatten
S1_EPOCHS = 30
S1_PATIENCE = 15
# Stage 2: SACQ-S2 joint
S2_EPOCHS = 20
S2_PATIENCE = 10
COARSE_LR_RATIO = 0.1
BACKBONE_LR_RATIO = 0.3


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def setting_name(model_id, des, pred_len, ii=0):
    return f"{model_id}_PatchTST_{DATA}_sl{SEQ_LEN}_pl{pred_len}_{des}_{ii}"


def run_train(cmd, gpu, log_path):
    ensure_dir(os.path.dirname(log_path))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    py = env.get("PYTHON", DEFAULT_PYTHON if os.path.isfile(DEFAULT_PYTHON) else sys.executable)
    if cmd[0] == "python":
        cmd[0] = py
    t0 = time.time()
    with open(log_path, "w") as lf:
        lf.write("cmd=" + " ".join(cmd) + "\n\n")
        lf.flush()
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
    return proc.returncode, time.time() - t0


def base_cmd(args, model_id, des, pred_len, epochs, patience):
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
        model_id,
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
        des,
        "--skip_visual",
        "1",
        "--dsacq_enable",
        "0",
        "--dsacq_mem_gate_enable",
        "0",
        "--dsacq_drift_gate_enable",
        "0",
    ]


def read_metrics(setting):
    path = os.path.join(PROJECT_ROOT, "results", setting, "metrics.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def write_results_md(out_dir, rows):
    md = os.path.join(out_dir, "RESULTS.md")
    with open(md, "w") as f:
        f.write("# ETTh2 two-stage Flatten → SACQ-S2\n\n")
        f.write(
            f"Stage1 Flatten: epochs={S1_EPOCHS}, patience={S1_PATIENCE}. "
            f"Stage2 SACQ-S2: epochs={S2_EPOCHS}, patience={S2_PATIENCE}, "
            f"coarse_lr={COARSE_LR_RATIO}x, backbone_lr={BACKBONE_LR_RATIO}x, "
            f"bootstrap_flatten_head=1.\n\n"
        )
        f.write("| pred_len | Flatten MSE | Flatten MAE | SACQ-S2 MSE | SACQ-S2 MAE | ΔMSE% |\n")
        f.write("|----------|-------------|-------------|-------------|-------------|-------|\n")
        for r in rows:
            dm = ""
            if r.get("f_mse") is not None and r.get("s_mse") is not None and r["f_mse"] > 0:
                dm = f"{(r['s_mse'] - r['f_mse']) / r['f_mse'] * 100:+.2f}%"
            f.write(
                f"| {r['pred_len']} | {r.get('f_mse', '—')} | {r.get('f_mae', '—')} | "
                f"{r.get('s_mse', '—')} | {r.get('s_mae', '—')} | {dm} |\n"
            )
    print(f"Wrote {md}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="logs/etth2_twostage_s2")
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
    parser.add_argument("--pred_lens", type=str, default="96,720")
    parser.add_argument("--skip_stage1", action="store_true")
    parser.add_argument("--skip_stage2", action="store_true")
    args = parser.parse_args()

    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    log_dir = os.path.join(out_dir, "logs")
    ensure_dir(out_dir)
    ensure_dir(log_dir)
    os.environ.setdefault("PYTHON", DEFAULT_PYTHON if os.path.isfile(DEFAULT_PYTHON) else sys.executable)

    pred_lens = [int(x) for x in args.pred_lens.split(",")]
    summary_rows = []

    for pred_len in pred_lens:
        s1_id = f"TwF2s512p{pred_len}"
        s2_id = f"TwS2s512p{pred_len}"
        s1_des = "hF"
        s2_des = "hS2"
        s1_setting = setting_name(s1_id, s1_des, pred_len)
        s2_setting = setting_name(s2_id, s2_des, pred_len)

        row = {"pred_len": pred_len, "s1_setting": s1_setting, "s2_setting": s2_setting}

        if not args.skip_stage1:
            cmd = base_cmd(args, s1_id, s1_des, pred_len, S1_EPOCHS, S1_PATIENCE)
            cmd += ["--head_type", "flatten"]
            log_path = os.path.join(log_dir, f"s1_pl{pred_len}.log")
            print(f"[S1 Flatten pl={pred_len}] -> {log_path}")
            rc, elapsed = run_train(cmd, args.gpu, log_path)
            row["s1_rc"] = rc
            row["s1_sec"] = round(elapsed, 1)
            if rc != 0:
                print(f"Stage1 failed rc={rc}")
        else:
            print(f"[S1 skip] pl={pred_len}")

        m1 = read_metrics(s1_setting)
        if m1:
            row["f_mse"] = m1.get("mse")
            row["f_mae"] = m1.get("mae")

        if not args.skip_stage2:
            cmd = base_cmd(args, s2_id, s2_des, pred_len, S2_EPOCHS, S2_PATIENCE)
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
                "0.1",
                "--sacq_coarse_lr_ratio",
                str(COARSE_LR_RATIO),
                "--sacq_backbone_lr_ratio",
                str(BACKBONE_LR_RATIO),
                "--init_checkpoint_setting",
                s1_setting,
                "--bootstrap_flatten_head",
                "1",
            ]
            log_path = os.path.join(log_dir, f"s2_pl{pred_len}.log")
            print(f"[S2 SACQ-S2 pl={pred_len}] init={s1_setting} -> {log_path}")
            rc, elapsed = run_train(cmd, args.gpu, log_path)
            row["s2_rc"] = rc
            row["s2_sec"] = round(elapsed, 1)
            if rc != 0:
                print(f"Stage2 failed rc={rc}")
        else:
            print(f"[S2 skip] pl={pred_len}")

        m2 = read_metrics(s2_setting)
        if m2:
            row["s_mse"] = m2.get("mse")
            row["s_mae"] = m2.get("mae")

        summary_rows.append(row)

    write_results_md(out_dir, summary_rows)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary_rows, f, indent=2)


if __name__ == "__main__":
    main()
