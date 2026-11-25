"""Finetune S1/S2/S12 on clean ETTh2 with gate/sign losses (better drift robustness)."""
import os
import subprocess
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PYTHON = os.environ.get("SACQ_PYTHON", "/home/bkai/miniconda3/envs/forecast/bin/python")

VARIANTS = [
    ("S1", "hS1g", "dual_residual", "default", True, "H2s512p720S1_PatchTST_ETTh2_sl512_pl720_hS1_0"),
    ("S2", "hS2g", "default", "relative_bias", False, "H2s512p720S2_PatchTST_ETTh2_sl512_pl720_hS2_0"),
    ("S12", "hS12g", "dual_residual", "relative_bias", True, "H2s512p720S12_PatchTST_ETTh2_sl512_pl720_hS12_0"),
]

S1_FLAGS = [
    "--sacq_dual_struct_gate", "0",
    "--sacq_dual_res_fuse", "0",
    "--sacq_dual_detach", "1",
    "--sacq_dual_scaffold_ln", "0",
]


def run_one(tag, des, scheme, pe_mode, dual, init_setting):
    mid = f"H2s512p720{tag}g"
    cmd = [
        PYTHON, "-u", os.path.join(PROJECT_ROOT, "run_longExp.py"),
        "--is_training", "1", "--random_seed", "2021",
        "--root_path", "./dataset/", "--data_path", "ETTh2.csv",
        "--model_id", mid, "--model", "PatchTST", "--data", "ETTh2",
        "--features", "M", "--freq", "h",
        "--seq_len", "512", "--pred_len", "720",
        "--enc_in", "7", "--dec_in", "7", "--c_out", "7",
        "--d_model", "16", "--n_heads", "4", "--e_layers", "3", "--d_ff", "128",
        "--dropout", "0.3", "--fc_dropout", "0.3", "--head_dropout", "0",
        "--patch_len", "16", "--stride", "8",
        "--head_type", "sacq_cross_attention", "--cross_attn_layers", "2",
        "--sacq_scheme", scheme, "--sacq_pe_mode", pe_mode,
        "--sacq_alpha_mode", "learnable", "--sacq_alpha_value", "0.1",
        "--sacq_gate_loss_weight", "0.3", "--sacq_corr_sign_loss_weight", "0.15",
        "--init_checkpoint_setting", init_setting,
        "--train_epochs", "15", "--patience", "8", "--batch_size", "128",
        "--learning_rate", "0.0001", "--lradj", "TST", "--pct_start", "0.2",
        "--des", des, "--short_setting", "1", "--itr", "1", "--num_workers", "4",
        "--gpu", "0", "--dsacq_enable", "0",
    ]
    if dual:
        cmd.extend(S1_FLAGS)
    log = os.path.join(PROJECT_ROOT, "logs/etth2_drift_shift", "logs", f"finetune_{tag}g.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "w") as f:
        subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=f, stderr=subprocess.STDOUT, check=False)
    return f"{mid}_PatchTST_ETTh2_sl512_pl720_{des}_0"


def main():
    for tag, des, scheme, pe, dual, init in VARIANTS:
        print("finetune", tag)
        setting = run_one(tag, des, scheme, pe, dual, init)
        print("  ->", setting)


if __name__ == "__main__":
    sys.exit(main())
