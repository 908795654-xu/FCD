"""Round-2 S1 tune: v2b (blend fuse fix) + v2sg (struct gate only)."""
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PY = os.environ.get("PYTHON", sys.executable)

VARIANTS = [
    ("v2b", "hS1v2b", 1, 1, 1, 0),
    ("v2sg", "hS1v2sg", 1, 0, 1, 0),
]


def main():
    gpus = os.environ.get("GPUS", "0,1,2,3").split(",")
    out = os.path.join(ROOT, "logs/etth2_s1_tune")
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    jobs = []
    for pl in (96, 720):
        for tag, des, sg, rf, dt, sln in VARIANTS:
            mid = f"E2s512p{pl}S1{tag}"
            jobs.append((pl, tag, des, mid, sg, rf, dt, sln))

    procs = []
    for i, (pl, tag, des, mid, sg, rf, dt, sln) in enumerate(jobs):
        gpu = gpus[i % len(gpus)]
        log = os.path.join(out, "logs", f"E2s512p{pl}S1{tag}_r2.log")
        cmd = [
            PY, "-u", "run_longExp.py",
            "--is_training", "1", "--random_seed", "2021",
            "--root_path", "./dataset/", "--data_path", "ETTh2.csv",
            "--model_id", mid, "--model", "PatchTST", "--data", "ETTh2",
            "--features", "M", "--freq", "h", "--seq_len", "512",
            "--pred_len", str(pl), "--enc_in", "7", "--dec_in", "7", "--c_out", "7",
            "--e_layers", "3", "--n_heads", "4", "--d_model", "16", "--d_ff", "128",
            "--dropout", "0.3", "--fc_dropout", "0.3", "--head_dropout", "0",
            "--patch_len", "16", "--stride", "8", "--cross_attn_layers", "2",
            "--train_epochs", "50", "--patience", "20", "--batch_size", "128",
            "--learning_rate", "0.0001", "--itr", "1", "--num_workers", "4",
            "--head_type", "sacq_cross_attention", "--sacq_scheme", "dual_residual",
            "--sacq_dual_struct_gate", str(sg),
            "--sacq_dual_res_fuse", str(rf),
            "--sacq_dual_detach", str(dt),
            "--sacq_dual_scaffold_ln", str(sln),
            "--sacq_alpha_mode", "learnable",
            "--short_setting", "1", "--des", des, "--skip_visual", "1",
            "--dsacq_enable", "0", "--dsacq_mem_gate_enable", "0", "--dsacq_drift_gate_enable", "0",
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu.strip()
        print(f"launch {mid} on GPU {gpu}", flush=True)
        lf = open(log, "w")
        lf.write(" ".join(cmd) + "\n\n")
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
        procs.append((tag, pl, gpu, log, p, lf))

    rows = []
    for tag, pl, gpu, log, p, lf in procs:
        p.wait()
        lf.close()
        mse = ""
        with open(log) as f:
            for line in f:
                if line.startswith("mse:"):
                    mse = line.strip().split(",")[0].split(":")[1]
        rows.append(f"{tag},pl{pl},gpu{gpu},rc={p.returncode},mse={mse}")
        print(f"done {tag} pl{pl} mse={mse}", flush=True)

    rep = os.path.join(out, "RESULTS_r2.txt")
    with open(rep, "w") as f:
        f.write("\n".join(rows))
    print(f"wrote {rep}", flush=True)


if __name__ == "__main__":
    main()
