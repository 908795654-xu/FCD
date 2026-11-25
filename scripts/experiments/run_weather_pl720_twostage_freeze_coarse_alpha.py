"""Weather pl=720 stage-2: frozen Flatten backbone+coarse, finetune SAR decoder/alpha.

Init from weather_512_720 MSE Flatten checkpoint; bootstrap coarse; freeze backbone+coarse.
Three runs with sacq_alpha_value in {0.5, 0.1, 0.01}, loss=logcosh_scaled (auto_mad beta).
"""
from __future__ import annotations

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
DEFAULT_PYTHON = "/home/bkai/miniconda3/envs/forecast/bin/python"

DATA = "weather"
DATA_PATH = "weather.csv"
ENC_IN = 21
SEQ_LEN = 512
PRED_LEN = 720

# Stage-1 Flatten MSE (logs/weather_pl720_mse_vs_sar/logs/weather_512_720_mse.log)
S1_FLATTEN_SETTING = (
    "weather_512_720_PatchTST_weather_ftM_sl512_ll48_pl720_"
    "dm128_nh16_el3_dl1_df256_fc1_ebtimeF_dtFalse_Exp_0"
)
# Stage-1 full SAR (logs/weather_pl720_mse_vs_sar/logs/weather_512_720_sar.log)
S1_SAR_SETTING = (
    "weather_512_720_SAR_PatchTST_weather_ftM_sl512_ll48_pl720_"
    "dm128_nh16_el3_dl1_df256_fc1_ebtimeF_dtFalse_Exp_0"
)

ALPHA_VARIANTS: list[tuple[str, float]] = [
    ("a05", 0.5),
    ("a01", 0.1),
    ("a001", 0.01),
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_cmd(
    args,
    alpha_tag: str,
    *,
    alpha_mode: str = "learnable",
    alpha_init: float = 0.5,
    init_setting: str = S1_FLATTEN_SETTING,
    bootstrap_flatten: bool = True,
    loss: str = "logcosh_scaled",
    lradj: str = "TST",
    alpha_granularity: str = "patch",
    alpha_lr_ratio: float = 1.0,
    alpha_softplus: bool = False,
) -> list[str]:
    py = os.environ.get("PYTHON", DEFAULT_PYTHON if os.path.isfile(DEFAULT_PYTHON) else sys.executable)
    model_id = f"Ws512p720SARtw_{alpha_tag}"
    des = f"wSARtw_{alpha_tag}"
    return [
        py,
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
        str(PRED_LEN),
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
        "--lradj",
        lradj,
        "--itr",
        "1",
        "--num_workers",
        str(args.num_workers),
        "--head_type",
        "sacq_cross_attention",
        "--sacq_autoregressive",
        "1",
        "--sacq_alpha_mode",
        alpha_mode,
        "--sacq_alpha_value",
        str(alpha_init),
        "--sacq_alpha_granularity",
        alpha_granularity,
        "--sacq_alpha_lr_ratio",
        str(alpha_lr_ratio),
        "--sacq_alpha_softplus",
        "1" if alpha_softplus else "0",
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
        "--loss",
        loss,
        "--init_checkpoint_setting",
        init_setting,
        "--bootstrap_flatten_head",
        "1" if bootstrap_flatten else "0",
        "--freeze_backbone",
        "1",
        "--freeze_coarse_head",
        "1",
    ] + (
        ["--logcosh_beta", "1.0", "--logcosh_beta_mode", "auto_mad"]
        if loss == "logcosh_scaled"
        else []
    ) + (
        ["--pct_start", "0.2"] if lradj == "TST" else []
    )


def run_one(task: dict, gpu_id: int) -> dict:
    setting = f"{task['model_id']}_PatchTST_{DATA}_sl{SEQ_LEN}_pl{PRED_LEN}_{task['des']}_0"
    metrics_path = os.path.join(PROJECT_ROOT, "results", setting, "metrics.json")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    t0 = time.time()
    with open(task["log_path"], "w") as lf:
        lf.write(f"alpha_mode={task.get('alpha_mode', 'learnable')}\n")
        lf.write(f"alpha_init={task['alpha_init']}\n")
        lf.write(f"loss={task.get('loss', 'logcosh_scaled')}\n")
        lf.write(f"lradj={task.get('lradj', 'TST')}\n")
        lf.write(f"alpha_granularity={task.get('alpha_granularity', 'patch')}\n")
        lf.write(f"alpha_lr_ratio={task.get('alpha_lr_ratio', 1.0)}\n")
        lf.write(f"alpha_softplus={task.get('alpha_softplus', False)}\n")
        lf.write(f"s1_setting={task.get('s1_setting', S1_FLATTEN_SETTING)}\n")
        lf.write(f"gpu={gpu_id}\n")
        lf.write("cmd=" + " ".join(task["cmd"]) + "\n\n")
        lf.flush()
        proc = subprocess.run(
            task["cmd"], cwd=PROJECT_ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT
        )
    row = {
        "task_id": task["task_id"],
        "alpha_tag": task["alpha_tag"],
        "alpha_init": task["alpha_init"],
        "gpu": gpu_id,
        "status": "ok" if proc.returncode == 0 else "fail",
        "return_code": proc.returncode,
        "elapsed_sec": round(time.time() - t0, 2),
        "setting": setting,
        "log_path": task["log_path"],
        "mse": "",
        "mae": "",
    }
    if os.path.isfile(metrics_path):
        with open(metrics_path) as f:
            m = json.load(f)
        row["mse"] = m.get("mse", "")
        row["mae"] = m.get("mae", "")
    return row


def worker(gpu_id: int, q: Queue, csv_path: str, header: list[str], lock: threading.Lock) -> None:
    while True:
        try:
            task = q.get_nowait()
        except Empty:
            return
        print(f"[GPU {gpu_id}] {task['task_id']} alpha={task['alpha_init']}", flush=True)
        row = run_one(task, gpu_id)
        with lock:
            exists = os.path.exists(csv_path)
            with open(csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=header)
                if not exists:
                    w.writeheader()
                w.writerow(row)
        print(
            f"[GPU {gpu_id}] {task['task_id']} -> {row['status']} "
            f"mse={row['mse']} ({row['elapsed_sec']}s)",
            flush=True,
        )
        q.task_done()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="0,2,3", help="comma-separated GPU ids")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="logs/weather_pl720_twostage_freeze_coarse_alpha",
    )
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--e_layers", type=int, default=3)
    parser.add_argument("--n_heads", type=int, default=16)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--fc_dropout", type=float, default=0.2)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--mode",
        type=str,
        default="alpha_sweep",
        choices=[
            "alpha_sweep",
            "sar_alpha1",
            "sar_alpha1_mse",
            "sar_alpha1_constlr",
            "sar_alpha1_mse_constlr",
            "sar_alpha1_global_a",
            "sar_alpha1_global_a05",
            "sar_alpha1_global_a_sp",
        ],
        help="alpha_sweep: 3 learnable-alpha runs from Flatten S1; "
        "sar_alpha1: fixed alpha=1, init from SAR S1, train decoder only; "
        "sar_alpha1_mse: same as sar_alpha1 but loss=mse; "
        "sar_alpha1_constlr: same as sar_alpha1 but lradj=constant (no TST decay); "
        "sar_alpha1_mse_constlr: MSE loss + constant lr; "
        "sar_alpha1_global_a: global learnable alpha init~1, alpha_lr x5, const lr; "
        "sar_alpha1_global_a05: same with alpha_init=0.5; "
        "sar_alpha1_global_a_sp: same as sar_alpha1_global_a but sacq_alpha_softplus=1",
    )
    args = parser.parse_args()

    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    log_dir = os.path.join(out_dir, "logs")
    ensure_dir(log_dir)

    tasks: list[dict] = []

    def _sar_alpha1_task(
        alpha_tag: str,
        loss: str,
        log_name: str,
        *,
        alpha_mode: str = "fixed",
        alpha_init: float = 1.0,
        lradj: str = "TST",
        alpha_granularity: str = "patch",
        alpha_lr_ratio: float = 1.0,
        alpha_softplus: bool = False,
    ) -> dict:
        s1_setting = S1_SAR_SETTING
        return {
            "task_id": f"Ws512p720SARtw_{alpha_tag}",
            "model_id": f"Ws512p720SARtw_{alpha_tag}",
            "des": f"wSARtw_{alpha_tag}",
            "alpha_tag": alpha_tag,
            "alpha_mode": alpha_mode,
            "alpha_init": alpha_init,
            "loss": loss,
            "lradj": lradj,
            "alpha_granularity": alpha_granularity,
            "alpha_lr_ratio": alpha_lr_ratio,
            "alpha_softplus": alpha_softplus,
            "s1_setting": s1_setting,
            "cmd": build_cmd(
                args,
                alpha_tag,
                alpha_mode=alpha_mode,
                alpha_init=alpha_init,
                init_setting=s1_setting,
                bootstrap_flatten=False,
                loss=loss,
                lradj=lradj,
                alpha_granularity=alpha_granularity,
                alpha_lr_ratio=alpha_lr_ratio,
                alpha_softplus=alpha_softplus,
            ),
            "log_path": os.path.join(log_dir, log_name),
        }

    if args.mode == "sar_alpha1":
        tasks.append(_sar_alpha1_task("a1fix", "logcosh_scaled", "pl720_a1fix_sarinit.log"))
    elif args.mode == "sar_alpha1_mse":
        tasks.append(_sar_alpha1_task("a1fixmse", "mse", "pl720_a1fix_sarinit_mse.log"))
    elif args.mode == "sar_alpha1_constlr":
        tasks.append(
            _sar_alpha1_task(
                "a1fixcst",
                "logcosh_scaled",
                "pl720_a1fix_sarinit_constlr.log",
                lradj="constant",
            )
        )
    elif args.mode == "sar_alpha1_mse_constlr":
        tasks.append(
            _sar_alpha1_task(
                "a1fixmsecst",
                "mse",
                "pl720_a1fix_sarinit_mse_constlr.log",
                lradj="constant",
            )
        )
    elif args.mode == "sar_alpha1_global_a":
        tasks.append(
            _sar_alpha1_task(
                "a1galpha",
                "logcosh_scaled",
                "pl720_a1fix_sarinit_global_a.log",
                alpha_mode="learnable",
                alpha_init=1.0,
                lradj="constant",
                alpha_granularity="global",
                alpha_lr_ratio=5.0,
            )
        )
    elif args.mode == "sar_alpha1_global_a05":
        tasks.append(
            _sar_alpha1_task(
                "a1galpha05",
                "logcosh_scaled",
                "pl720_a1fix_sarinit_global_a05.log",
                alpha_mode="learnable",
                alpha_init=0.5,
                lradj="constant",
                alpha_granularity="global",
                alpha_lr_ratio=5.0,
            )
        )
    elif args.mode == "sar_alpha1_global_a_sp":
        tasks.append(
            _sar_alpha1_task(
                "a1galphasp",
                "logcosh_scaled",
                "pl720_a1fix_sarinit_global_a_sp.log",
                alpha_mode="learnable",
                alpha_init=1.0,
                lradj="constant",
                alpha_granularity="global",
                alpha_lr_ratio=5.0,
                alpha_softplus=True,
            )
        )
    else:
        s1_setting = S1_FLATTEN_SETTING
        for alpha_tag, alpha_init in ALPHA_VARIANTS:
            model_id = f"Ws512p720SARtw_{alpha_tag}"
            des = f"wSARtw_{alpha_tag}"
            tasks.append(
                {
                    "task_id": model_id,
                    "model_id": model_id,
                    "des": des,
                    "alpha_tag": alpha_tag,
                    "alpha_mode": "learnable",
                    "alpha_init": alpha_init,
                    "s1_setting": s1_setting,
                    "cmd": build_cmd(
                        args,
                        alpha_tag,
                        alpha_mode="learnable",
                        alpha_init=alpha_init,
                        init_setting=s1_setting,
                        bootstrap_flatten=True,
                    ),
                    "log_path": os.path.join(log_dir, f"pl720_{alpha_tag}.log"),
                }
            )

    s1_ckpt = os.path.join(PROJECT_ROOT, "checkpoints", tasks[0]["s1_setting"], "checkpoint.pth")
    if not os.path.isfile(s1_ckpt):
        print(f"ERROR: stage-1 checkpoint missing: {s1_ckpt}", file=sys.stderr)
        sys.exit(1)

    gpu_ids = [int(x) for x in args.gpus.split(",") if x.strip()]
    if len(gpu_ids) < len(tasks):
        print(
            f"WARNING: {len(gpu_ids)} GPUs for {len(tasks)} tasks; will queue.",
            flush=True,
        )

    meta_path = os.path.join(out_dir, f"experiment_{args.mode}.json")
    meta = {
        "mode": args.mode,
        "s1_setting": tasks[0]["s1_setting"],
        "s1_checkpoint": s1_ckpt,
        "freeze_backbone": True,
        "freeze_coarse_head": True,
        "loss": "logcosh_scaled",
        "logcosh_beta_mode": "auto_mad",
        "tasks": [t["task_id"] for t in tasks],
    }
    if args.mode in (
        "sar_alpha1",
        "sar_alpha1_mse",
        "sar_alpha1_constlr",
        "sar_alpha1_mse_constlr",
        "sar_alpha1_global_a",
        "sar_alpha1_global_a05",
        "sar_alpha1_global_a_sp",
    ):
        meta.update(
            {
                "bootstrap_flatten_head": False,
                "sacq_alpha_mode": "fixed",
                "sacq_alpha_value": 1.0,
                "loss": tasks[0].get("loss", "logcosh_scaled"),
                "lradj": tasks[0].get("lradj", "TST"),
                "note": "y = scaffold + 1.0 * corr; only cross-attn decoder trains",
            }
        )
    else:
        meta.update(
            {
                "bootstrap_flatten_head": True,
                "alpha_variants": ALPHA_VARIANTS,
            }
        )
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    csv_path = os.path.join(out_dir, "summary.csv")
    header = [
        "task_id",
        "alpha_tag",
        "alpha_init",
        "gpu",
        "status",
        "return_code",
        "elapsed_sec",
        "setting",
        "log_path",
        "mse",
        "mae",
    ]
    q: Queue = Queue()
    for t in tasks:
        q.put(t)

    lock = threading.Lock()
    threads = [
        threading.Thread(target=worker, args=(gid, q, csv_path, header, lock), daemon=False)
        for gid in gpu_ids
    ]
    launcher_log = os.path.join(out_dir, "launcher.log")
    with open(launcher_log, "w") as lf:
        lf.write(f"started={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        lf.write(f"gpus={gpu_ids}\n")
        lf.write(f"mode={args.mode}\n")
        lf.write(f"s1={tasks[0]['s1_setting']}\n")
        for t in tasks:
            lf.write(
                f"  {t['task_id']} alpha_mode={t.get('alpha_mode')} "
                f"alpha={t['alpha_init']}\n"
            )
    print(f"Launcher log: {launcher_log}", flush=True)
    print(f"Stage-1 init: {tasks[0]['s1_setting']}", flush=True)

    for th in threads:
        th.start()
    for th in threads:
        th.join()
    print(f"Done. summary -> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
