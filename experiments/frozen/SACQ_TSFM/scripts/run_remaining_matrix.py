"""Sequential, detached experiment queue for the remaining benchmark matrix."""

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime


ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / "logs"
POINTER = LOG_ROOT / "remaining_matrix.logpath"
PID_FILE = LOG_ROOT / "remaining_matrix.pid"
LOCK_FILE = LOG_ROOT / "remaining_matrix.lock"


def make_jobs():
    jobs = []

    # Phase 1: Weather configurations already tested with seed2026.
    configs = (
        ("ttm", 96),
        ("ttm", 720),
        ("chronos", 720),
        ("timer", 720),
    )
    for seed in (2027, 2028):
        for backbone, horizon in configs:
            for head in ("linear", "sacq"):
                jobs.append(("weather", backbone, horizon, head, seed, 128))

    # Phase 2: Missing short-horizon configurations, seed2026 first.
    for seed in (2026, 2027, 2028):
        for dataset, batch in (("weather", 128), ("electricity", 8)):
            for backbone in ("chronos", "timer"):
                for head in ("linear", "sacq"):
                    jobs.append((dataset, backbone, 96, head, seed, batch))

    assert len(jobs) == 40
    return jobs


def job_name(job):
    dataset, backbone, horizon, head, seed, batch = job
    return f"{dataset}_p{horizon}_{backbone}_{head}_seed{seed}_frozen_v1"


def result_path(job):
    return ROOT / "results/formal" / f"{job_name(job)}.json"


def checkpoint_path(job):
    return ROOT / "checkpoints/formal" / f"{job_name(job)}.pt"


def find_values(obj, key):
    """Find configuration fields in flat or nested result dictionaries."""
    values = []
    if isinstance(obj, dict):
        for name, value in obj.items():
            if name == key:
                values.append(value)
            if isinstance(value, dict):
                values.extend(find_values(value, key))
    return values


def validate_result(job):
    path = result_path(job)
    data = json.loads(path.read_text(encoding="utf-8"))

    dataset, backbone, horizon, head, seed, batch = job
    expected = {
        "run_name": job_name(job),
        "dataset": dataset,
        "backbone": backbone,
        "head": head,
        "context_length": 512,
        "prediction_length": horizon,
        "seed": seed,
        "batch_size": batch,
        "max_epochs": 50,
        "patience": 7,
        "loss": "mse",
        "backbone_unchanged": True,
    }

    for key, expected_value in expected.items():
        if key not in data or data[key] != expected_value:
            raise RuntimeError(
                f"Configuration mismatch: {path.name}; "
                f"{key}={data.get(key)!r}, expected={expected_value!r}"
            )

    for key, expected_value in (
        ("learning_rate", 1e-4),
        ("weight_decay", 1e-2),
    ):
        if key not in data or not math.isclose(
            float(data[key]), expected_value,
            rel_tol=1e-9, abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"Configuration mismatch: {path.name}; "
                f"{key}={data.get(key)!r}"
            )

    # Older results identify the protocol through run_name.
    # If an explicit field exists, verify it too.
    if data.get("training_protocol", "frozen_v1") != "frozen_v1":
        raise RuntimeError(f"Protocol mismatch: {path.name}")

    test = data.get("test")
    if not isinstance(test, dict):
        raise RuntimeError(f"Missing test metrics: {path}")

    for metric in ("mse", "mae"):
        value = float(test[metric])
        if not math.isfinite(value) or value < 0:
            raise RuntimeError(f"Invalid {metric}: {path}")

    if not checkpoint_path(job).is_file():
        raise RuntimeError(f"Missing checkpoint: {checkpoint_path(job)}")

    return test


def acquire_lock():
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    handle = LOCK_FILE.open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another remaining-matrix queue is running")
    return handle


def preflight(jobs):
    if Path(sys.prefix).name != "sacq_tsfm":
        raise RuntimeError("Activate conda environment sacq_tsfm first")

    completed = 0
    for job in jobs:
        if result_path(job).exists():
            validate_result(job)
            completed += 1
        elif checkpoint_path(job).exists():
            raise RuntimeError(
                "Incomplete checkpoint exists; preserve it before restarting:\n"
                f"{checkpoint_path(job)}"
            )

    print(f"Verified existing results: {completed}/{len(jobs)}", flush=True)
    print(f"Pending jobs: {len(jobs) - completed}", flush=True)


def run_command(command, log_path, env):
    print("COMMAND:", " ".join(map(str, command)), flush=True)
    print("JOB LOG:", log_path, flush=True)

    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(map(str, command)),
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise

    if code != 0:
        raise RuntimeError(f"Command failed with exit={code}; see {log_path}")


def worker(gpu, run_dir):
    os.chdir(ROOT)
    env = os.environ.copy()
    env.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "PYTHONPATH": str(ROOT),
        "PYTHONUNBUFFERED": "1",
    })

    # Set CUDA visibility before importing torch in this worker.
    os.environ.update(env)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    print("Python:", sys.executable, flush=True)
    print("Physical GPU:", gpu, flush=True)
    print("Logical cuda:0:", torch.cuda.get_device_name(0), flush=True)
    print("Weather batch=128; Electricity batch=8", flush=True)
    print("MSE / frozen backbone / frozen_v1", flush=True)

    jobs = make_jobs()
    preflight(jobs)
    (run_dir / "jobs.json").write_text(
        json.dumps(jobs, indent=2), encoding="utf-8"
    )

    finished = 0
    for index, job in enumerate(jobs, 1):
        name = job_name(job)

        if result_path(job).exists():
            validate_result(job)
            print(f"SKIP [{index}/40]: {name}", flush=True)
            finished += 1
            continue

        # Recheck immediately before running.
        if checkpoint_path(job).exists():
            raise RuntimeError(f"Incomplete checkpoint: {checkpoint_path(job)}")

        dataset, backbone, horizon, head, seed, batch = job
        common = [
            "--dataset", dataset,
            "--backbone", backbone,
            "--head", head,
            "--prediction-length", str(horizon),
            "--batch-size", str(batch),
            "--device", "cuda:0",
            "--seed", str(seed),
        ]

        print(f"\nSTART [{index}/40]: {name}", flush=True)
        print(datetime.now().isoformat(timespec="seconds"), flush=True)

        # Independent process: its test update is not used in training.
        run_command(
            [sys.executable, "-u", "scripts/batch_memory_test.py", *common],
            run_dir / f"{name}_memory.log",
            env,
        )

        run_command(
            [
                sys.executable, "-u", "scripts/train.py", *common,
                "--epochs", "50",
                "--patience", "7",
                "--learning-rate", "0.0001",
                "--weight-decay", "0.01",
                "--num-workers", "0",
                "--training-protocol", "frozen_v1",
            ],
            run_dir / f"{name}_train.log",
            env,
        )

        test = validate_result(job)
        finished += 1
        with (run_dir / "completed.txt").open("a", encoding="utf-8") as file:
            file.write(name + "\n")

        print(f"DONE [{index}/40]: {name}; test={test}", flush=True)

    if finished != 40:
        raise RuntimeError(f"Incomplete queue: {finished}/40")

    print("ALL_REMAINING_MATRIX_FINISHED: 40", flush=True)
    print(datetime.now().isoformat(timespec="seconds"), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()

    if args.gpu < 0:
        raise ValueError("gpu must be nonnegative")

    # Launcher passes this lock to its detached worker.
    inherited = os.environ.get("SACQ_QUEUE_LOCK_FD")
    if args.worker:
        if inherited is None:
            raise RuntimeError("Start without --worker")
        lock = os.fdopen(int(inherited), "a+")
        try:
            worker(args.gpu, args.run_dir)
        except Exception:
            import traceback
            traceback.print_exc()
            print("QUEUE_FAILED: no automatic batch changes or retries", flush=True)
            raise
        return

    lock = acquire_lock()
    preflight(make_jobs())

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = LOG_ROOT / f"remaining_matrix_{stamp}_{os.getpid()}"
    run_dir.mkdir()
    master_log = run_dir / "queue.log"

    env = os.environ.copy()
    env["SACQ_QUEUE_LOCK_FD"] = str(lock.fileno())

    with master_log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            [
                sys.executable, "-u", str(Path(__file__).resolve()),
                "--worker", "--gpu", str(args.gpu),
                "--run-dir", str(run_dir),
            ],
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            pass_fds=(lock.fileno(),),
        )

    PID_FILE.write_text(str(process.pid) + "\n", encoding="utf-8")
    POINTER.write_text(str(master_log) + "\n", encoding="utf-8")
    print("Background PID:", process.pid)
    print("Log:", master_log)
    print("Check the log to confirm that training starts.")


if __name__ == "__main__":
    main()