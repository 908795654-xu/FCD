"""Run independent weight-decay trials concurrently on explicitly selected GPUs."""
import argparse
import concurrent.futures
import csv
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import tune_etth1_wd as core


def run(gpus, smoke):
    import fcntl
    core.OUT.mkdir(parents=True, exist_ok=True)
    # Same lock as the single-GPU queue; only one queue may write these trials.
    with (core.OUT / 'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fingerprint = core.identity()
        jobs = core.jobs()
        stamp = time.strftime('%Y%m%d_%H%M%S') + '_' + str(os.getpid())
        logdir = ROOT / 'logs' / ('etth1_wd_parallel_' + stamp)
        logdir.mkdir(parents=True)
        tasks = queue.Queue()
        rows, errors = [], []
        stop = threading.Event()

        def validate(job):
            path = core.OUT / (core.name(job) + '.json')
            r = json.loads(path.read_text())
            assert r['complete'] and r['fingerprint'] == fingerprint, path
            assert r['backbone_unchanged'] and not r['test_evaluated'], path
            assert math.isfinite(r['best_val']['mse']), path
            assert Path(r['checkpoint']).is_file(), path
            assert (r['backbone'], r['prediction_length'], r['head'],
                    r['learning_rate'], r['weight_decay']) == job, path
            return r

        # Validate all existing results before dispatching any new training.
        for job in jobs:
            path = core.OUT / (core.name(job) + '.json')
            if path.exists() and not smoke:
                rows.append(validate(job))
                print('SKIP', core.name(job), flush=True)
            else:
                tasks.put(job)
        print('GPUs:', gpus, 'Pending:', tasks.qsize(), 'Batch: 32',
              'Per-trial logs:', logdir, flush=True)

        def gpu_worker(gpu):
            while not stop.is_set():
                try:
                    job = tasks.get_nowait()
                except queue.Empty:
                    return
                try:
                    print('START GPU', gpu, core.name(job), flush=True)
                    cmd = [sys.executable, '-u', str(ROOT / 'scripts/tune_etth1_wd.py'),
                           '--worker', *map(str, job)]
                    if smoke:
                        cmd.append('--smoke')
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONPATH=str(ROOT))
                    log = logdir / (core.name(job) + '.log')
                    with log.open('x') as stream:
                        subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, check=True)
                    if not smoke:
                        rows.append(validate(job))
                    print('DONE GPU', gpu, core.name(job), 'log=', log, flush=True)
                except Exception as exc:
                    errors.append(f'GPU {gpu}: {core.name(job)}: {exc}')
                    stop.set()
                    print('FAILED:', errors[-1], flush=True)
                finally:
                    tasks.task_done()

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            list(pool.map(gpu_worker, gpus))
        if errors:
            raise RuntimeError('Queue stopped; already running trials were allowed to finish. '
                               'No batch changes or automatic retries. ' + '; '.join(errors))
        if smoke:
            print('ALL_36_PARALLEL_SMOKE_CHECKS_OK', flush=True)
            return
        assert len(rows) == 36
        rows.sort(key=lambda r: (r['backbone'], r['prediction_length'], r['head'], r['weight_decay']))
        selected = []
        for b in ('ttm', 'chronos', 'timer'):
            for h in (96, 720):
                for head in ('linear', 'sacq'):
                    group = [r for r in rows if (r['backbone'], r['prediction_length'], r['head']) == (b, h, head)]
                    winner = min(group, key=lambda r: (r['best_val']['mse'], r['weight_decay']))
                    selected.append(winner)
                    print('SELECT', b, h, head, 'lr=', winner['learning_rate'],
                          'wd=', winner['weight_decay'], winner['best_val'], flush=True)
        core.save_json(core.OUT / 'selected_by_validation.json', selected)
        with (core.OUT / 'validation_summary.csv').open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['backbone', 'horizon', 'head', 'lr', 'weight_decay', 'best_epoch', 'val_mse', 'val_mae'])
            for r in rows:
                w.writerow([r['backbone'], r['prediction_length'], r['head'], r['learning_rate'],
                            r['weight_decay'], r['best_epoch'], r['best_val']['mse'], r['best_val']['mae']])
        print('ALL_36_WD_TRIALS_FINISHED_NO_TEST_EVALUATION', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', nargs='+', type=int, required=True)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus) or any(g < 0 for g in args.gpus):
        parser.error('GPU indices must be unique nonnegative integers')
    if args.launch:
        (ROOT / 'logs').mkdir(exist_ok=True)
        log = ROOT / 'logs' / f'etth1_wd_parallel_{time.strftime("%Y%m%d_%H%M%S")}_{os.getpid()}.log'
        cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--gpus', *map(str, args.gpus)]
        if args.smoke:
            cmd.append('--smoke')
        with log.open('x') as f:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL,
                                    stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        (ROOT / 'logs/etth1_tune_wd_parallel.logpath').write_text(str(log))
        (ROOT / 'logs/etth1_tune_wd_parallel.pid').write_text(str(proc.pid))
        print('Dispatched PID:', proc.pid, 'Log:', log)
    else:
        run(args.gpus, args.smoke)
