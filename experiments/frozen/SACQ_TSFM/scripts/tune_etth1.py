"""Frozen-head ETTh1 learning-rate search; validation only, no test access."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
LRS = (1e-5, 3e-5, 1e-4, 3e-4)
OUT = ROOT / 'results' / 'tuning_etth1_lr_v1'


def identity():
    paths = [Path(__file__), ROOT / 'datasets' / 'ETTh1.csv']
    for folder in ('models', 'adapters', 'sacq_core', 'data_provider'):
        paths.extend(sorted((ROOT / folder).rglob('*.py')))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def jobs():
    return [(b, h, head, lr) for b in ('ttm', 'chronos', 'timer')
            for h in (96, 720) for head in ('linear', 'sacq') for lr in LRS]


def name(job):
    b, h, head, lr = job
    return f'etth1_p{h}_{b}_{head}_seed2026_lr{lr:.0e}'


def save_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def worker(job, smoke):
    import random
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from data_provider import BenchmarkDataset
    from models import build_forecaster
    from scripts.train import MODEL_PATHS

    b, h, head, lr = job
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    device = torch.device('cuda:0')
    loaders = {}
    for split in ('train', 'val'):
        ds = BenchmarkDataset(csv_path=ROOT / 'datasets' / 'ETTh1.csv',
                              dataset_name='etth1', split=split,
                              context_length=512, prediction_length=h)
        loaders[split] = DataLoader(ds, batch_size=32, shuffle=split == 'train',
                                   generator=torch.Generator().manual_seed(2026),
                                   num_workers=0, pin_memory=True)
    model, metadata = build_forecaster(
        backbone_name=b, head_name=head, model_path=MODEL_PATHS[(b, h)],
        context_length=512, prediction_length=h, n_vars=7,
        future_patch_length=16, cross_attn_layers=2, n_heads=16,
        head_dropout=0.0)
    model.to(device)
    backbone = list(model.adapter.parameters())
    assert backbone and not any(p.requires_grad for p in backbone)
    # Exact before/after frozen parameter verification, including integer buffers.
    snapshot = {k: v.detach().cpu().clone() for k, v in model.adapter.state_dict().items()}
    opt = torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=1, min_lr=1e-6,
        threshold=1e-4, threshold_mode='rel')
    kwargs = {'freq_token': 7} if b == 'ttm' else {}

    def epoch(split):
        training = split == 'train'
        model.train(training)
        assert not any(m.training for m in model.adapter.modules())
        total, absolute, count = 0.0, 0.0, 0
        with torch.set_grad_enabled(training):
            for i, batch in enumerate(loaders[split]):
                x = batch['values'].to(device)
                y = batch['target'].to(device)
                if training:
                    opt.zero_grad(set_to_none=True)
                pred = model(x, adapter_kwargs=kwargs).prediction
                assert pred.shape == y.shape
                loss = torch.nn.functional.mse_loss(pred, y)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss')
                if training:
                    loss.backward()
                    opt.step()
                diff = pred.detach().double() - y.double()
                total += diff.square().sum().item()
                absolute += diff.abs().sum().item()
                count += y.numel()
                if smoke:
                    break
        return dict(mse=total / count, mae=absolute / count)

    initial = epoch('val')
    # Match earlier training protocol: select from trained epochs 1..50.
    best, best_epoch, stale, history = math.inf, None, 0, []
    checkpoint = OUT / (name(job) + '.pt')
    started = time.perf_counter()
    for e in range(1, (1 if smoke else 50) + 1):
        train = epoch('train')
        val = epoch('val')
        history.append(dict(epoch=e, train=train, val=val,
                            learning_rate=opt.param_groups[0]['lr']))
        sched.step(val['mse'])
        print(name(job), f'epoch={e} train={train["mse"]:.6f} val={val["mse"]:.6f}', flush=True)
        if val['mse'] < best:
            best, best_epoch, stale = val['mse'], e, 0
            best_metrics = val
            if not smoke:
                tmp = checkpoint.with_suffix('.tmp')
                torch.save(dict(head_state_dict=model.head.state_dict(), epoch=e,
                                val_metrics=val, metadata=metadata), tmp)
                os.replace(tmp, checkpoint)
        else:
            stale += 1
        if stale >= 7:
            break
    unchanged = all(torch.equal(snapshot[k], v.detach().cpu())
                    for k, v in model.adapter.state_dict().items())
    assert unchanged
    if not smoke:
        save_json(OUT / (name(job) + '.json'), dict(
            complete=True, fingerprint=identity(), backbone=b, prediction_length=h,
            head=head, learning_rate=lr, seed=2026, batch_size=32,
            weight_decay=0.01, max_epochs=50, patience=7, initial_val=initial,
            best_val=best_metrics, best_epoch=best_epoch, history=history,
            backbone_unchanged=unchanged, checkpoint=str(checkpoint),
            elapsed_seconds=time.perf_counter() - started, selection='validation_mse',
            test_evaluated=False))


def queue(smoke):
    import fcntl
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fingerprint = identity()
        rows = []
        for i, job in enumerate(jobs(), 1):
            path = OUT / (name(job) + '.json')
            if not smoke and path.exists():
                data = json.loads(path.read_text())
                if not (data['complete'] and data['fingerprint'] == fingerprint
                        and data['backbone_unchanged'] and not data['test_evaluated']
                        and math.isfinite(data['best_val']['mse'])
                        and Path(data['checkpoint']).is_file()):
                    raise RuntimeError(f'Existing result mismatch: {path}')
                print('SKIP', name(job), flush=True)
            else:
                print(f'START [{i}/48]', name(job), flush=True)
                cmd = [sys.executable, '-u', str(Path(__file__).resolve()),
                       '--worker', job[0], str(job[1]), job[2], str(job[3])]
                if smoke:
                    cmd.append('--smoke')
                subprocess.run(cmd, cwd=ROOT, check=True)
                if not smoke:
                    data = json.loads(path.read_text())
            if not smoke:
                rows.append(data)
        if smoke:
            print('ALL_48_SMOKE_CHECKS_OK', flush=True)
            return
        selected = []
        for b in ('ttm', 'chronos', 'timer'):
            for h in (96, 720):
                for head in ('linear', 'sacq'):
                    group = [r for r in rows if (r['backbone'], r['prediction_length'], r['head']) == (b, h, head)]
                    winner = min(group, key=lambda r: (r['best_val']['mse'], r['learning_rate']))
                    selected.append(winner)
                    print('SELECT', b, h, head, winner['learning_rate'], winner['best_val'], flush=True)
        save_json(OUT / 'selected_by_validation.json', selected)
        with (OUT / 'validation_summary.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['backbone', 'horizon', 'head', 'lr', 'best_epoch', 'val_mse', 'val_mae'])
            for r in rows:
                writer.writerow([r['backbone'], r['prediction_length'], r['head'],
                                 r['learning_rate'], r['best_epoch'], r['best_val']['mse'], r['best_val']['mae']])
        print('ALL_48_LR_TRIALS_FINISHED_NO_TEST_EVALUATION', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', type=int)
    p.add_argument('--launch', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--worker', nargs=4)
    args = p.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONPATH=str(ROOT))
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    if not os.environ.get('CUDA_VISIBLE_DEVICES', '').isdigit():
        p.error('Specify one physical GPU with --gpu N')
    sys.path.insert(0, str(ROOT))
    if args.worker:
        b, h, head, lr = args.worker
        worker((b, int(h), head, float(lr)), args.smoke)
    elif args.launch:
        (ROOT / 'logs').mkdir(exist_ok=True)
        log = ROOT / 'logs' / f'etth1_tune_{time.strftime("%Y%m%d_%H%M%S")}_{os.getpid()}.log'
        command = [sys.executable, '-u', str(Path(__file__).resolve())]
        if args.smoke:
            command.append('--smoke')
        with log.open('x') as f:
            proc = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                    stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        (ROOT / 'logs' / 'etth1_tune.logpath').write_text(str(log))
        (ROOT / 'logs' / 'etth1_tune.pid').write_text(str(proc.pid))
        print('Dispatched PID:', proc.pid, 'Log:', log)
    else:
        queue(args.smoke)
