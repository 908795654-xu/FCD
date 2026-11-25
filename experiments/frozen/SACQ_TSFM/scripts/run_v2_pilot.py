"""18 fixed-protocol ETTh1 trials: Linear, SACQ PE-off, SACQ PE-on; validation only."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/pilot_pe_control_v1'
CONFIG = dict(dataset='etth1', seed=2026, batch_size=32, max_epochs=50, patience=7,
              learning_rate=1e-4, weight_decay=0.01, loss='mse', optimizer='AdamW',
              scheduler='ReduceLROnPlateau', scheduler_patience=1, factor=0.5,
              threshold=1e-4, threshold_mode='rel', min_lr=1e-6,
              context_length=512, future_patch_length=16, cross_attn_layers=2,
              n_heads=16, head_dropout=0.0, num_workers=0, selection='validation_mse',
              test_evaluated=False, frozen_backbone=True)


def jobs():
    return [(b, h, v) for b in ('ttm', 'chronos', 'timer')
            for h in (96, 720) for v in ('linear', 'sacq_pe_off', 'sacq_pe_on')]


def name(job):
    return f'etth1_p{job[1]}_{job[0]}_{job[2]}_seed2026'


def atomic(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def fingerprint():
    receipt = json.loads((ROOT/'installation_manifest.json').read_text())
    assert receipt['profile'] == 'readme_sacq_default_pe_v2'
    digest = hashlib.sha256()
    for rel, expected in sorted(receipt['installed_hashes'].items()):
        raw = (ROOT/rel).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected, f'Source changed: {rel}'
        digest.update(rel.encode())
        digest.update(raw)
    for p in (Path(__file__).resolve(), ROOT/'datasets/ETTh1.csv'):
        digest.update(p.read_bytes())
    digest.update(json.dumps(CONFIG, sort_keys=True).encode())
    return digest.hexdigest()


def worker(job, smoke, report):
    import random
    import numpy as np
    import torch
    import transformers
    from torch.utils.data import DataLoader
    from data_provider import BenchmarkDataset
    from models.factory_pe import build_forecaster
    from settings import MODEL_PATHS
    b, h, variant = job
    fp = fingerprint()
    assert torch.cuda.is_available()
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda:0')
    loaders, lengths = {}, {}
    for split in ('train', 'val'):
        ds = BenchmarkDataset(csv_path=ROOT/'datasets/ETTh1.csv', dataset_name='etth1',
                              split=split, context_length=512, prediction_length=h)
        lengths[split] = len(ds)
        loaders[split] = DataLoader(ds, batch_size=32, shuffle=split == 'train',
                                   generator=torch.Generator().manual_seed(2026), num_workers=0,
                                   pin_memory=True)
    model, metadata = build_forecaster(
        backbone_name=b, head_name='linear' if variant == 'linear' else 'sacq',
        model_path=MODEL_PATHS[(b, h)], context_length=512, prediction_length=h, n_vars=7,
        future_patch_length=16, cross_attn_layers=2, n_heads=16, head_dropout=0,
        future_query_pe=variant == 'sacq_pe_on')
    # Hash before GPU conversion, comparing common weights independently of the PE table.
    def state_hash(state, exclude_pe=False):
        out = hashlib.sha256()
        for key, value in sorted(state.items()):
            if exclude_pe and key == 'query_pos_emb':
                continue
            out.update(key.encode())
            out.update(str((value.dtype, tuple(value.shape))).encode())
            out.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        return out.hexdigest()
    core_init = state_hash(model.head.state_dict(), exclude_pe=True)
    rng_init = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    assert not any(p.requires_grad for p in model.adapter.parameters())
    frozen_hash = state_hash(model.adapter.state_dict())
    model.to(device)
    params = [p for p in model.head.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)
    assert {id(p) for p in params} == {id(p) for p in model.parameters() if p.requires_grad}
    pe = getattr(model.head, 'query_pos_emb', None)
    if variant == 'sacq_pe_on':
        assert pe is not None and pe.requires_grad
        assert id(pe) in {id(p) for p in params}
    elif variant == 'sacq_pe_off':
        assert pe is None
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1, threshold=1e-4,
        threshold_mode='rel', min_lr=1e-6)
    kwargs = {'freq_token': 7} if b == 'ttm' else {}
    gradient_checked = False

    def epoch(split):
        nonlocal gradient_checked
        train = split == 'train'
        model.train(train)
        assert not any(m.training for m in model.adapter.modules())
        total = absolute = 0.0
        count = 0
        with torch.set_grad_enabled(train):
            for batch in loaders[split]:
                x, y = batch['values'].to(device), batch['target'].to(device)
                if train:
                    optimizer.zero_grad(set_to_none=True)
                prediction = model(x, adapter_kwargs=kwargs).prediction
                assert prediction.shape == y.shape
                loss = torch.nn.functional.mse_loss(prediction, y)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss')
                if train:
                    loss.backward()
                    if pe is not None and not gradient_checked:
                        assert pe.grad is not None and torch.isfinite(pe.grad).all()
                        assert pe.grad.abs().sum().item() > 0
                        gradient_checked = True
                    optimizer.step()
                diff = prediction.detach().double() - y.double()
                total += diff.square().sum().item()
                absolute += diff.abs().sum().item()
                count += y.numel()
                if smoke:
                    break
        return dict(mse=total/count, mae=absolute/count)

    initial = epoch('val')
    print('RUN', name(job), 'smoke=', smoke, 'lengths=', lengths, 'initial_val=', initial, flush=True)
    started = time.perf_counter()
    best, stale, history = math.inf, 0, []
    checkpoint = ROOT/'checkpoints/pilot_pe_control_v1'/(name(job)+'.pt')
    if not smoke:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for e in range(1, (1 if smoke else 50)+1):
        tick = time.perf_counter()
        used_lr = optimizer.param_groups[0]['lr']
        train, val = epoch('train'), epoch('val')
        scheduler.step(val['mse'])
        row = dict(epoch=e, train_mse=train['mse'], val_mse=val['mse'], val_mae=val['mae'],
                   learning_rate_used=used_lr, learning_rate_next=optimizer.param_groups[0]['lr'],
                   seconds=time.perf_counter()-tick)
        history.append(row)
        print(f'epoch={e:02d} train_mse={train["mse"]:.6f} val_mse={val["mse"]:.6f} '
              f'val_mae={val["mae"]:.6f} lr={used_lr:.2e} seconds={row["seconds"]:.2f}', flush=True)
        if val['mse'] < best:
            best, best_epoch, best_val, stale = val['mse'], e, val, 0
            if not smoke:
                temp = checkpoint.with_suffix('.tmp')
                torch.save(dict(head_state_dict=model.head.state_dict(), epoch=e, val=val,
                                config=CONFIG, metadata=metadata, variant=variant,
                                fingerprint=fp, core_init_sha256=core_init), temp)
                os.replace(temp, checkpoint)
        else:
            stale += 1
        if stale >= 7:
            print('EARLY_STOP', e, 'best_epoch=', best_epoch, flush=True)
            break
    assert all(p.grad is None for p in model.adapter.parameters())
    assert not any(p.requires_grad for p in model.adapter.parameters())
    assert frozen_hash == state_hash(model.adapter.state_dict())
    assert fingerprint() == fp, 'Source or dataset changed during run'
    record = dict(complete=True, smoke=smoke, config=CONFIG, fingerprint=fp,
                  backbone=b, horizon=h, variant=variant, metadata=metadata,
                  core_init_sha256=core_init, post_build_cpu_rng_sha256=rng_init,
                  initial_val=initial, best_val=best_val, best_epoch=best_epoch,
                  completed_epochs=len(history), history=history, dataset_lengths=lengths,
                  pe_gradient_checked=gradient_checked, backbone_unchanged=True, test_evaluated=False,
                  checkpoint=None if smoke else str(checkpoint), elapsed_seconds=time.perf_counter()-started,
                  peak_allocated_mb=torch.cuda.max_memory_allocated()/1024**2,
                  physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'], torch_version=torch.__version__,
                  transformers_version=transformers.__version__)
    atomic(report, record)
    print('PILOT_JOB_OK', name(job), best_val, flush=True)


def queue_run(smoke):
    import fcntl
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fp = fingerprint()
        run_dir = ROOT/'logs'/('v2_pilot_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid()))
        run_dir.mkdir(parents=True)
        records = []
        for i, job in enumerate(jobs(), 1):
            result = (run_dir if smoke else OUT)/(name(job)+'.json')
            if result.exists() and not smoke:
                print('CHECK_EXISTING', name(job), flush=True)
            else:
                print(f'START [{i}/18]', name(job), flush=True)
                log = run_dir/(name(job)+'.log')
                cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker',
                       job[0], str(job[1]), job[2], '--report', str(result)]
                if smoke:
                    cmd.append('--smoke')
                print('JOB_LOG:', log, flush=True)
                with log.open('x') as stream:
                    subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
            r = json.loads(result.read_text())
            assert r['complete'] and r['fingerprint'] == fp and r['config'] == CONFIG
            assert (r['backbone'], r['horizon'], r['variant']) == job
            assert r['smoke'] == smoke and r['backbone_unchanged'] and not r['test_evaluated']
            assert all(math.isfinite(r['best_val'][m]) and r['best_val'][m] >= 0 for m in ('mse','mae'))
            if not smoke:
                assert Path(r['checkpoint']).is_file()
            records.append(r)
            if job[2] == 'sacq_pe_on':
                off = next(x for x in records if (x['backbone'],x['horizon'],x['variant']) == (job[0],job[1],'sacq_pe_off'))
                assert off['core_init_sha256'] == r['core_init_sha256'], 'Common head initialization mismatch'
                assert off['post_build_cpu_rng_sha256'] == r['post_build_cpu_rng_sha256'], 'RNG mismatch'
                print('PAIRED_INITIALIZATION_OK', job[0], job[1], flush=True)
            print(f'DONE [{i}/18]', name(job), r['best_val'], flush=True)
        assert len(records) == 18
        if not smoke:
            atomic(OUT/'all_results.json', records)
            summary = []
            for b in ('ttm','chronos','timer'):
                for h in (96,720):
                    group = {r['variant']:r for r in records if r['backbone']==b and r['horizon']==h}
                    line = dict(backbone=b, horizon=h, **{v:group[v]['best_val'] for v in group})
                    summary.append(line)
                    print('SUMMARY', json.dumps(line), flush=True)
            atomic(OUT/'comparison.json', summary)
        print('ALL_18_V2_PILOT_SMOKE_OK' if smoke else 'ALL_18_V2_PILOT_FINISHED_NO_TEST_EVALUATION', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--worker', nargs=3)
    parser.add_argument('--report')
    args = parser.parse_args()
    if args.gpu is not None:
        if args.gpu < 0:
            parser.error('GPU index must be nonnegative')
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    if not os.environ.get('CUDA_VISIBLE_DEVICES', '').isdigit():
        parser.error('Specify one physical GPU with --gpu N')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2',
                      MKL_NUM_THREADS='2', PYTHONPATH=str(ROOT))
    sys.path.insert(0, str(ROOT))
    fingerprint()
    if args.worker:
        b,h,v = args.worker
        job = (b,int(h),v)
        assert job in jobs() and args.report
        report = Path(args.report)
        assert report.resolve().is_relative_to(ROOT.resolve())
        worker(job,args.smoke,report)
    elif args.launch:
        (ROOT/'logs').mkdir(exist_ok=True)
        log = ROOT/'logs'/('v2_pilot_background_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid())+'.log')
        cmd = [sys.executable,'-u',str(Path(__file__).resolve())]
        if args.smoke:
            cmd.append('--smoke')
        with log.open('x') as stream:
            proc = subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,
                                    stderr=subprocess.STDOUT,start_new_session=True)
        (ROOT/'logs/v2_pilot.logpath').write_text(str(log))
        (ROOT/'logs/v2_pilot.pid').write_text(str(proc.pid))
        print('Dispatched PID:',proc.pid,'Log:',log)
    else:
        queue_run(args.smoke)
