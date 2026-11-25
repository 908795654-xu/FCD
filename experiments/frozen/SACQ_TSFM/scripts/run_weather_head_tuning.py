"""Weather head tuning on validation, followed by three-seed test confirmation."""
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
DATASET = 'weather'
BATCH = {DATASET: 128}
OUT = ROOT / 'results/weather_head_tuning_v1'
BACKBONES = ('ttm', 'chronos', 'timer')
HEADS = ('linear', 'mlp', 'sacq')
SEEDS = (2026, 2027, 2028)
# Same candidate grid for each head. Values are (lr, weight decay, dropout).
GRID = {
    'base': (1e-4, .01, 0.),
    'lr_low': (3e-5, .01, 0.),
    'lr_high': (3e-4, .01, 0.),
    'wd_high': (1e-4, .1, 0.),
    'drop': (1e-4, .01, .1),
}


def job(backbone, horizon, head, candidate, seed, final=False):
    return backbone, horizon, f'{head}:{candidate}:{"final" if final else "tune"}', seed


def config(item):
    backbone, horizon, variant, seed = item
    head, candidate, phase = variant.split(':')
    assert backbone in BACKBONES and horizon in (96, 720) and head in HEADS
    assert phase in ('tune', 'final') and seed in SEEDS
    assert phase == 'final' or seed == SEEDS[0]
    lr, wd, dropout = GRID[candidate]
    return dict(dataset=DATASET, backbone=backbone, horizon=horizon, variant=variant,
                head=head, candidate=candidate, seed=seed, batch_size=128, max_epochs=50,
                patience=7, learning_rate=lr, weight_decay=wd, head_dropout=dropout,
                context_length=512, loss='mse', optimizer='AdamW', selection='validation_mse',
                evaluate_test_after_selection=phase == 'final', frozen_backbone=True)


def name(item):
    backbone, horizon, variant, seed = item
    tag = variant.replace(':', '_')
    return f'weather_p{horizon}_{backbone}_{tag}_seed{seed}'


def build_head(cfg, **kwargs):
    from models import build_forecaster

    if cfg['head'] == 'mlp':
        return build_mlp(**kwargs)
    kwargs['head_name'] = cfg['head']
    model, metadata = build_forecaster(**kwargs)
    ref = references()[cfg['backbone'], cfg['horizon'], cfg['head'], cfg['seed']]
    assert metadata['trainable_parameters'] == ref['trainable_parameters']
    assert getattr(model.head, 'query_pos_emb', None) is None
    return model, metadata


def references():
    index = {}
    for backbone in BACKBONES:
        for horizon in (96, 720):
            for head in ('linear', 'sacq'):
                for seed in SEEDS:
                    path = ROOT / 'results/formal' / f'{DATASET}_p{horizon}_{backbone}_{head}_seed{seed}_frozen_v1.json'
                    row = json.loads(path.read_text(encoding='utf-8'))
                    expected = dict(backbone=backbone, head=head, prediction_length=horizon,
                                    seed=seed, context_length=512, batch_size=BATCH[DATASET],
                                    max_epochs=50, patience=7, learning_rate=1e-4,
                                    weight_decay=.01, loss='mse', backbone_unchanged=True)
                    for field, value in expected.items():
                        if row.get(field) != value:
                            raise ValueError(f'{path.name}: {field}={row.get(field)!r}; expected {value!r}')
                    assert row['dataset'].lower() == DATASET, path
                    assert row.get('training_protocol', 'frozen_v1') == 'frozen_v1', path
                    assert isinstance(row['trainable_parameters'], int) and row['trainable_parameters'] > 0, path
                    assert all(math.isfinite(row['test'][m]) and row['test'][m] >= 0 for m in ('mse','mae')), path
                    index[backbone,horizon,head,seed] = row
    return index

def fingerprint():
    digest = hashlib.sha256()
    paths = [Path(__file__), ROOT / 'scripts/train.py', ROOT / 'datasets' / (DATASET + '.csv')]
    for folder in ('models','adapters','sacq_core','data_provider'):
        paths.extend(sorted((ROOT/folder).rglob('*.py')))
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    digest.update(json.dumps(list(references().values()), sort_keys=True).encode())
    mlp_reference = ROOT / 'results/mlp_control_frozen_v1/weather/all_confirmation_results.json'
    digest.update(mlp_reference.read_bytes())
    return digest.hexdigest()

def width_for(input_dim,horizon,target):
    ideal=(target-horizon)/(input_dim+horizon+1)
    candidates={max(1,math.floor(ideal)),max(1,math.ceil(ideal))}
    width=min(candidates,key=lambda w:(abs(w*(input_dim+horizon+1)+horizon-target),w))
    return width,width*(input_dim+horizon+1)+horizon

def atomic(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,path)

def build_mlp(**kw):
    import torch
    from torch import nn
    from models import build_forecaster
    # Count the actual corresponding SACQ from the legacy factory, then replace only its head.
    kw['head_name'] = 'sacq'
    with torch.random.fork_rng(devices=[]):
        model,meta=build_forecaster(**kw)
    target=sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    for seed in SEEDS:
        row = references()[(kw['backbone_name'],kw['prediction_length'],'sacq',seed)]
        assert target == row['trainable_parameters'], 'Legacy SACQ parameter target differs from baseline'
        assert meta['hidden_dim'] == row['hidden_dim'] and meta['patch_count'] == row['patch_count']
    assert getattr(model.head, 'query_pos_emb', None) is None, 'Expected legacy SACQ without future query PE' 
    d,p,h=meta['hidden_dim'],meta['patch_count'],kw['prediction_length']
    width,count=width_for(d*p,h,target)
    class MatchedMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.network=nn.Sequential(nn.Flatten(start_dim=-2),nn.Linear(d*p,width),nn.GELU(),
                                       nn.Dropout(kw['head_dropout']),nn.Linear(width,h))
        def forward(self,memory):
            assert memory.ndim==4 and tuple(memory.shape[-2:])==(d,p)
            return self.network(memory)
    model.head=MatchedMLP()
    assert sum(p.numel() for p in model.head.parameters() if p.requires_grad)==count
    assert abs(count-target)/target<.01
    meta=dict(meta,head='mlp',future_query_pe=False,pe_learnable=False,pe_initializer=None,
              trainable_parameters=count,target_sacq_parameters=target,
              mlp_hidden_width=width,parameter_gap_pct=100*(count-target)/target)
    print('MLP_PARAMETER_MATCH',json.dumps(meta),flush=True)
    return model,meta

def worker(job, smoke, report):
    import random
    import numpy as np
    import torch
    import transformers
    from torch.utils.data import DataLoader
    from data_provider import BenchmarkDataset
    
    from scripts.train import DATASET_CONFIGS, MODEL_PATHS, get_adapter_kwargs
    b, h, variant, seed = job
    cfg = config(job)
    fp = fingerprint()
    dataset_cfg = DATASET_CONFIGS[DATASET]
    assert dataset_cfg == dict(filename=DATASET+'.csv', n_vars={'weather':21,'electricity':321}[DATASET],
                               freq_token={'weather':4,'electricity':7}[DATASET])
    reference = references()[(b,h,'sacq',seed)]
    assert torch.cuda.is_available()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda:0')
    loaders, lengths = {}, {}
    for split in ('train', 'val'):
        ds = BenchmarkDataset(csv_path=ROOT/'datasets'/dataset_cfg['filename'], dataset_name=DATASET,
                              split=split, context_length=512, prediction_length=h)
        lengths[split] = len(ds)
        loaders[split] = DataLoader(ds, batch_size=cfg['batch_size'], shuffle=split == 'train',
                                   generator=torch.Generator().manual_seed(seed), num_workers=0,
                                   pin_memory=True)
    model, metadata = build_head(
        cfg,
        backbone_name=b, head_name=cfg['head'],
        model_path=MODEL_PATHS[(b, h)], context_length=512, prediction_length=h, n_vars=dataset_cfg['n_vars'],
        future_patch_length=16, cross_attn_layers=2, n_heads=16, head_dropout=cfg['head_dropout'],
    )
    def state_hash(state):
        out = hashlib.sha256()
        for key, value in sorted(state.items()):
            out.update(key.encode())
            out.update(str((value.dtype, tuple(value.shape))).encode())
            out.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        return out.hexdigest()
    core_init = state_hash(model.head.state_dict())
    rng_init = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    assert not any(p.requires_grad for p in model.adapter.parameters())
    frozen_hash = state_hash(model.adapter.state_dict())
    model.to(device)
    params = [p for p in model.head.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    assert {id(p) for p in params} == {id(p) for p in model.parameters() if p.requires_grad}
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1, threshold=1e-4,
        threshold_mode='rel', min_lr=1e-6)
    kwargs = get_adapter_kwargs(b, DATASET)
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
                    if not gradient_checked:
                        assert all(torch.isfinite(p.grad).all() for p in params if p.grad is not None)
                        assert any(p.grad.abs().sum().item() > 0 for p in params if p.grad is not None)
                        gradient_checked = True
                    optimizer.step()
                diff = prediction.detach().double() - y.double()
                total += diff.square().sum().item()
                absolute += diff.abs().sum().item()
                count += y.numel()
                if smoke:
                    break
        return dict(mse=total/count, mae=absolute/count)

    if 'dataset_lengths' in reference:
        assert all(reference['dataset_lengths'][s] == lengths[s] for s in ('train','val'))
    initial = epoch('val')
    print('RUN', name(job), 'smoke=', smoke, 'lengths=', lengths, 'initial_val=', initial, flush=True)
    started = time.perf_counter()
    best, stale, history = math.inf, 0, []
    checkpoint = ROOT/'checkpoints/weather_head_tuning_v1'/(name(job)+'.pt')
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
                                config=cfg, metadata=metadata, variant=variant,
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
    test_metrics = None
    if not smoke and cfg['evaluate_test_after_selection']:
        best_checkpoint = torch.load(checkpoint, map_location=device, weights_only=False)
        assert best_checkpoint['fingerprint'] == fp
        model.head.load_state_dict(best_checkpoint['head_state_dict'])
        test_ds = BenchmarkDataset(csv_path=ROOT/'datasets'/dataset_cfg['filename'], dataset_name=DATASET,
                                    split='test', context_length=512, prediction_length=h)
        loaders['test'] = DataLoader(test_ds,batch_size=cfg['batch_size'],shuffle=False,num_workers=0,pin_memory=True,
                                    generator=torch.Generator().manual_seed(seed))
        lengths['test'] = len(test_ds)
        if 'dataset_lengths' in reference:
            assert reference['dataset_lengths']['test'] == lengths['test']
        test_metrics = epoch('test')
        assert frozen_hash == state_hash(model.adapter.state_dict())
        print('TEST', name(job), test_metrics, flush=True)
    record = dict(complete=True, smoke=smoke, config=cfg, fingerprint=fp,
                  backbone=b, horizon=h, variant=variant, seed=seed, metadata=metadata,
                  core_init_sha256=core_init, post_build_cpu_rng_sha256=rng_init,
                  initial_val=initial, best_val=best_val, best_epoch=best_epoch,
                  completed_epochs=len(history), history=history, dataset_lengths=lengths,
                  head_gradient_checked=gradient_checked, backbone_unchanged=True, test_evaluated=(not smoke and cfg['evaluate_test_after_selection']), test=test_metrics,
                  checkpoint=None if smoke else str(checkpoint), elapsed_seconds=time.perf_counter()-started,
                  peak_allocated_mb=torch.cuda.max_memory_allocated()/1024**2,
                  physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'], torch_version=torch.__version__,
                  transformers_version=transformers.__version__)
    atomic(report, record)
    print('WEATHER_HEAD_JOB_OK', name(job), best_val, flush=True)

def read_result(path,j,smoke=False):
    r=json.loads(path.read_text())
    assert r['complete'] and r['config']==config(j) and r['fingerprint']==fingerprint()
    assert r['smoke']==smoke and r['backbone_unchanged']
    expected=not smoke and config(j)['evaluate_test_after_selection']
    assert r['test_evaluated']==expected
    assert all(math.isfinite(r['best_val'][m]) and r['best_val'][m]>=0 for m in ('mse','mae'))
    if not smoke:assert Path(r['checkpoint']).is_file()
    if expected:assert all(math.isfinite(r['test'][m]) and r['test'][m]>=0 for m in ('mse','mae'))
    else:assert r['test'] is None and 'test' not in r['dataset_lengths']
    return r

def run_stage(items,gpus,run_dir,smoke=False):
    from concurrent.futures import ThreadPoolExecutor
    import queue
    tasks=queue.Queue()
    for j in items:tasks.put(j)
    results={};failures=[]
    def consume(gpu):
        while not failures:
            try:j=tasks.get_nowait()
            except queue.Empty:return
            path=(run_dir if smoke else OUT)/(name(j)+('_smoke' if smoke else '')+'.json')
            try:
                if not path.exists():
                    log=run_dir/(name(j)+('_smoke' if smoke else '')+'.log')
                    cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--gpu',str(gpu),
                         '--worker',j[0],str(j[1]),j[2],str(j[3]),'--report',str(path)]
                    if smoke:cmd+=['--smoke']
                    print('START GPU',gpu,name(j),'LOG',log,flush=True)
                    with log.open('x') as f:subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
                results[j]=read_result(path,j,smoke)
                print('DONE',name(j),'val=',results[j]['best_val'],flush=True)
            except Exception as exc:
                failures.append((j,str(exc)))
                print('FAILED',name(j),str(exc),'No batch changes; active jobs finish.',flush=True)
            finally:tasks.task_done()
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:list(pool.map(consume,gpus))
    if failures:raise RuntimeError(str(failures))
    return [results[j] for j in items]


def original_mlp():
    path = ROOT / 'results/mlp_control_frozen_v1/weather/all_confirmation_results.json'
    rows = json.loads(path.read_text(encoding='utf-8'))
    result = {}
    for row in rows:
        cfg = row['config']
        key = row['backbone'], row['horizon'], 'mlp', row['seed']
        assert key not in result
        assert row['complete'] and row['test_evaluated'] and row['backbone_unchanged']
        assert (cfg['dataset'], cfg['batch_size'], cfg['max_epochs'], cfg['patience']) == ('weather', 128, 50, 7)
        assert (cfg['learning_rate'], cfg['weight_decay'], cfg['head_dropout']) == GRID['base']
        assert all(math.isfinite(row['test'][metric]) for metric in ('mse', 'mae'))
        result[key] = row
    assert set(result) == {(b,h,'mlp',s) for b in BACKBONES for h in (96,720) for s in SEEDS}
    return result


def summarize(rows, horizons, output):
    import statistics
    before = {**references(), **original_mlp()}
    after = {(r['backbone'], r['horizon'], r['config']['head'], r['seed']): r for r in rows}
    report = []
    for b in BACKBONES:
        for h in horizons:
            for metric in ('mse', 'mae'):
                result = dict(dataset=DATASET, backbone=b, horizon=h, metric=metric)
                samples = {}
                for head in HEADS:
                    keys = [(b,h,head,s) for s in SEEDS]
                    old = [before[k]['test'][metric] for k in keys]
                    new = [after[k]['test'][metric] for k in keys]
                    samples[head] = new
                    result[head+'_old_mean'] = statistics.mean(old)
                    result[head+'_mean'] = statistics.mean(new)
                    result[head+'_sd'] = statistics.stdev(new)
                    result[head+'_gain_pct'] = 100 * (1 - statistics.mean(new) / statistics.mean(old))
                    result[head+'_candidate'] = after[keys[0]]['config']['candidate']
                for head in ('linear', 'mlp'):
                    result[f'sacq_vs_{head}_gain_pct'] = 100 * (1 - result['sacq_mean'] / result[head+'_mean'])
                    result[f'sacq_vs_{head}_wins'] = sum(a < b for a,b in zip(samples['sacq'], samples[head]))
                print('SUMMARY', json.dumps(result), flush=True)
                report.append(result)
    atomic(output, report)


def queue_run(horizons, gpus, smoke_only):
    import fcntl
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fingerprint()
        original_mlp()
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = ROOT/'logs'/f'weather_tuning_{stamp}_{os.getpid()}'
        run_dir.mkdir(parents=True)
        groups = [(b,h,head) for b in BACKBONES for h in horizons for head in HEADS]
        run_stage([job(*group,'base',SEEDS[0]) for group in groups], gpus, run_dir, True)
        print('WEATHER_SMOKE_OK', len(groups), flush=True)
        if smoke_only:
            return
        trials = [job(*group,candidate,SEEDS[0]) for group in groups for candidate in GRID]
        tuning = run_stage(trials, gpus, run_dir)
        choices = {}
        for group in groups:
            matches = [r for r in tuning if (r['backbone'],r['horizon'],r['config']['head']) == group]
            assert len(matches) == len(GRID) and all(not r['test_evaluated'] for r in matches)
            selected = min(matches, key=lambda r: r['best_val']['mse'])
            choices[group] = selected['config']['candidate']
            print('SELECT', group, choices[group], selected['best_val'], flush=True)
        scope = '_'.join(map(str,horizons))
        atomic(OUT/f'selection_p{scope}.json', [dict(backbone=b,horizon=h,head=head,candidate=c)
                                                for (b,h,head),c in choices.items()])
        confirmations = [job(*group,candidate,seed,final=True)
                         for group,candidate in choices.items() for seed in SEEDS]
        rows = run_stage(confirmations, gpus, run_dir)
        atomic(OUT/f'confirmations_p{scope}.json', rows)
        summarize(rows, horizons, OUT/f'comparison_p{scope}.json')
        print('WEATHER_TUNING_CONFIRMATION_FINISHED', len(rows), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--horizons', nargs='+', type=int, choices=(96,720), default=[96])
    parser.add_argument('--gpus', nargs='+', type=int)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--worker', nargs=4)
    parser.add_argument('--report')
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2',
                      MKL_NUM_THREADS='2', PYTHONPATH=str(ROOT))
    sys.path.insert(0, str(ROOT))
    if args.worker:
        assert args.gpu is not None and args.gpu >= 0 and args.report
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        b,h,v,s = args.worker
        report = Path(args.report).resolve()
        assert report.is_relative_to(ROOT)
        worker((b,int(h),v,int(s)), args.smoke, report)
        return
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0:
        parser.error('Specify available physical GPUs with --gpus')
    horizons = sorted(set(args.horizons))
    if not args.launch:
        queue_run(horizons, args.gpus, args.smoke)
        return
    fingerprint()
    original_mlp()
    (ROOT/'logs').mkdir(exist_ok=True)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log = ROOT/'logs'/f'weather_tuning_background_{stamp}_{os.getpid()}.log'
    command = [sys.executable,'-u',str(Path(__file__).resolve()),'--horizons',*map(str,horizons),
               '--gpus',*map(str,args.gpus)]
    if args.smoke:
        command.append('--smoke')
    with log.open('x') as stream:
        proc = subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,
                                stderr=subprocess.STDOUT,start_new_session=True)
    (ROOT/'logs/weather_tuning.logpath').write_text(str(log))
    print('Dispatched PID:',proc.pid,'Log:',log)


if __name__ == '__main__':
    main()
