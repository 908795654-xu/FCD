"""Matched MLP controls for Weather and Electricity, with frozen_v1 references."""
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
BACKBONES = ('ttm', 'chronos', 'timer')
SEEDS = (2026, 2027, 2028)
BATCH = {'weather': 128, 'electricity': 8}
DATASET = None
OUT = None


def select_dataset(dataset):
    global DATASET, OUT
    DATASET = dataset
    OUT = ROOT / 'results/mlp_control_frozen_v1' / dataset


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
    return digest.hexdigest()


def job(backbone, horizon, seed):
    return backbone, horizon, 'mlp', seed


def config(j):
    backbone,horizon,variant,seed = j
    assert backbone in BACKBONES and horizon in (96,720) and variant == 'mlp' and seed in SEEDS
    return dict(dataset=DATASET, backbone=backbone, horizon=horizon, variant=variant, seed=seed,
                batch_size=BATCH[DATASET], max_epochs=50, patience=7, learning_rate=1e-4,
                weight_decay=.01, head_dropout=0., context_length=512, loss='mse', optimizer='AdamW',
                scheduler='ReduceLROnPlateau', scheduler_patience=1, factor=.5, threshold=1e-4,
                threshold_mode='rel', min_lr=1e-6, num_workers=0, selection='validation_mse',
                evaluate_test_after_selection=True, frozen_backbone=True,
                parameter_target='legacy_frozen_v1_sacq', training_protocol='frozen_v1')


def name(j):
    return f'{DATASET}_p{j[1]}_{j[0]}_mlp_seed{j[3]}_frozen_v1'


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
    model, metadata = build_mlp(
        backbone_name=b, head_name='linear' if variant == 'linear' else 'sacq',
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
                        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
                        assert any(p.grad.abs().sum().item() > 0 for p in params)
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
    checkpoint = ROOT/'checkpoints/mlp_control_frozen_v1'/DATASET/(name(job)+'.pt')
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
                  mlp_gradient_checked=gradient_checked, backbone_unchanged=True, test_evaluated=(not smoke and cfg['evaluate_test_after_selection']), test=test_metrics,
                  checkpoint=None if smoke else str(checkpoint), elapsed_seconds=time.perf_counter()-started,
                  peak_allocated_mb=torch.cuda.max_memory_allocated()/1024**2,
                  physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'], torch_version=torch.__version__,
                  transformers_version=transformers.__version__)
    atomic(report, record)
    print('MLP_JOB_OK', name(job), best_val, flush=True)

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
                    cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--gpu',str(gpu),'--dataset',DATASET,
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

def summarize(rows):
    import csv,statistics as st
    reference=references();summary=[]
    for b in BACKBONES:
        for h in (96,720):
            mlp=sorted([r for r in rows if r['backbone']==b and r['horizon']==h],key=lambda r:r['seed'])
            assert [r['seed'] for r in mlp]==[2026,2027,2028]
            for metric in ('mse','mae'):
                vals={v:[reference[(b,h,v,s)]['test'][metric] for s in (2026,2027,2028)] for v in ('linear','sacq')}
                vals['mlp']=[r['test'][metric] for r in mlp]
                row=dict(dataset=DATASET,backbone=b,horizon=h,metric=metric,mlp_parameters=mlp[0]['metadata']['trainable_parameters'],
                         sacq_parameters=mlp[0]['metadata']['target_sacq_parameters'],
                         parameter_gap_pct=mlp[0]['metadata']['parameter_gap_pct'])
                for v,x in vals.items():row.update({v+'_mean':st.mean(x),v+'_sd':st.stdev(x)})
                for v in ('mlp','sacq'):
                    row[v+'_vs_linear_gain_pct']=100*(st.mean(vals['linear'])-st.mean(vals[v]))/st.mean(vals['linear'])
                row['sacq_vs_mlp_gain_pct']=100*(st.mean(vals['mlp'])-st.mean(vals['sacq']))/st.mean(vals['mlp'])
                row['sacq_vs_mlp_wins']=sum(s<m for s,m in zip(vals['sacq'],vals['mlp']))
                summary.append(row);print('SUMMARY',json.dumps(row),flush=True)
    with (OUT/'comparison_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    atomic(OUT/'comparison_summary.json',summary)


def queue_run(datasets, gpus, smoke_only):
    import fcntl
    output_root = ROOT / 'results/mlp_control_frozen_v1'
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = ROOT/'logs'/f'mlp_weather_electricity_{stamp}_{os.getpid()}'
        run_dir.mkdir(parents=True)
        for dataset in datasets:
            select_dataset(dataset)
            fingerprint()
            OUT.mkdir(parents=True, exist_ok=True)
            print('DATASET', dataset, 'BATCH', BATCH[dataset], 'GPUS', gpus, flush=True)
            run_stage([job(b,h,2026) for b in BACKBONES for h in (96,720)], gpus, run_dir, True)
            print('ALL_6_MLP_SMOKE_OK', dataset, flush=True)
        if smoke_only:
            return
        for dataset in datasets:
            select_dataset(dataset)
            items = [job(b,h,seed) for b in BACKBONES for h in (96,720) for seed in SEEDS]
            rows = run_stage(items, gpus, run_dir)
            atomic(OUT/'all_confirmation_results.json', rows)
            summarize(rows)
            print('ALL_18_MLP_CONFIRMATIONS_FINISHED', dataset, flush=True)
        print('MLP_FIXED_PROTOCOL_QUEUE_FINISHED', ','.join(datasets), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('both', *BATCH), default='both')
    parser.add_argument('--gpus', type=int, nargs='+')
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
        assert args.dataset in BATCH and args.gpu is not None and args.gpu >= 0 and args.report
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        select_dataset(args.dataset)
        backbone,horizon,variant,seed = args.worker
        report = Path(args.report)
        assert report.resolve().is_relative_to(ROOT.resolve())
        worker((backbone,int(horizon),variant,int(seed)), args.smoke, report)
        return
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or any(g < 0 for g in args.gpus):
        parser.error('Specify distinct available physical GPUs with --gpus')
    datasets = list(BATCH) if args.dataset == 'both' else [args.dataset]
    if args.launch:
        for dataset in datasets:
            select_dataset(dataset)
            fingerprint()
        (ROOT/'logs').mkdir(exist_ok=True)
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log = ROOT/'logs'/f'mlp_weather_electricity_background_{stamp}_{os.getpid()}.log'
        cmd = [sys.executable,'-u',str(Path(__file__)), '--dataset',args.dataset, '--gpus',*map(str,args.gpus)]
        if args.smoke:
            cmd.append('--smoke')
        with log.open('x') as stream:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        (ROOT/'logs/mlp_weather_electricity.logpath').write_text(str(log))
        print('Dispatched PID:',proc.pid,'Log:',log)
    else:
        queue_run(datasets, args.gpus, args.smoke)


if __name__ == '__main__':
    main()
