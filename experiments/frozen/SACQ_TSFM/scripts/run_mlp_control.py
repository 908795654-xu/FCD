"""ETTh1 parameter-matched MLP: validation-only tuning then three-seed confirmation.
No changes to the installed model factory, SACQ source, batch size, or prior results.
"""
import argparse, datetime, hashlib, json, math, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/mlp_control_v1'
REFERENCE=ROOT/'results/confirmation_fixed_v1/all_results.json'
LRS=(1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,6e-4,1e-3)
BACKBONES=('ttm','chronos','timer')

def width_for(input_dim,horizon,target):
    ideal=(target-horizon)/(input_dim+horizon+1)
    candidates={max(1,math.floor(ideal)),max(1,math.ceil(ideal))}
    width=min(candidates,key=lambda w:(abs(w*(input_dim+horizon+1)+horizon-target),w))
    return width,width*(input_dim+horizon+1)+horizon

def references():
    rows=json.loads(REFERENCE.read_text())
    assert len(rows)==36
    index={}
    for r in rows:
        key=(r['backbone'],r['horizon'],r['variant'],r['seed'])
        assert key not in index and r['complete'] and r['test_evaluated'] and r['backbone_unchanged']
        assert r['config']['batch_size']==32 and r['config']['max_epochs']==50 and r['config']['patience']==7
        assert all(math.isfinite(r['test'][m]) and r['test'][m]>=0 for m in ('mse','mae'))
        index[key]=r
    assert set(index)=={(b,h,v,s) for b in BACKBONES for h in (96,720) for v in ('linear','sacq_pe_on') for s in (2026,2027,2028)}
    return index

def fingerprint():
    receipt=json.loads((ROOT/'installation_manifest.json').read_text())
    assert receipt['profile']=='readme_sacq_default_pe_v2'
    digest=hashlib.sha256()
    for rel,expected in sorted(receipt['installed_hashes'].items()):
        raw=(ROOT/rel).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==expected, f'Source changed: {rel}'
        digest.update(rel.encode());digest.update(raw)
    references()
    for p in (Path(__file__).resolve(),ROOT/'datasets/ETTh1.csv',REFERENCE):digest.update(p.read_bytes())
    return digest.hexdigest()

def job(b,h,lr,wd,drop,seed=2026,final=False):
    tag=f'{"final" if final else "tune"}_lr{lr:.0e}_wd{wd:g}_drop{drop:g}'
    return (b,h,tag,seed)

def config(j):
    b,h,tag,seed=j
    phase,lr,wd,drop=tag.split('_')
    assert b in BACKBONES and h in (96,720) and seed in (2026,2027,2028)
    assert phase in ('tune','final')
    lr,wd,drop=float(lr[2:]),float(wd[2:]),float(drop[4:])
    assert lr in LRS and wd in (0.,.01,.1) and drop in (0.,.1,.2)
    return dict(dataset='etth1',backbone=b,horizon=h,variant=tag,seed=seed,
        batch_size=32,max_epochs=50,patience=7,learning_rate=lr,weight_decay=wd,head_dropout=drop,
        loss='mse',optimizer='AdamW',scheduler='ReduceLROnPlateau',scheduler_patience=1,
        factor=.5,threshold=1e-4,threshold_mode='rel',min_lr=1e-6,context_length=512,
        num_workers=0,selection='validation_mse',evaluate_test_after_selection=phase=='final',
        frozen_backbone=True,architecture='flatten_linear_gelu_dropout_linear_shared_channels',
        parameter_target='actual_sacq_pe_on_trainable_count',width_selection='nearest_integer')

def name(j):return f'etth1_p{j[1]}_{j[0]}_mlp_{j[2]}_seed{j[3]}'

def atomic(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,path)

def build_mlp(**kw):
    import torch
    from torch import nn
    from models.factory_pe import build_forecaster
    # Count the actual corresponding SACQ with future query PE, then replace only its head.
    kw['head_name']='sacq';kw['future_query_pe']=True
    with torch.random.fork_rng(devices=[]):
        model,meta=build_forecaster(**kw)
    target=sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    reference=references()[(kw['backbone_name'],kw['prediction_length'],'sacq_pe_on',2026)]
    assert target==reference['metadata']['trainable_parameters'], 'SACQ parameter target differs from confirmation'
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
    
    from settings import MODEL_PATHS
    b, h, variant, seed = job
    cfg = config(job)
    fp = fingerprint()
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
        ds = BenchmarkDataset(csv_path=ROOT/'datasets/ETTh1.csv', dataset_name='etth1',
                              split=split, context_length=512, prediction_length=h)
        lengths[split] = len(ds)
        loaders[split] = DataLoader(ds, batch_size=32, shuffle=split == 'train',
                                   generator=torch.Generator().manual_seed(seed), num_workers=0,
                                   pin_memory=True)
    model, metadata = build_mlp(
        backbone_name=b, head_name='linear' if variant == 'linear' else 'sacq',
        model_path=MODEL_PATHS[(b, h)], context_length=512, prediction_length=h, n_vars=7,
        future_patch_length=16, cross_attn_layers=2, n_heads=16, head_dropout=cfg['head_dropout'],
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
    optimizer = torch.optim.AdamW(params, lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
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
    checkpoint = ROOT/'checkpoints/mlp_control_v1'/(name(job)+'.pt')
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
        test_ds = BenchmarkDataset(csv_path=ROOT/'datasets/ETTh1.csv', dataset_name='etth1',
                                    split='test', context_length=512, prediction_length=h)
        loaders['test'] = DataLoader(test_ds,batch_size=32,shuffle=False,num_workers=0,pin_memory=True,
                                    generator=torch.Generator().manual_seed(seed))
        lengths['test'] = len(test_ds)
        test_metrics = epoch('test')
        assert frozen_hash == state_hash(model.adapter.state_dict())
        print('TEST', name(job), test_metrics, flush=True)
    record = dict(complete=True, smoke=smoke, config=cfg, fingerprint=fp,
                  backbone=b, horizon=h, variant=variant, seed=seed, metadata=metadata,
                  core_init_sha256=core_init, post_build_cpu_rng_sha256=rng_init,
                  initial_val=initial, best_val=best_val, best_epoch=best_epoch,
                  completed_epochs=len(history), history=history, dataset_lengths=lengths,
                  pe_gradient_checked=gradient_checked, backbone_unchanged=True, test_evaluated=(not smoke and cfg['evaluate_test_after_selection']), test=test_metrics,
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

def best(rows):
    # Selection is exclusively by validation MSE; deterministic ties favor earlier grid entries.
    return min(rows,key=lambda r:r['best_val']['mse'])['config']

def summarize(rows):
    import csv,statistics as st
    reference=references();summary=[]
    for b in BACKBONES:
        for h in (96,720):
            mlp=sorted([r for r in rows if r['backbone']==b and r['horizon']==h],key=lambda r:r['seed'])
            assert [r['seed'] for r in mlp]==[2026,2027,2028]
            for metric in ('mse','mae'):
                vals={v:[reference[(b,h,v,s)]['test'][metric] for s in (2026,2027,2028)] for v in ('linear','sacq_pe_on')}
                vals['mlp']=[r['test'][metric] for r in mlp]
                row=dict(backbone=b,horizon=h,metric=metric,mlp_parameters=mlp[0]['metadata']['trainable_parameters'],
                         sacq_parameters=mlp[0]['metadata']['target_sacq_parameters'],
                         parameter_gap_pct=mlp[0]['metadata']['parameter_gap_pct'])
                for v,x in vals.items():row.update({v+'_mean':st.mean(x),v+'_sd':st.stdev(x)})
                for v in ('mlp','sacq_pe_on'):
                    row[v+'_vs_linear_gain_pct']=100*(st.mean(vals['linear'])-st.mean(vals[v]))/st.mean(vals['linear'])
                row['sacq_vs_mlp_gain_pct']=100*(st.mean(vals['mlp'])-st.mean(vals['sacq_pe_on']))/st.mean(vals['mlp'])
                row['sacq_vs_mlp_wins']=sum(s<m for s,m in zip(vals['sacq_pe_on'],vals['mlp']))
                summary.append(row);print('SUMMARY',json.dumps(row),flush=True)
    with (OUT/'comparison_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    atomic(OUT/'comparison_summary.json',summary)

def queue_run(gpus,smoke):
    import fcntl
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run_dir=ROOT/'logs'/('mlp_control_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid()))
        run_dir.mkdir(parents=True)
        print('GPUs',gpus,'batch=32; max_epochs=50; patience=7; no automatic batch changes',flush=True)
        if smoke:
            run_stage([job(b,h,1e-4,.01,0) for b in BACKBONES for h in (96,720)],gpus,run_dir,True)
            print('ALL_6_MLP_SMOKE_OK_NO_TEST',flush=True);return
        # Validate actual forward, backward and parameter matching before the formal queue.
        run_stage([job(b,h,1e-4,.01,0) for b in BACKBONES for h in (96,720)],gpus,run_dir,True)
        print('ALL_6_MLP_SMOKE_OK_NO_TEST',flush=True)
        # Stage 1: 48 LR trials, seed 2026, validation only.
        lr_rows=run_stage([job(b,h,lr,.01,0) for b in BACKBONES for h in (96,720) for lr in LRS],gpus,run_dir)
        lr_selected={(b,h):best([r for r in lr_rows if r['backbone']==b and r['horizon']==h]) for b in BACKBONES for h in (96,720)}
        # Stage 2: 18 WD candidates (6 reused from LR stage).
        wd_rows=run_stage([job(b,h,c['learning_rate'],wd,0) for (b,h),c in lr_selected.items() for wd in (0.,.01,.1)],gpus,run_dir)
        wd_selected={(b,h):best([r for r in wd_rows if r['backbone']==b and r['horizon']==h]) for b in BACKBONES for h in (96,720)}
        # Stage 3: 18 dropout candidates (6 reused from WD stage).
        drop_rows=run_stage([job(b,h,c['learning_rate'],c['weight_decay'],drop) for (b,h),c in wd_selected.items() for drop in (0.,.1,.2)],gpus,run_dir)
        selected=[best([r for r in drop_rows if r['backbone']==b and r['horizon']==h]) for b in BACKBONES for h in (96,720)]
        atomic(OUT/'selected_validation_configs.json',dict(fingerprint=fingerprint(),configs=selected,selection='seed2026_validation_mse_only'))
        print('MLP_TUNING_FINISHED_NO_TEST; 72 unique validation trials; freezing six configurations',flush=True)
        final_jobs=[job(c['backbone'],c['horizon'],c['learning_rate'],c['weight_decay'],c['head_dropout'],s,True) for c in selected for s in (2026,2027,2028)]
        rows=run_stage(final_jobs,gpus,run_dir)
        assert len(rows)==18
        atomic(OUT/'all_confirmation_results.json',rows);summarize(rows)
        print('ALL_18_MLP_CONFIRMATIONS_FINISHED',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus',type=int,nargs='+',default=[3]);p.add_argument('--gpu',type=int)
    p.add_argument('--launch',action='store_true');p.add_argument('--smoke',action='store_true')
    p.add_argument('--worker',nargs=4);p.add_argument('--report')
    a=p.parse_args()
    assert a.gpus and len(set(a.gpus))==len(a.gpus) and all(g>=0 for g in a.gpus)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONPATH=str(ROOT))
    sys.path.insert(0,str(ROOT));fingerprint()
    if a.worker:
        assert a.gpu is not None and a.gpu>=0 and a.report
        os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
        b,h,v,s=a.worker;j=(b,int(h),v,int(s));config(j)
        report=Path(a.report);assert report.resolve().is_relative_to(ROOT.resolve())
        worker(j,a.smoke,report)
    elif a.launch:
        (ROOT/'logs').mkdir(exist_ok=True)
        log=ROOT/'logs'/('mlp_control_background_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid())+'.log')
        cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--gpus']+list(map(str,a.gpus))
        if a.smoke:cmd+=['--smoke']
        with log.open('x') as f:
            proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        (ROOT/'logs/mlp_control.logpath').write_text(str(log));(ROOT/'logs/mlp_control.pid').write_text(str(proc.pid))
        print('Dispatched PID:',proc.pid,'Log:',log)
    else:queue_run(a.gpus,a.smoke)

if __name__=='__main__':main()
