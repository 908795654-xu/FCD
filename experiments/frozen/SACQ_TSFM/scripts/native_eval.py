"""Native pretrained heads: test-only evaluation, no downstream fitting."""
import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'native_v1'
BATCH = {'etth1': 32, 'weather': 128, 'electricity': 8}
CHUNK = 128  # Fixed inference-only chunk of independent univariate series.


def jobs():
    return [(ds, bb, h) for ds in BATCH
            for bb in ('ttm', 'chronos', 'timer') for h in (96, 720)]


def name(job):
    ds, bb, h = job
    return f'{ds}_p{h}_{bb}_native_seed2026_{PROTOCOL}'


def result_path(job):
    return ROOT / 'results/native' / (name(job) + '.json')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected(job):
    ds, bb, h = job
    return dict(protocol=PROTOCOL, dataset=ds, backbone=bb,
                prediction_length=h, seed=2026, context_length=512,
                batch_size=BATCH[ds], series_chunk=CHUNK,
                script_sha256=digest(__file__), downstream_training=False)


def valid(job):
    path = result_path(job)
    if not path.exists():
        return False
    d = json.loads(path.read_text())
    for k, v in expected(job).items():
        if d.get(k) != v:
            raise RuntimeError(f'Existing result mismatch: {path}: {k}')
    if not d.get('complete') or d['evaluated_windows'] != d['test_windows']:
        raise RuntimeError(f'Incomplete result: {path}')
    for v in d['test'].values():
        if not math.isfinite(v) or v < 0:
            raise RuntimeError(f'Invalid metrics: {path}')
    return True


def evaluate(job, smoke):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from data_provider import BenchmarkDataset
    from scripts.train import DATASET_CONFIGS, MODEL_PATHS, set_seed

    ds, bb, h = job
    set_seed(2026)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; CPU fallback is disabled')
    device = torch.device('cuda:0')
    cfg = DATASET_CONFIGS[ds]
    model_path = MODEL_PATHS[bb, h]
    dataset = BenchmarkDataset(
        csv_path=ROOT / 'datasets' / cfg['filename'], dataset_name=ds,
        split='test', context_length=512, prediction_length=h)
    loader = DataLoader(dataset, batch_size=BATCH[ds], shuffle=False,
                        drop_last=False, num_workers=0, pin_memory=True)
    pipeline = None
    if bb == 'ttm':
        from adapters.ttm_adapter import TTMAdapter
        adapter = TTMAdapter(model_path).to(device)
        model = adapter.model
        method = 'native_direct_point_forecast'
    elif bb == 'chronos':
        from chronos import ChronosBoltPipeline
        pipeline = ChronosBoltPipeline.from_pretrained(
            str(model_path), device_map='cpu', torch_dtype=torch.float32)
        model = pipeline.model.to(device)
        sig = inspect.signature(pipeline.predict)
        if 'limit_prediction_length' not in sig.parameters:
            raise RuntimeError(f'Unsupported Chronos predict API: {sig}')
        levels = list(model.chronos_config.quantiles)
        median_index = next(i for i, q in enumerate(levels) if abs(q - .5) < 1e-8)
        method = 'official_pipeline_long_forecast_median_quantile'
    else:
        from adapters.timer_adapter import TimerAdapter
        adapter = TimerAdapter(model_path, context_length=512).to(device)
        model = adapter.model
        if list(model.config.output_token_lens) != [96]:
            raise RuntimeError('Unexpected Timer output configuration')
        method = 'native_head_manual_autoregression_fixed_context_scaling'
    model.float().requires_grad_(False).eval()
    print('RUN:', name(job), 'method:', method, 'smoke:', smoke, flush=True)
    print('device:', torch.cuda.get_device_name(device),
          'test_windows:', len(dataset), 'batch:', BATCH[ds],
          'fixed_series_chunk:', CHUNK, flush=True)

    @torch.inference_mode()
    def predict(x):
        b, c, t = x.shape
        if t != 512 or c != cfg['n_vars'] or not torch.isfinite(x).all():
            raise RuntimeError('Unexpected input shape or nonfinite input')
        if bb == 'ttm':
            token = adapter._prepare_frequency_token(cfg['freq_token'], b, device)
            out = model(past_values=x.transpose(1, 2).contiguous(),
                        past_observed_mask=torch.ones_like(x.transpose(1, 2), dtype=torch.bool),
                        freq_token=token, return_loss=False, return_dict=True)
            y = out.prediction_outputs.transpose(1, 2).contiguous()
            # Native TTM output already restores its internal loc and scale.
        else:
            flat = x.reshape(b * c, t)
            pieces = []
            for z in flat.split(CHUNK):
                if bb == 'chronos':
                    q = pipeline.predict(z, prediction_length=h,
                                         limit_prediction_length=False)
                    if q.shape != (len(z), len(levels), h):
                        raise RuntimeError(f'Unexpected Chronos output {q.shape}')
                    ypart = q[:, median_index, :].to(device)
                else:
                    z, loc, scale = adapter._masked_revin(z, torch.isfinite(z))
                    context = F.pad(z, (adapter.left_padding, 0), value=0.)
                    generated = []
                    # Append full predicted tokens; only crop the final result.
                    # No target values are fed back, no rolling-window truncation.
                    for _ in range(math.ceil(h / 96)):
                        out = model(input_ids=context, revin=False,
                                    use_cache=False, return_dict=True,
                                    max_output_length=96)
                        block = out.logits
                        if block.shape != (len(z), 96):
                            raise RuntimeError(f'Unexpected Timer output {block.shape}')
                        generated.append(block)
                        context = torch.cat((context, block), dim=-1)
                    ypart = torch.cat(generated, dim=-1)[:, :h] * scale + loc
                pieces.append(ypart)
            y = torch.cat(pieces).reshape(b, c, h)
        if y.shape != (b, c, h) or not torch.isfinite(y).all():
            raise RuntimeError(f'Invalid prediction {y.shape}')
        return y

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    squared, absolute, count, windows = 0., 0., 0, 0
    for i, batch in enumerate(loader, 1):
        x = batch['values'].to(device)
        target = batch['target'].to(device)
        if not torch.isfinite(target).all():
            raise RuntimeError('Nonfinite target')
        y = predict(x)
        if y.shape != target.shape:
            raise RuntimeError('Prediction/target mismatch')
        error = y.double() - target.double()
        squared += error.square().sum().item()
        absolute += error.abs().sum().item()
        count += error.numel()
        windows += len(x)
        if i == 1 or i % 25 == 0 or i == len(loader):
            elapsed = time.perf_counter() - start
            eta = elapsed / i * (len(loader) - i)
            print(f'batch={i}/{len(loader)} windows={windows} '
                  f'mse={squared/count:.6f} mae={absolute/count:.6f} '
                  f'seconds={elapsed:.1f} eta_minutes={eta/60:.1f}', flush=True)
        if smoke:
            print('NATIVE_ONE_BATCH_OK (no result saved)', flush=True)
            return
    torch.cuda.synchronize()
    assert windows == len(dataset) and count > 0
    result = dict(expected(job), run_name=name(job), complete=True,
                  model_path=str(model_path.resolve()),
                  model_config_sha256=digest(model_path / 'config.json'),
                  dataset_sha256=digest(ROOT / 'datasets' / cfg['filename']),
                  test_windows=len(dataset), evaluated_windows=windows,
                  evaluated_elements=count, test={'mse': squared/count, 'mae': absolute/count},
                  method=method, dtype='float32', metric_accumulation='float64',
                  metric_scale='BenchmarkDataset standardized scale; no dataset inverse transform',
                  timer_padding=64 if bb == 'timer' else None,
                  elapsed_seconds=time.perf_counter()-start,
                  peak_memory_mb=torch.cuda.max_memory_allocated()/1024**2,
                  torch_version=torch.__version__)
    path = result_path(job)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: never replace an existing result.
    with path.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
    print('NATIVE_EVAL_OK', path, result['test'], flush=True)


def queue(smoke):
    import fcntl
    lock_path = ROOT / 'logs/native_eval.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for i, job in enumerate(jobs(), 1):
            if not smoke and valid(job):
                print(f'SKIP [{i}/18]: {name(job)}', flush=True)
                continue
            print(f'START [{i}/18]: {name(job)}', flush=True)
            cmd = [sys.executable, '-u', str(Path(__file__).resolve()),
                   '--worker', *map(str, job)]
            if smoke:
                cmd.append('--smoke')
            subprocess.run(cmd, cwd=ROOT, check=True)
            print(f'DONE [{i}/18]: {name(job)}', flush=True)
        print('ALL_NATIVE_SMOKE_FINISHED: 18' if smoke else
              'ALL_NATIVE_EVAL_FINISHED: 18', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--worker', nargs=3, metavar=('DATASET', 'BACKBONE', 'HORIZON'))
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                      PYTHONPATH=str(ROOT))
    sys.path.insert(0, str(ROOT))
    if args.gpu is not None:
        if args.gpu < 0:
            raise ValueError('GPU must be nonnegative')
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    if args.worker:
        ds, bb, h = args.worker
        job = ds, bb, int(h)
        if job not in jobs():
            raise ValueError('Invalid job')
        evaluate(job, args.smoke)
    elif args.launch:
        if args.gpu is None or args.smoke:
            raise ValueError('--launch requires --gpu and cannot use --smoke')
        folder = ROOT / 'logs'
        folder.mkdir(exist_ok=True)
        log = folder / f'native_eval_{time.strftime("%Y%m%d_%H%M%S")}_{os.getpid()}.log'
        with log.open('x') as out:
            proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve())],
                                    cwd=ROOT, env=os.environ.copy(), stdin=subprocess.DEVNULL,
                                    stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        (folder / 'native_eval.logpath').write_text(str(log))
        (folder / 'native_eval.pid').write_text(str(proc.pid))
        print('Dispatched PID:', proc.pid, 'GPU:', args.gpu)
        print('Log:', log, '(check log to confirm worker started)')
    else:
        if args.gpu is None and 'CUDA_VISIBLE_DEVICES' not in os.environ:
            raise ValueError('Specify --gpu explicitly')
        queue(args.smoke)


if __name__ == '__main__':
    main()
