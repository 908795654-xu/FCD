"""Readiness checks only: no epoch training, no validation/test-set scoring."""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def static_checks():
    manifest = json.loads((ROOT/'installation_manifest.json').read_text())
    for rel, digest in manifest['installed_hashes'].items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == digest, rel
    original = ast.parse((ROOT/'reference/original/layers/PatchTST_backbone.py').read_text())
    current = ast.parse((ROOT/'sacq_core/head.py').read_text())
    for name in ('Flatten_Head', 'SACQDecoderStack', 'SACQCrossAttentionHead'):
        left = next(n for n in original.body if isinstance(n, ast.ClassDef) and n.name == name)
        right = next(n for n in current.body if isinstance(n, ast.ClassDef) and n.name == name)
        assert ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False), name
    print('SOURCE_HASHES_AND_CORE_AST_OK', flush=True)


def imports():
    import torch
    import models
    import adapters
    import data_provider
    import sacq_core
    for mod in (models, adapters, data_provider, sacq_core):
        assert Path(mod.__file__).resolve().is_relative_to(ROOT.resolve()), mod.__file__
    return torch


def check_head(head, memory, target, torch):
    # No dropout; compare equal core weights and zero PE with the PE-off path.
    head.eval()
    pe = head.query_pos_emb
    assert pe is not None and pe.requires_grad
    b, c, d, p = memory.shape
    h = head.target_window
    o = head.out_patch_num
    assert tuple(pe.shape) == (p+h, d)
    assert head.sacq_pe_mode == 'default'
    assert not head.dsacq_enable and not head.sacq_autoregressive
    assert not head.decoder.force_causal
    assert head.query_pos_emb is dict(head.named_parameters())['query_pos_emb']
    base = copy.deepcopy(head)
    base.query_pos_emb = None
    saved = pe.detach().clone()
    with torch.no_grad():
        pe.zero_()
        zero_pred = head(memory)[0]
        off_pred = base(memory)[0]
        torch.testing.assert_close(zero_pred, off_pred, rtol=1e-5, atol=1e-6)
        pe.copy_(saved)
    del base
    captured = []
    hook = head.decoder.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
    with torch.no_grad():
        pred, aux = head(memory)
        coarse = head.query_flatten_head(memory).reshape(b, c, o, head.patch_len)
        content = head.query_patch_proj(coarse).reshape(b*c, o, d)
        expected = content + pe[p:p+o].unsqueeze(0)
        assert len(captured) == 1
        torch.testing.assert_close(captured[0], expected)
    hook.remove()
    assert pred.shape == target.shape and torch.isfinite(pred).all()
    head.train()
    optimizer = torch.optim.AdamW([x for x in head.parameters() if x.requires_grad], lr=1e-4, weight_decay=0)
    ids = {id(x) for group in optimizer.param_groups for x in group['params']}
    assert ids == {id(x) for x in head.parameters() if x.requires_grad}
    assert id(pe) in ids
    optimizer.zero_grad(set_to_none=True)
    prediction = head(memory)[0]
    loss = torch.nn.functional.mse_loss(prediction, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert pe.grad is not None and torch.isfinite(pe.grad).all()
    assert pe.grad[p:p+o].abs().sum().item() > 0
    assert pe.grad[:p].abs().sum().item() == 0
    assert pe.grad[p+o:].abs().sum().item() == 0
    for parameter in head.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
    optimizer.step()
    assert not torch.equal(pe.detach()[p:p+o], saved[p:p+o])
    return dict(output_shape=list(pred.shape), pe_shape=list(pe.shape), active_pe_rows=o,
                pe_gradient_nonzero=True, pe_in_optimizer=True, pe_step_changed=True,
                zero_pe_matches_pe_off=True, query_addition_verified=True)


def head_only():
    torch = imports()
    from models.factory_pe import build_head
    for h in (96, 720):
        torch.manual_seed(2026)
        head = build_head('sacq', 2, 128, 9, h)
        memory = torch.randn(2, 2, 128, 9)
        target = torch.randn(2, 2, h)
        print('HEAD_CHECK', h, check_head(head, memory, target, torch), flush=True)
    print('V2_HEAD_ONLY_OK', flush=True)


def gpu_worker(backbone, horizon):
    torch = imports()
    from torch.utils.data import DataLoader
    from data_provider import BenchmarkDataset
    from models.factory_pe import build_forecaster
    from settings import MODEL_PATHS
    assert torch.cuda.is_available(), 'CUDA unavailable'
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    ds = BenchmarkDataset(csv_path=ROOT/'datasets/ETTh1.csv', dataset_name='etth1',
                          split='train', context_length=512, prediction_length=horizon)
    batch = next(iter(DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)))
    x, y = batch['values'].cuda(), batch['target'].cuda()
    assert torch.isfinite(x).all() and torch.isfinite(y).all()
    model, metadata = build_forecaster(backbone, 'sacq', MODEL_PATHS[(backbone, horizon)],
                                       512, horizon, 7)
    model.cuda().train()
    assert not any(m.training for m in model.adapter.modules())
    assert not any(p.requires_grad for p in model.adapter.parameters())
    frozen = {k: v.detach().cpu().clone() for k, v in model.adapter.state_dict().items()}
    kwargs = {'freq_token': 7} if backbone == 'ttm' else {}
    # Capture real backbone features, retaining each adapter's normalization convention.
    model.eval()
    with torch.no_grad():
        features = model.adapter(x, **kwargs)
        output = model(x, adapter_kwargs=kwargs)
        assert output.prediction.shape == y.shape and torch.isfinite(output.prediction).all()
    checks = check_head(model.head, features.memory.detach(), torch.zeros_like(y), torch)
    model.train()
    # Exercise actual wrapper denormalization and backward with real training targets.
    optimizer = torch.optim.AdamW([p for p in model.head.parameters() if p.requires_grad], lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    prediction = model(x, adapter_kwargs=kwargs).prediction
    loss = torch.nn.functional.mse_loss(prediction, y)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.head.query_pos_emb.grad is not None
    assert torch.isfinite(model.head.query_pos_emb.grad).all()
    assert model.head.query_pos_emb.grad.abs().sum().item() > 0
    optimizer.step()
    assert all(p.grad is None for p in model.adapter.parameters())
    assert not any(m.training for m in model.adapter.modules())
    assert all(torch.equal(frozen[k], v.detach().cpu()) for k, v in model.adapter.state_dict().items())
    # Linear baseline interface uses the same real features and scale restoration.
    from models.factory_pe import build_head
    linear = build_head('linear', 7, metadata['hidden_dim'], metadata['patch_count'], horizon).cuda()
    with torch.no_grad():
        linear_prediction = features.denormalize(linear(features.memory))
        assert linear_prediction.shape == y.shape and torch.isfinite(linear_prediction).all()
    record = dict(backbone=backbone, horizon=horizon, metadata=metadata, checks=checks,
                  backbone_unchanged=True, adapter_gradients_absent=True, linear_interface_ok=True,
                  formal_training=False, test_evaluated=False, split='train', batch_size=2)
    out = ROOT/'logs/migration_checks'
    out.mkdir(parents=True, exist_ok=True)
    (out/f'{backbone}_p{horizon}.json').write_text(json.dumps(record, indent=2))
    print('V2_BACKBONE_CHECK_OK', backbone, horizon, json.dumps(checks), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--head-only', action='store_true')
    parser.add_argument('--all-backbones', action='store_true')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--worker', nargs=2)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2',
                      MKL_NUM_THREADS='2', PYTHONPATH=str(ROOT))
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    static_checks()
    if args.worker:
        gpu_worker(args.worker[0], int(args.worker[1]))
    elif args.head_only:
        head_only()
    elif args.all_backbones:
        if args.gpu is None:
            parser.error('--all-backbones requires an explicit physical --gpu')
        for b in ('ttm', 'chronos', 'timer'):
            for h in (96, 720):
                subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()),
                                '--worker', b, str(h)], cwd=ROOT, check=True)
        print('ALL_6_V2_BACKBONE_CHECKS_OK', flush=True)
    else:
        print('STATIC_CHECKS_ONLY_OK; choose --head-only or --all-backbones --gpu N')
