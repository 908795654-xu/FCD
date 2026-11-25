"""Synthetic CPU head check only: no datasets, weights, GPU, or optimizer step."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile',choices=['v1','v2'],required=True)
    args=ap.parse_args()
    project=ROOT/'experiments/frozen'/'SACQ_TSFM'
    sys.path.insert(0,str(project))
    import torch
    from sacq_core import SACQCrossAttentionHead
    if args.profile=='v2':
        from sacq_core.pe_initializer import positional_encoding
    torch.set_num_threads(2)
    reports=[]
    for horizon in (96,720):
        torch.manual_seed(2026)
        head=SACQCrossAttentionHead(n_vars=3,nf=32*6,d_model=32,target_window=horizon,
            patch_len=16,cross_attn_layers=2,n_heads=4,head_dropout=0,
            individual=False,sacq_alpha_mode='learnable',sacq_alpha_value=.5,
            sacq_autoregressive=False,sacq_pe_mode='default',hist_patch_num=6,
            dsacq_enable=False,dsacq_mem_gate_enable=False,dsacq_drift_gate_enable=False)
        if args.profile=='v2':
            with torch.random.fork_rng(devices=[]):
                head.query_pos_emb=positional_encoding('zeros',True,6+horizon,32)
        memory=torch.randn(2,3,32,6);target=torch.randn(2,3,horizon)
        output=head(memory)[0]
        assert tuple(output.shape)==tuple(target.shape)
        assert torch.isfinite(output).all()
        loss=torch.nn.functional.mse_loss(output,target);loss.backward()
        grads=[p.grad for p in head.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum().item()>0 for g in grads)
        if args.profile=='v2':
            assert head.query_pos_emb.grad is not None
            assert head.query_pos_emb.grad[6:6+head.out_patch_num].abs().sum().item()>0
        reports.append({'horizon':horizon,'shape':list(output.shape),'finite_output':True,
                        'finite_nonzero_gradients':True,'additional_pe':args.profile=='v2'})
    print(json.dumps({'profile':args.profile,'synthetic_cpu_only':True,'optimizer_steps':0,
                      'checks':reports},indent=2))
    print('HEAD_SMOKE_OK')

if __name__=='__main__':main()
