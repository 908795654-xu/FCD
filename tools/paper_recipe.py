"""Display one recorded paper recipe and a command plan; never execute a job."""
from pathlib import Path
import argparse
import hashlib
import json
import shlex

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--id',help='dataset/backbone/horizon/head; e.g. weather/ttm/96/sacq')
    ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--gpu',type=int,default=0,help='Physical GPU for displayed command only')
    ap.add_argument('--list',action='store_true')
    args=ap.parse_args()
    groups=json.loads((ROOT/'configs/paper/frozen_table.json').read_text(encoding='utf-8'))['groups']
    if args.list:
        for g in groups:print(g['id'])
        return
    if not args.id:ap.error('Specify --list or --id')
    if args.gpu<0:ap.error('--gpu must be nonnegative')
    found=[g for g in groups if g['id']==args.id]
    if len(found)!=1:ap.error('Unknown recipe; use --list')
    g=found[0];items=[r for r in g['records'] if r['seed']==args.seed]
    if len(items)!=1:ap.error('Seed is not recorded for this recipe; native has only seed2026')
    r=items[0];source=ROOT/r['source']
    assert hashlib.sha256(source.read_bytes()).hexdigest()==r['source_sha256'],'Source record changed'
    entry=ROOT/r['entrypoint'];project=entry.parents[1]
    command=['python','-u',entry.relative_to(project).as_posix()]
    ds,bb,h,head=g['dataset'],g['backbone'],g['horizon'],g['head'];cfg=r['config']
    report='results/reproduction/'+args.id.replace('/','_')+f'_seed{args.seed}.json'
    if head=='native':
        command+=['--gpu',str(args.gpu),'--worker',ds,bb,str(h)]
        note='Native entry writes its standard results/native path; use an isolated working copy for a new evaluation.'
    elif entry.name=='train.py':
        command+=['--dataset',ds,'--backbone',bb,'--head',head,'--prediction-length',str(h),
                  '--seed',str(args.seed),'--device','cuda:0','--training-protocol','frozen_v1']
        for key,flag in [('batch_size','--batch-size'),('max_epochs','--epochs'),('patience','--patience'),
                         ('learning_rate','--learning-rate'),('weight_decay','--weight-decay')]:
            command.extend([flag,str(cfg[key])])
        note='Train entry uses its standard results/formal path; do not overwrite reference records during a new run.'
    else:
        variant=cfg.get('variant')
        assert variant,'Recorded worker variant is required'
        if entry.name=='run_mlp_weather_electricity.py':command+=['--dataset',ds]
        command+=['--gpu',str(args.gpu),'--worker',bb,str(h),variant,str(args.seed),'--report',report]
        note='Worker uses the recorded final variant, not the whole tuning queue. Existing resource/fingerprint checks remain active.'
    plan={'id':g['id'],'seed':args.seed,'record':r['source'],'json_pointer':r['json_pointer'],
          'configuration_as_recorded':cfg,'head_metadata':r['metadata'],'recorded_test':r['test'],
          'working_directory':str(project),'command_plan':shlex.join(command),
          'required_environment':{'CUDA_VISIBLE_DEVICES':str(args.gpu),'PYTHONPATH':str(project),
                                  'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'},
          'report_parent_to_create':None if head=='native' or entry.name=='train.py' else str(project/'results/reproduction'),
          'executed':False,'note':note,
          'prerequisites':'Use sacq_tsfm; configure existing data/model assets; review paths and fingerprint checks. No GPU rerun was performed by this tool.'}
    print(json.dumps(plan,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
