"""Map completed frozen-study records to the paper, without fitting/selecting by test."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import re
import statistics as stats

ROOT=Path(__file__).resolve().parents[1]
V1=ROOT/'experiments/frozen/SACQ_TSFM'
V2=V1  # merged confirmation/PE records

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rel(p):return str(p.relative_to(ROOT)).replace('\\','/')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--paper',type=Path,help='Optional current manuscript; compare only, never edit it.')
    args=ap.parse_args()
    ref=read(V2/'results/confirmation_fixed_v1/all_results.json')
    mlp=read(V2/'results/mlp_control_v1/all_confirmation_results.json')
    weather_path=V1/'results/weather_head_tuning_v1/confirmations_p96_720.json'
    weather=read(weather_path)
    selection=read(V1/'results/weather_head_tuning_v1/selection_p96_720.json')
    choices={(r['backbone'],r['horizon'],r['head']):r['candidate'] for r in selection}
    electricity_mlp_path=V1/'results/mlp_control_frozen_v1/electricity/all_confirmation_results.json'
    electricity_mlp=read(electricity_mlp_path)
    groups=[];widths=[]
    for ds in ('etth1','weather','electricity'):
      for bb in ('ttm','chronos','timer'):
       for h in (96,720):
        for head in ('native','linear','mlp','sacq'):
         records=[]
         for seed in ((2026,) if head=='native' else (2026,2027,2028)):
          pointer=None
          if head=='native':
            p=V1/f'results/native/{ds}_p{h}_{bb}_native_seed2026_native_v1.json';r=read(p)
            entry=V1/'scripts/native_eval.py'
          elif ds=='etth1':
            p=V2/('results/mlp_control_v1/all_confirmation_results.json' if head=='mlp'
                   else 'results/confirmation_fixed_v1/all_results.json')
            source=mlp if head=='mlp' else ref
            variant='sacq_pe_on' if head=='sacq' else head
            matching=[(i,r) for i,r in enumerate(source) if r['backbone']==bb and r['horizon']==h and r['seed']==seed
                      and (head=='mlp' or r['variant']==variant)]
            assert len(matching)==1,(ds,bb,h,head,seed)
            pointer,r=matching[0];entry=V2/'scripts'/('run_mlp_control.py' if head=='mlp' else 'run_v2_confirmation.py')
          elif ds=='weather':
            p=weather_path
            matching=[(i,r) for i,r in enumerate(weather) if r['backbone']==bb and r['horizon']==h and r['seed']==seed
                       and r['config']['head']==head]
            assert len(matching)==1,(ds,bb,h,head,seed)
            pointer,r=matching[0]
            assert r['config']['candidate']==choices[(bb,h,head)]
            entry=V1/'scripts/run_weather_head_tuning.py'
          elif head=='mlp':
            p=electricity_mlp_path
            matching=[(i,r) for i,r in enumerate(electricity_mlp) if r['backbone']==bb and r['horizon']==h and r['seed']==seed]
            assert len(matching)==1
            pointer,r=matching[0];entry=V1/'scripts/run_mlp_weather_electricity.py'
          else:
            p=V1/f'results/formal/{ds}_p{h}_{bb}_{head}_seed{seed}_frozen_v1.json'
            r=read(p);entry=V1/'scripts/train.py'
          assert entry.is_file() and p.is_file()
          assert r.get('complete',True) and not r.get('smoke',False)
          assert all(math.isfinite(float(r['test'][m])) for m in ('mse','mae'))
          cfg=r.get('config') or {k:r[k] for k in ('dataset','backbone','head','seed','batch_size','max_epochs',
                'patience','learning_rate','weight_decay','loss','context_length','prediction_length') if k in r}
          record={'seed':seed,'source':rel(p),'json_pointer':None if pointer is None else f'/{pointer}',
                  'source_sha256':digest(p),'entrypoint':rel(entry),'config':cfg,
                  'metadata':r.get('metadata',{}),'test':r['test']}
          records.append(record)
          if head=='mlp':
            meta=r['metadata'];w=meta['mlp_hidden_width'];n=meta['trainable_parameters'];target=meta['target_sacq_parameters']
            assert n==w*(meta['hidden_dim']*meta['patch_count']+h+1)+h
            assert abs(100*(n-target)/target-meta['parameter_gap_pct'])<1e-9
         g={'id':f'{ds}/{bb}/{h}/{head}','dataset':ds,'backbone':bb,'horizon':h,'head':head,
            'seed_count':len(records),'records':records,
            'summary':{m:{'mean':stats.mean(r['test'][m] for r in records),
               'sample_std':stats.stdev(r['test'][m] for r in records) if len(records)>1 else None} for m in ('mse','mae')}}
         groups.append(g)
         if head=='mlp':
            meta=records[0]['metadata']
            assert all(r['metadata']['mlp_hidden_width']==meta['mlp_hidden_width'] for r in records)
            widths.append(dict(dataset=ds,backbone=bb,horizon=h,hidden_width=meta['mlp_hidden_width'],
                input_dim=meta['hidden_dim']*meta['patch_count'],mlp_parameters=meta['trainable_parameters'],
                sacq_target=meta['target_sacq_parameters'],parameter_gap_pct=meta['parameter_gap_pct']))
    assert len(groups)==72 and sum(len(g['records']) for g in groups)==180
    output=ROOT/'configs/paper';output.mkdir(parents=True,exist_ok=True)
    content={'scope':'Frozen-pretrained table; recorded confirmations selected by validation, not test ranking.',
             'native_seed_count':1,'trained_seed_count':3,'groups':groups}
    (output/'frozen_table.json').write_text(json.dumps(content,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (ROOT/'paper_results/mlp_parameter_matching.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(widths[0]));writer.writeheader();writer.writerows(widths)
    rows=['# 冻结预训练主表的逐项来源','',
          '由已完成确认记录建立，不按测试集重新选参。Native 每项一份评估，新头每项三个种子。',
          '完整配置、精确记录位置、JSON pointer 和哈希见 `configs/paper/frozen_table.json`。',
          '', '|数据集|骨干|预测长度|Native MSE/MAE|Linear MSE/MAE|MLP MSE/MAE|FCD MSE/MAE|',
          '|---|---|---|---|---|---|---|']
    for i in range(0,len(groups),4):
        block=groups[i:i+4];g=block[0]
        vals=[f"{v['summary']['mse']['mean']:.4f} / {v['summary']['mae']['mean']:.4f}" for v in block]
        rows.append('|'+ '|'.join([g['dataset'],g['backbone'],str(g['horizon']),*vals])+'|')
    mismatches=[]
    if args.paper:
        s=args.paper.read_text(encoding='utf-8');a=s.index(r'\label{tab:frozen_transfer}');b=s.index(r'\end{table}',a)
        paper_rows=[]
        for line in s[a:b].splitlines():
            if '&' not in line:continue
            values=re.findall(r'(?<!\d)0\.\d+',line)
            if len(values)==8:paper_rows.append([float(x) for x in values])
        assert len(paper_rows)==18
        for row_no,values in enumerate(paper_rows):
            expected=[groups[row_no*4+j]['summary'][m]['mean'] for j in range(4) for m in ('mse','mae')]
            for col,(actual,expected) in enumerate(zip(values,expected)):
                if f'{actual:.4f}'!=f'{expected:.4f}':mismatches.append({'row':row_no+1,'metric_column':col+1,'paper':actual,'records':expected})
        rows+=['',f'与指定 V5 表格核对：18 行、144 个指标单元格；不匹配项：{len(mismatches)}。',
               '这里只比较显示精度及记录来源，不代表重新训练或重新评估 checkpoint。']
    (ROOT/'docs/FROZEN_TABLE_MAPPING.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    report={'groups':len(groups),'source_records':180,'mlp_configurations':len(widths),
            'paper_compared':bool(args.paper),'paper_mismatches':mismatches}
    (ROOT/'paper_results/frozen_mapping_check.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    if mismatches:raise SystemExit('Mapping generated; discrepancies recorded without editing the manuscript.')
    print('FROZEN_PAPER_MAPPING_OK')

if __name__=='__main__':main()
