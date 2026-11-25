"""Index historical evidence without importing project code or executing jobs."""
from pathlib import Path
import argparse
import ast
import csv
import hashlib
import json
import re
import shutil

REPO = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--paper', type=Path, required=True)
    parser.add_argument('--figure-root', type=Path, required=True)
    args = parser.parse_args()
    source = args.original.resolve()
    assert source != REPO
    out = REPO / 'reproduction/source_relations'
    out.mkdir(parents=True, exist_ok=True)
    imported = {}

    def keep(path, group='original'):
        base = source if group == 'original' else args.figure_root.resolve()
        rel = path.resolve().relative_to(base)
        dest = out / group / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        checksum = digest(path)
        if dest.exists() and digest(dest) != checksum:
            raise FileExistsError(f'Refusing to overwrite differing evidence: {dest}')
        if not dest.exists():
            shutil.copyfile(path, dest)
        key = dest.relative_to(REPO).as_posix()
        imported[key] = dict(path=key, original_path=str(path), sha256=checksum)
        return key

    tex = args.paper.read_text(encoding='utf-8')

    def table(label):
        start = tex.index('\\label{' + label + '}')
        start = tex.index(r'\begin{tabular}', start)
        end = tex.index(r'\end{tabular}', start)
        return tex[start:end], tex[:start].count('\n') + 1

    def nums(line):
        return [float(x) for x in re.findall(r'(?<![\w.])0\.\d+', line)]

    cells = []
    body, firstline = table('tab:main_results')
    ds = None
    for offset, line in enumerate(body.splitlines()):
        match = re.search(r'\\multirow\{4\}\{\*\}\{(\w+)\}', line)
        if match:
            ds = match[1]
        cols = line.split('&')
        if ds and len(cols) >= 4 and re.fullmatch(r'\s*\d+\s*', cols[1]):
            cells.append(dict(table='main', variant='FCD', dataset=ds,
                              horizon=int(cols[1]), paper_line=firstline + offset,
                              reported=nums('&'.join(cols[2:4]))))
    body, firstline = table('tab:component_ablation')
    for offset, line in enumerate(body.splitlines()):
        values = nums(line)
        if len(values) != 8:
            continue
        variant = line.split('&')[0].strip()
        for i, ds in enumerate(('ETTh1', 'ETTm1', 'Weather', 'ECL')):
            cells.append(dict(table='component', variant=variant, dataset=ds,
                              horizon=720, paper_line=firstline + offset,
                              reported=values[2*i:2*i+2]))

    # Associate each evaluation with the Namespace active at that line, rather
    # than attaching the first job's configuration to a concatenated log.
    finals = []
    unreadable = []
    for path in sorted(source.rglob('*.log')):
        config = None
        setting = None
        try:
            with path.open(encoding='utf-8', errors='replace') as handle:
                for lineno, line in enumerate(handle, 1):
                    if line.startswith('Namespace('):
                        try:
                            node = ast.parse(line.strip(), mode='eval').body
                            config = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
                        except (ValueError, SyntaxError, AttributeError):
                            config = None
                    if 'testing :' in line:
                        setting = line.strip().strip('><')
                    match = re.match(r'^mse:([\d.eE+-]+),\s*mae:([\d.eE+-]+)', line)
                    if match:
                        finals.append(dict(original_path=str(path), line=lineno,
                                           metrics=[float(match[1]), float(match[2])],
                                           config=config, setting=setting))
        except OSError as exc:
            unreadable.append(dict(path=str(path), error=str(exc)))

    def normalize(ds):
        return {'ecl': 'electricity'}.get(str(ds).lower(), str(ds).lower())

    for cell in cells:
        candidates = []
        for record in finals:
            cfg = record['config'] or {}
            # Missing dataset/horizon is not sufficient provenance evidence.
            if normalize(cfg.get('data')) != normalize(cell['dataset']):
                continue
            if cfg.get('pred_len') != cell['horizon']:
                continue
            if [round(x, 3) for x in record['metrics']] != cell['reported']:
                continue
            if str(cfg.get('model', '')).lower() != 'patchtst':
                continue
            if (cell['table'] == 'main' or cell['variant'] == 'Full') and cfg.get('head_type') not in ('sacq_cross_attention',):
                continue
            entry = dict(record)
            entry['path'] = keep(Path(record['original_path']))
            entry['qualification'] = 'dataset/horizon/model/rounded metrics match; source attribution unconfirmed'
            candidates.append(entry)
        cell['candidates'] = candidates
        cell['status'] = 'metric_candidates' if candidates else 'no_matching_final_log'
        # No module-switch attribution is inferred from metric coincidence.
        cell['switch_attribution_confirmed'] = False

    # Join every corruption-plot data row to raw evaluation CSV rows using all
    # available identifiers, not MSE alone. Preserve all duplicates as candidates.
    plot_path = args.figure_root / 'Figs/new/corrpdata.csv'
    plot_rows = list(csv.DictReader(plot_path.open(encoding='utf-8-sig')))
    display_path = source / 'logs/etth1_pl720_seqlen_sweep/train_corruption_adj_figures_mse.csv'
    display_rows = list(csv.DictReader(display_path.open(encoding='utf-8-sig')))
    display_copy = keep(display_path)
    raw_csv = []
    for path in sorted((source / 'logs').rglob('results_test_*.csv')):
        if not re.search(r'pl(96|720)', str(path)):
            continue
        rows = list(csv.DictReader(path.open(encoding='utf-8-sig')))
        for lineno, row in enumerate(rows, 2):
            if {'model_tag', 'corruption', 'mse'} <= row.keys():
                raw_csv.append((path, lineno, row))
    joins = []
    for plotline, row in enumerate(plot_rows, 2):
        matches = []
        comparable = []
        for path, lineno, raw in raw_csv:
            if row['corruption_type'] not in path.stem:
                continue
            if raw['model_tag'] != row['model_tag'] or raw['corruption'] != row['corruption']:
                continue
            if f"pl{row['pred_len']}_" not in raw.get('setting', ''):
                continue
            comparable.append(dict(path=keep(path), line=lineno, row=raw))
            if abs(float(raw['mse']) - float(row['mse'])) <= 1e-9:
                matches.append(dict(path=keep(path), line=lineno, row=raw))
        display_matches = [dict(path=display_copy, line=i, row=r)
                           for i, r in enumerate(display_rows, 2) if r == row]
        training_candidates = []
        settings = {r['row'].get('setting') for r in comparable}
        for record in finals:
            if not record['setting']:
                continue
            if any(s and s in record['setting'] for s in settings):
                training_candidates.append(dict(path=keep(Path(record['original_path'])),
                                                 line=record['line'], config=record['config'],
                                                 setting=record['setting']))
        joins.append(dict(plot_csv_line=plotline, plot_row=row, matches=matches,
                          same_identifiers_raw_rows=comparable,
                          display_data_matches=display_matches,
                          training_log_candidates=training_candidates,
                          status='matching_evaluation_rows' if matches else 'unmatched'))
    figure_files = [keep(plot_path, 'figure_inputs')]
    for name in ('plot_etth1_corruption_2x2.py', 'Figs/new/96mask.png', 'Figs/new/96spike.png',
                 'Figs/new/720mask.png', 'Figs/new/720spike.png'):
        path = args.figure_root / name
        if path.is_file():
            figure_files.append(keep(path, 'figure_inputs'))
    for name in ('logs/etth1_sar_vs_sacq_search/spike_train_tail_shift.csv',
                 'logs/weather_sar_vs_sacq_search/spike_train_tail_shift.csv',
                 'logs/sacq_param_sweep/sacq_param_sweep.csv',
                 'logs/sacq_param_sweep/sacq_latency_sweep.csv'):
        figure_files.append(keep(source / name))
    for name in ('logs/etth1_spike_outlier_pl720/summary.csv',
                 'logs/etth1_spike_outlier_pl720/figures/alpha_variants_cache.npz',
                 'logs/etth1_spike_outlier_pl720/figures/alpha_horizon_v2_clean_vs_spiked.png'):
        figure_files.append(keep(source / name))
    for path in sorted((source/'logs/etth1_pl720_seqlen_sweep').rglob('experiment_meta*.json')):
        figure_files.append(keep(path))
    figure_relations = [
        dict(paper_label='fig:corruption_etth1', paper_line=722,
             plot='external figure_inputs/plot_etth1_corruption_2x2.py',
             evaluation_entry='scripts/experiments/eval_etth1_train_corruption_sl512.py',
             joins='corruption_sources.json', historical_injection_cli_confirmed=False),
        dict(paper_label='fig:sar_tail_shift', paper_line=742,
             plot='scripts/experiments/plot_tail_shift_noall_vertical_composite.py',
             data=['logs/etth1_sar_vs_sacq_search/spike_train_tail_shift.csv',
                   'logs/weather_sar_vs_sacq_search/spike_train_tail_shift.csv'],
             etth1=dict(pair='spike_train', shift_min=10, swap_near_far=False, swap_far_models=False),
             weather=dict(pair='spike_train_p5_pl720', shift_min=0, swap_near_far=True, swap_far_models=True)),
        dict(paper_label='fig:alpha_horizon_spiked', paper_line=761,
             plot='scripts/experiments/plot_spike_outlier_alpha_variants.py',
             function='fig_clean_vs_spiked',
             helper='scripts/experiments/plot_spike_outlier_alpha_curves.py',
             data=['logs/etth1_spike_outlier_pl720/summary.csv',
                   'logs/etth1_spike_outlier_pl720/figures/alpha_variants_cache.npz'],
             code_observations=dict(training_pcts=[0,1,3,5], forecast_steps=720,
                                    channel='OT', injected_rng_seed='2021 + test batch index'),
             historical_cli_confirmed=False,
             note='Cache retained without executing it; max_batches/noise settings are not inferred from argparse defaults.'),
        dict(paper_label='fig:efficiency_dmodel', paper_line=802,
             plot='scripts/experiments/plot_sacq_params_latency_combined.py',
             profiling='scripts/experiments/plot_sacq_latency_vs_dmodel.py',
             data=['logs/sacq_param_sweep/sacq_param_sweep.csv',
                   'logs/sacq_param_sweep/sacq_latency_sweep.csv'],
             recorded_batch_size=64, historical_warmup=None, historical_repeats=None)
    ]

    summary = dict(paper=str(args.paper), paper_sha256=digest(args.paper),
                   final_evaluation_records_scanned=len(finals), unreadable_logs=unreadable,
                   table_cells=len(cells), main_cells=sum(c['table']=='main' for c in cells),
                   component_cells=sum(c['table']=='component' for c in cells),
                   table_cells_with_candidates=sum(bool(c['candidates']) for c in cells),
                   corruption_plot_rows=len(joins),
                   corruption_rows_matched=sum(bool(j['matches']) for j in joins),
                   corruption_display_rows_matched=sum(bool(j['display_data_matches']) for j in joins),
                   corruption_rows_with_training_log_candidates=sum(bool(j['training_log_candidates']) for j in joins),
                   verified_historical_latency_warmup=None, verified_historical_latency_repeats=None)
    for name, value in [('table_sources.json', cells), ('corruption_sources.json', joins),
                        ('summary.json', summary), ('copied_evidence.json', list(imported.values())),
                        ('figure_files.json', figure_files), ('figure_relations.json', figure_relations)]:
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

    lines = ['# 监督实验与图表来源索引', '',
             '只建立原记录之间的关系，不修改论文、数值、模型或训练配置。', '',
             '## 监督主表与组件表', '',
             '下表“候选”要求数据集、预测长度、PatchTST 模型及两项指标三位小数同时匹配；主表与 Full 还要求预测头类型匹配。',
             '候选不等于历史来源已确认，多个候选全部保留；组件开关不能仅由数值反推。缺少候选不表示实验一定未做。', '',
             '|表|变体|数据集|长度|论文行号|MSE / MAE|候选日志数|',
             '|---|---|---|---:|---:|---|---:|']
    for c in cells:
        lines.append(f"|{c['table']}|{c['variant']}|{c['dataset']}|{c['horizon']}|{c['paper_line']}|{c['reported'][0]:.3f} / {c['reported'][1]:.3f}|{len(c['candidates'])}|")
    lines += ['', '完整候选路径、最终评估所在行、原 Namespace、setting 见 `reproduction/source_relations/table_sources.json`。',
              '本索引只对应本方法列；主表中其他方法的来源应按引用文献记录，不自动归入本地 SACQ 实验。', '',
              '## 污染图与效率图', '',
              f"- ETTh1 mask/spike：绘图用 corrpdata.csv 的 {len(joins)} 行中，{sum(bool(j['matches']) for j in joins)} 行能按 horizon、model_tag、corruption 和 MSE 精度匹配 results_test CSV。逐行关系见 corruption_sources.json，setting 与样本数保留在原始行。",
              f"- 图表数据层：corrpdata.csv 有 {sum(bool(j['display_data_matches']) for j in joins)}/{len(joins)} 行与原项目 train_corruption_adj_figures_mse.csv 完全一致。另有 {sum(not bool(j['matches']) for j in joins)} 行尚不能与 results_test 原始评估数值精确对应；不能把展示数据层的匹配写成全部原始评估已经核实。差异逐行列在下方，未修改任何值。",
              '- 四联图的原组合入口为 plot_etth1_corruption_2x2.py，输入是四张面板 PNG；corrpdata.csv 与评估 CSV 的匹配不单独证明原面板绘制命令。面板及组合脚本已保留。',
              '- tail-shift：scripts/experiments/plot_tail_shift_noall_vertical_composite.py → ETTh1/Weather 各自 logs/*_sar_vs_sacq_search/spike_train_tail_shift.csv。ETTh1 筛选 pair=spike_train、shift>=10；Weather 筛选 pair=spike_train_p5_pl720、shift>=0，保留原脚本的展示映射。',
              '- 门控 α 污染图（fig:alpha_horizon_spiked）：plot_spike_outlier_alpha_variants.py 的 fig_clean_vs_spiked → etth1_spike_outlier_pl720/summary.csv 与 figures/alpha_variants_cache.npz。缓存、原 PNG 已保留；辅助 collect_alpha_curves 使用 OT 通道，输入注入随机种子为 2021+batch序号。历史 CLI 中 max_batches、noise_mult 等尚未确认，不按当前默认值补写。',
              '- 参数/延迟：scripts/experiments/plot_sacq_params_latency_combined.py → logs/sacq_param_sweep/{sacq_param_sweep,sacq_latency_sweep}.csv；n_heads=4，d_model=8/16/32/64/96/128/192/256，延迟 CSV batch=64。',
              '- 计时实现见 scripts/experiments/plot_sacq_latency_vs_dmodel.py：CUDA 同步、perf_counter、排序后取中间位置值；head 开销按完整与 encoder+coarse 计时之差估计。原图实际 warm-up/repeats 没有找到记录，当前脚本默认值不能补为历史参数。',
              '- 以上绘图入口指整理库中继承的原脚本；V5 的图例更新入口另为 paper_v5/figure_updates/regenerate_fcd_figures.py，仅是图例与组合关系，未在本轮重新出图。', '',
              '## 证据保留与使用', '',
              '复制的原始日志、CSV、面板位于 reproduction/source_relations/；copied_evidence.json 记录原路径与 SHA256。',
              'figure_relations.json 逐图列出 V5 行号、绘图/评估入口、输入记录及已确认的筛选关系。',
              'table_sources.json 内配置是对应最终评估之前实际打印的 Namespace，不使用后来代码默认值。没有 Namespace 的日志不作配置匹配。',
              '污染注入的 seed/幅度须继续以对应 setting 的运行配置或 metadata 为准；本索引没有把绘图脚本默认值视作实际训练配置。',
              '这些记录可用于后续人工确认来源；本工具不选择最佳测试分数、不启动评估或补跑。', '']
    lines += ['## 展示数据与原始评估未精确对应的点', '',
              '|面板|模型|扰动|展示 MSE|相同标识的原始 MSE|', '|---|---|---|---:|---|']
    for j in joins:
        if j['matches']:
            continue
        row = j['plot_row']
        vals = sorted({r['row']['mse'] for r in j['same_identifiers_raw_rows']})
        lines.append(f"|{row['figure']}|{row['model_tag']}|{row['corruption']}|{row['mse']}|{', '.join(vals) or '未找到'}|")
    (REPO / 'docs/SUPERVISED_SOURCE_MAPPING.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('SOURCE_RELATIONS_BUILT_NO_TRAINING')


if __name__ == '__main__':
    main()
