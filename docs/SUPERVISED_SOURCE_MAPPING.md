# 监督实验与图表来源索引

只建立原记录之间的关系，不修改论文、数值、模型或训练配置。

## 监督主表与组件表

下表“候选”要求数据集、预测长度、PatchTST 模型及两项指标三位小数同时匹配；主表与 Full 还要求预测头类型匹配。
候选不等于历史来源已确认，多个候选全部保留；组件开关不能仅由数值反推。缺少候选不表示实验一定未做。

|表|变体|数据集|长度|论文行号|MSE / MAE|候选日志数|
|---|---|---|---:|---:|---|---:|
|main|FCD|ETTh1|96|333|0.361 / 0.390|0|
|main|FCD|ETTh1|192|334|0.398 / 0.416|0|
|main|FCD|ETTh1|336|335|0.417 / 0.427|0|
|main|FCD|ETTh1|720|336|0.429 / 0.452|7|
|main|FCD|ETTh2|96|338|0.269 / 0.334|1|
|main|FCD|ETTh2|192|339|0.333 / 0.377|1|
|main|FCD|ETTh2|336|340|0.324 / 0.380|1|
|main|FCD|ETTh2|720|341|0.376 / 0.419|3|
|main|FCD|ETTm1|96|343|0.286 / 0.333|2|
|main|FCD|ETTm1|192|344|0.328 / 0.361|0|
|main|FCD|ETTm1|336|345|0.356 / 0.379|1|
|main|FCD|ETTm1|720|346|0.413 / 0.412|1|
|main|FCD|ETTm2|96|348|0.159 / 0.244|1|
|main|FCD|ETTm2|192|349|0.216 / 0.283|1|
|main|FCD|ETTm2|336|350|0.271 / 0.320|1|
|main|FCD|ETTm2|720|351|0.358 / 0.375|1|
|main|FCD|Weather|96|353|0.146 / 0.185|0|
|main|FCD|Weather|192|354|0.192 / 0.230|0|
|main|FCD|Weather|336|355|0.243 / 0.271|1|
|main|FCD|Weather|720|356|0.310 / 0.326|1|
|main|FCD|ECL|96|358|0.128 / 0.218|0|
|main|FCD|ECL|192|359|0.144 / 0.234|0|
|main|FCD|ECL|336|360|0.162 / 0.251|1|
|main|FCD|ECL|720|361|0.198 / 0.282|1|
|main|FCD|Traffic|96|363|0.356 / 0.245|0|
|main|FCD|Traffic|192|364|0.374 / 0.251|0|
|main|FCD|Traffic|336|365|0.387 / 0.255|0|
|main|FCD|Traffic|720|366|0.427 / 0.276|0|
|component|Full|ETTh1|720|524|0.429 / 0.452|7|
|component|Full|ETTm1|720|524|0.413 / 0.412|1|
|component|Full|Weather|720|524|0.310 / 0.326|1|
|component|Full|ECL|720|524|0.198 / 0.282|1|
|component|$-$CA|ETTh1|720|525|0.441 / 0.462|1|
|component|$-$CA|ETTm1|720|525|0.425 / 0.421|0|
|component|$-$CA|Weather|720|525|0.322 / 0.336|0|
|component|$-$CA|ECL|720|525|0.206 / 0.290|0|
|component|$-$G|ETTh1|720|526|0.436 / 0.458|0|
|component|$-$G|ETTm1|720|526|0.420 / 0.418|0|
|component|$-$G|Weather|720|526|0.317 / 0.332|0|
|component|$-$G|ECL|720|526|0.202 / 0.286|0|
|component|$-$Fus|ETTh1|720|527|0.433 / 0.455|0|
|component|$-$Fus|ETTm1|720|527|0.417 / 0.415|0|
|component|$-$Fus|Weather|720|527|0.314 / 0.329|0|
|component|$-$Fus|ECL|720|527|0.200 / 0.284|0|
|component|Coarse|ETTh1|720|528|0.447 / 0.466|0|
|component|Coarse|ETTm1|720|528|0.430 / 0.426|0|
|component|Coarse|Weather|720|528|0.325 / 0.339|0|
|component|Coarse|ECL|720|528|0.208 / 0.293|0|

完整候选路径、最终评估所在行、原 Namespace、setting 见 `reproduction/source_relations/table_sources.json`。
本索引只对应本方法列；主表中其他方法的来源应按引用文献记录，不自动归入本地 SACQ 实验。

## 污染图与效率图

- ETTh1 mask/spike：绘图用 corrpdata.csv 的 104 行中，92 行能按 horizon、model_tag、corruption 和 MSE 精度匹配 results_test CSV。逐行关系见 corruption_sources.json，setting 与样本数保留在原始行。
- 图表数据层：corrpdata.csv 有 104/104 行与原项目 train_corruption_adj_figures_mse.csv 完全一致。另有 12 行尚不能与 results_test 原始评估数值精确对应；不能把展示数据层的匹配写成全部原始评估已经核实。差异逐行列在下方，未修改任何值。
- 四联图的原组合入口为 plot_etth1_corruption_2x2.py，输入是四张面板 PNG；corrpdata.csv 与评估 CSV 的匹配不单独证明原面板绘制命令。面板及组合脚本已保留。
- tail-shift：scripts/experiments/plot_tail_shift_noall_vertical_composite.py → ETTh1/Weather 各自 logs/*_sar_vs_sacq_search/spike_train_tail_shift.csv。ETTh1 筛选 pair=spike_train、shift>=10；Weather 筛选 pair=spike_train_p5_pl720、shift>=0，保留原脚本的展示映射。
- 门控 α 污染图（fig:alpha_horizon_spiked）：plot_spike_outlier_alpha_variants.py 的 fig_clean_vs_spiked → etth1_spike_outlier_pl720/summary.csv 与 figures/alpha_variants_cache.npz。缓存、原 PNG 已保留；辅助 collect_alpha_curves 使用 OT 通道，输入注入随机种子为 2021+batch序号。历史 CLI 中 max_batches、noise_mult 等尚未确认，不按当前默认值补写。
- 参数/延迟：scripts/experiments/plot_sacq_params_latency_combined.py → logs/sacq_param_sweep/{sacq_param_sweep,sacq_latency_sweep}.csv；n_heads=4，d_model=8/16/32/64/96/128/192/256，延迟 CSV batch=64。
- 计时实现见 scripts/experiments/plot_sacq_latency_vs_dmodel.py：CUDA 同步、perf_counter、排序后取中间位置值；head 开销按完整与 encoder+coarse 计时之差估计。原图实际 warm-up/repeats 没有找到记录，当前脚本默认值不能补为历史参数。
- 以上绘图入口指整理库中继承的原脚本；V5 的图例更新入口另为 paper_v5/figure_updates/regenerate_fcd_figures.py，仅是图例与组合关系，未在本轮重新出图。

## 证据保留与使用

复制的原始日志、CSV、面板位于 reproduction/source_relations/；copied_evidence.json 记录原路径与 SHA256。
figure_relations.json 逐图列出 V5 行号、绘图/评估入口、输入记录及已确认的筛选关系。
table_sources.json 内配置是对应最终评估之前实际打印的 Namespace，不使用后来代码默认值。没有 Namespace 的日志不作配置匹配。
污染注入的 seed/幅度须继续以对应 setting 的运行配置或 metadata 为准；本索引没有把绘图脚本默认值视作实际训练配置。
这些记录可用于后续人工确认来源；本工具不选择最佳测试分数、不启动评估或补跑。

## 展示数据与原始评估未精确对应的点

|面板|模型|扰动|展示 MSE|相同标识的原始 MSE|
|---|---|---|---:|---|
|pl96_mask|sar_slc|mask_10|0.3653883443|0.3668883442878723|
|pl96_mask|sar_slc|mask_15|0.3714835166|0.3765835165977478|
|pl96_mask|sar_slc|mask_20|0.3826043565|0.3894043564796448|
|pl96_mask|sar_slc|mask_25|0.3958724577|0.40637245774269104|
|pl96_mask|sar_slc|mask_30|0.4116559644|0.42585596442222595|
|pl96_spike|sar_slc|spike_5sigma|0.3899208598|0.3929208598136902|
|pl96_spike|sar_slc|spike_7sigma|0.409|0.415|
|pl720_mask|sar_slc|mask_20|0.4462|0.4472|
|pl720_mask|sar_slc|mask_25|0.4569|0.4599|
|pl720_mask|sar_slc|mask_30|0.4705|0.4755|
|pl720_spike|sar_slc|spike_5sigma|0.4678270471|0.4728270471096039|
|pl720_spike|sar_slc|spike_7sigma|0.501274755|0.5112747550010681|