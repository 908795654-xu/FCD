# 论文与实验入口索引

本索引区分已收集材料、可定位入口与尚未逐项补齐的命令；不把所有探索脚本都称为正式论文配方。

## 原项目

|论文内容|代码/记录位置|对应说明|
|---|---|---|
|监督训练通用入口|`run_longExp.py`、`exp/exp_main.py`|原入口与训练循环保持原文|
|Flatten 基线|`scripts/PatchTST/`|按任务确认配置，不自动视为论文所有默认设置|
|跨骨干|`scripts/experiments/run_backbone_flatten_vs_sacq.py`|Flatten 与 FCD 使用不同损失的完整配置对照|
|损失与超参数|`reproduction/historical_runs/`、`configs/recorded/`、`paper_results/supervised/`|已收集的定点复现和参数记录；保留历史 loss 版本|
|mask/spike/tail shift|`scripts/experiments/` 下相应 `run_*`、`eval_*`、`plot_*`|保留实现，具体种子与开关从对应记录填写|
|效率|`scripts/experiments/plot_sacq_params_latency_combined.py` 及 profiling 脚本、`paper_results/supervised/logs/sacq_param_sweep/`|绘图数据与实际计时实现分开对应|

已核实主结果配置：ETTh1 720 为 L=2,k=1；Weather 720 为 L=1,k=4。它们不意味着其余任务参数相同。完整命令以对应记录为准。

监督主表 28 组、组件表 20 组的逐项日志候选，以及四类实验图的入口/数据关系见 [监督来源索引](SUPERVISED_SOURCE_MAPPING.md)。源日志原 Namespace 与最终评估行号保存在 `reproduction/source_relations/table_sources.json`，复制证据附 SHA256。候选匹配不等于已确认历史来源；缺少对应的表行不按测试分数自动选择替代项。

## 冻结骨干的三类对照

|对照/数据|入口|已有记录|
|---|---|---|
|最初固定协议 Linear/FCD|v1 `scripts/train.py`|v1 `results/formal/`|
|调参后 ETTh1 旧版确认|v1 `scripts/confirm_etth1_worker.py`|v1 `results/confirmation_etth1_v1/`|
|调参后 ETTh1 v2 确认|v2 `scripts/run_v2_confirmation.py`|v2 `results/confirmation_fixed_v1/`|
|ETTh1 参数匹配 MLP|v2 `scripts/run_mlp_control.py`|v2 `results/mlp_control_v1/`|
|Weather/Electricity MLP|v1 `scripts/run_mlp_weather_electricity.py`|v1 `results/mlp_control_frozen_v1/`|
|Weather 局部调参|v1 `scripts/run_weather_head_tuning.py`|v1 `results/weather_head_tuning_v1/`|
|原生预测|v1 `scripts/native_eval.py`|v1 `results/native/`|

所有冻结入口统一位于 `experiments/frozen/SACQ_TSFM`。表中 v1/v2 仅标识历史配方：默认 factory.py 和显式 PE 开关 factory_pe.py。

冻结骨干实验的历史 native 对照为 seed2026 的一次评估；新头的最终对照有三种子。不要把单次 native 评估称为三种子均值，也不要把历史 seed2026 对照和调参后均值自动混用。

`paper_results/frozen_record_index.csv` 枚举了已收集的单任务测试记录，包括历史和调参版本，不是新的最终汇总表。现成汇总 CSV/JSON 保留在相应结果目录；后续论文表的逐行来源映射应注明使用哪一个版本，不按测试误差重新挑选配置。

## 已完成的冻结主表逐行对应

最终对应见 `docs/FROZEN_TABLE_MAPPING.md`，精确配置和来源见 `configs/paper/frozen_table.json`。共 18 行、72 组方案、180 份种子记录，已与当前 V5 的 144 个指标单元格核对，无显示精度差异。这里的结果版本已经明确，不需要从上面的所有历史目录中重新选择。

可用 `python tools/paper_recipe.py --id etth1/chronos/720/sacq --seed 2026` 查看一项的实际参数、来源 pointer、元数据及命令计划。该工具绝不执行任务。对 180 项计划都已检查入口、来源哈希和配置可定位性；尚未以这些计划重跑 GPU 训练。
