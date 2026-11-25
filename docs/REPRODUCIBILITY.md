# 复现说明与尚需补齐的材料

本文件对应老师清单第三部分。整理保留真实代码和记录；以下未注明已完成的部分，不推测默认值或补造历史参数。

|项目|已有材料|尚需补齐/确认|
|---|---|---|
|输入污染|`TSRBench_official/`、`scripts/experiments/` 和 CSV；旧工具按 EXTERNAL_BASELINES.md 外部获取|论文各图对应的 seed、位置抽样、幅度和命令|
|mask、spike、tail shift|原训练、评估及绘图脚本|逐图标明执行入口与固定参数；保持现有数据映射|
|效率测试|已定位同步、perf_counter、返回中位位置值和完整/coarse 计时范围|CSV 的 batch=64；脚本默认 warmup=30/repeats=80，但原图实际 warmup/repeats 仍待命令证据|
|参数匹配 MLP|已从记录提取 18 组隐藏宽度、参数量、匹配误差，并检查公式及种子一致性|见 paper_results/mlp_parameter_matching.csv；未重新训练|
|adapters|两版 TTM、Chronos、Timer adapters 及原验证脚本|在 GPU/模型资源齐备处验证整理后的实际特征输出；本地 CPU head 检查不替代此项|
|scaled log-cosh|原项目当前实现与复现成功的历史代码快照分开保存|论文复现入口使用匹配版本，明确 beta 下界、稳定表达式及 detach，不混为同一个默认文件|
|optimizer/环境|完整环境快照、历史 Namespace、冻结固定配置|将剩余表行映射到实际调度/优化器/早停和种子；环境快照不证明历史环境从未变化|
|组件消融|原消融脚本及保留结果|逐行给出 Full 与消融开关和其余固定设置，不依赖后来 argparse 默认值|

## 模型与数据约定

- 原代码中的 FCD/SACQ 为 patch 级预测头，粗预测、查询构造、注意力修正与门控的实现均未在整理中改写。
- 数据预处理标准化与模型内部 RevIN 区分；输出恢复及损失/指标空间由实际数据加载器、adapter 和训练入口共同确定。
- frozen profile 的具体开关在各工程 `configs/profile.json` 中保留，不能从“Full”一词推导所有实验的开关。
- 参数搜索由验证集选择，测试集用于选定配置评估；整理不重新依据测试集挑选配置。

## 当前检查范围

已新增 [逐项来源索引](SUPERVISED_SOURCE_MAPPING.md)：监督主表 28 组中 16 组、组件表 20 组中 5 组找到符合匹配条件的最终评估日志候选，尚不称为全部来源已确认。mask/spike 绘图数据 104 行均对应原项目展示 CSV，其中 92 行还对应原始评估 CSV；另外 12 行的展示值与原始评估值分别保留，不自动统一。tail-shift、门控 α 图和效率图已建立绘图入口到数据记录的关系，具体历史运行参数仍以证据是否存在区分。效率图实际 warm-up/repeats 尚未找到记录。

`tools/check_repository.py` 检查语法、导入文件完整性及记录的四处纯路径修正。`tools/smoke_head.py` 检查合成输入的形状、有限输出和有限梯度。没有执行完整训练、GPU adapter 或跨机器全部队列复现。
