# 已能从代码与记录明确的复现细节

本文件只记录已经找到的实现/数值，并区分“代码默认值”与“原结果的实际运行参数”。不因整理代码而重跑实验。

2026-11-24：根目录 [README](../README.md) 第 6–13 节已合并原监督工程与新增冻结实验的统一入口，并逐项说明污染采样、计时、MLP 宽度、adapter 张量转换、损失、优化器及消融分支。以下是先前的详细记录，最新统一使用说明以 README 为准。

## 1. 冻结主表、配置与单项运行计划

`configs/paper/frozen_table.json` 对每项提供：数据集、骨干、预测长度、预测头、三个种子或 native 单次评估、结果文件路径、聚合 JSON 内的 pointer、文件哈希、实际 config 和入口脚本。

该文件来自已选定的完成记录：ETTh1 使用 v2 确认与 MLP 确认；Weather 使用已保存的验证集选定候选及三种子确认；Electricity 使用原固定协议和 MLP 对照。未按测试误差重新选择配置。

```bash
python tools/paper_recipe.py --list
python tools/paper_recipe.py --id etth1/chronos/720/sacq --seed 2026
python tools/paper_recipe.py --id weather/ttm/96/mlp --seed 2027 --gpu 2
```

这里只显示配置和命令计划，不执行命令。计划明确 cwd、环境变量和输出位置；正式运行仍需数据、模型资源及依赖，并保留原 fingerprint 检查。对于使用标准结果路径的历史入口，重跑需使用独立副本或先明确新的输出安排，不能覆盖保留的参考记录。

原生预测为每项 seed2026 的一次评估。它不是三种子均值。新头每项用 2026、2027、2028 三个种子，均值和样本标准差按原精度保留。

## 2. 参数匹配 MLP

源代码：

- ETTh1：`experiments/frozen/SACQ_TSFM/scripts/run_mlp_control.py` 的 `build_mlp`、`width_for`。
- Electricity：v1 `scripts/run_mlp_weather_electricity.py` 的同名函数。
- 调参后 Weather：v1 `scripts/run_weather_head_tuning.py` 的 `build_mlp`。

结构为 channel-shared 的 `Flatten → Linear → GELU → Dropout → Linear`。设输入维度 `I=D×P`、隐藏宽度 `W`、预测长度 `H`，则参数量为：

```text
P_mlp = W × (I + H + 1) + H
```

代码先确定实际 FCD 目标参数量，在理想宽度的 floor/ceil 两个整数候选中选择参数差最小者，再断言实际参数量符合公式。ETTh1 的目标来自 v2 PE-on head；Weather/Electricity 来自相应旧版 head。

18 组数据集/骨干/预测长度的实际隐藏宽度、MLP 参数量、FCD 目标参数量和相对差异已导出到 `paper_results/mlp_parameter_matching.csv`，并按种子检查宽度一致。文件中的差异使用百分数，不把代码中较宽松的断言阈值直接当成全部实验的实际匹配误差。

从这 18 组最终记录计算，最大绝对参数差异为 **0.087383%**，均低于论文所写的 0.1%。此处比较的是对应预测头的可训练参数，不是整个冻结骨干的总参数量。

## 3. 延迟计时

实际定位到的代码为 `scripts/experiments/plot_sacq_latency_vs_dmodel.py`：

- `bench_forward` 在 warm-up 后同步 CUDA；每次测量前后同步，使用 `time.perf_counter()`。
- 返回排序后中间位置的测量值，即代码所取的中位位置值；虽然原 docstring 写 mean，实际返回值不是平均值。本次不改原函数。
- `bench_sacq_breakdown` 分别计时完整 forward 与 encoder+coarse，取两者差，并用 `max(..., 0)` 截断负值。
- 代码默认参数是 batch=32、warmup=30、repeats=80；输入是随机 CUDA tensor，形状为 `[batch,512,7]`，预测长度 336，运行在 eval/no_grad 下。
- 已保存的延迟 CSV 中 batch_size 为 64，因此不能把脚本当前默认 batch=32 写成原图的运行 batch。CSV 没有保存 warmup/repeats，暂不能确认原图实际使用的这两个参数。

因此，同步方式、计时范围和统计方法已能说明；原图的具体 warm-up/repeat 运行参数仍需原命令或日志，不猜测，也不为整理而重跑。

## 4. 训练 spike 生成

`scripts/experiments/outlier_utils.py` 包含训练标量级 spike 注入。实现使用独立的 `np.random.default_rng(seed)`；抽取 `(时间,变量)` 标量位置，加入随机正负号、幅度按该变量训练标准差缩放的扰动。具体实验比例、幅度和 seed 由调用脚本传入。

其中 `inject_spike_outliers` 当前默认 seed 为 2021；同时污染训练/测试的函数分别接受 train_seed、test_seed。存在个别脚本对特定任务使用单独注入种子的分支，所以不能把某个函数的默认 seed 一概写成所有图的实际 seed。

这一项明确了可定位的实现，但没有将所有污染图的结果与某一个默认注入器强行绑定。input masking、target spike、tail shift 仍需各图对应的调用入口和配置一一列明。

## 5. 历史损失实现

与 Weather 720 历史 scaled-log-cosh 复现对应的文件为：

```text
reproduction/historical_runs/weather_legacy_loss_verification/
20261114_105719_1606233/code/utils/losses.py
```

自动尺度为 `k × median(abs(error)) / 0.6745`，截断到 `[1e-6, 1e6]` 后 detach。稳定计算采用 `abs(z) + log1p(exp(-2*abs(z)))`，再乘 beta² 并取均值。文件按历史实现保留了省略 `-log(2)` 的约定，本轮不改写历史损失数值或训练选择逻辑。

根目录当前 `utils/losses.py` 包含后来 IRSE/EMA 等实现，与上述历史版本分开保存。论文对应复现需选择完整的匹配快照，而不仅是把单个损失文件覆盖回根目录。

## 6. 后续仍未完成的对应项

- 监督主表和组件表尚未全部完成逐行命令/配置映射；已有定点复现参数继续保存在 `configs/recorded/`。
- 各污染图的精确注入种子及执行入口还需从对应记录定位。
- 原延迟图的 warm-up/repeat 实际命令仍缺记录。
- 完整 GPU adapters 测试尚未在整理副本中执行；CPU head smoke 不能替代它。
- 当前环境快照已保存，但不能代替每个历史任务的环境记录；第三方许可、最终仓库地址仍待处理。
