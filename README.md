# FCD: Forecast-Conditioned Decoding for Time Series Forecasting

本仓库提供 FCD 的模型实现、训练与评估代码。FCD 利用粗预测构造内容条件查询，从历史表示中提取修正信息，并通过门控融合生成最终预测。代码中保留 `SACQ`、`SAR` 等类名和配置名，以兼容实验脚本及 checkpoint。

论文通过监督预测、组件与损失消融、鲁棒性及效率分析，以及冻结预训练骨干上的预测头对照，评估该方法的预测性能与适用性。

## 1. 目录结构

```text
run_longExp.py                  监督预测训练与评估入口
models/                        模型定义
layers/                        骨干、FCD 预测头、RevIN 与位置编码
exp/                           训练、验证及测试流程
data_provider/                 数据加载与标准化
utils/                         损失函数及辅助工具
scripts/                       实验配置、消融、污染评估与绘图
experiments/frozen/
  SACQ_TSFM/                   冻结预训练骨干实验主工程
configs/                       实验配置与结果来源索引
paper_results/                 实验结果与汇总记录
reproduction/                  复现快照及图表输入数据
environments/                  软件环境配置
docs/                          补充实现与复现说明
tools/                         配置查询、资源关联及检查工具
```

## 2. 环境与数据

监督预测与冻结骨干实验分别提供环境配置：

- [forecast.yml](environments/snapshots/forecast.yml)：监督预测实验。
- [sacq_tsfm.yml](environments/snapshots/sacq_tsfm.yml)：冻结预训练骨干实验。

环境导出记录 Python 3.10.0、PyTorch 2.10.0、NumPy 1.24.4、pandas 2.3.3。冻结环境另包含 transformers 4.53.3、granite-tsfm 0.3.1 和 chronos-forecasting 1.5.3。更早的依赖文件保存在 `environments/requirements_original.txt`；环境导出不代表全部历史任务均使用相同版本。

数据集、预训练权重和 checkpoints 单独准备。资源路径可通过以下工具关联，默认只显示计划，添加 `--apply` 后创建链接：

```bash
python tools/configure_assets.py \
  --data-root /path/to/dataset \
  --model-cache /path/to/models_cache
```

数据目录、预训练模型布局及运行环境说明见 [GETTING_STARTED](docs/GETTING_STARTED.md)。

## 3. 实验与运行入口

|实验|骨干与训练方式|入口|
|---|---|---|
|监督预测|骨干与 FCD 预测头联合训练|`run_longExp.py`、`scripts/`|
|组件与损失消融|按配置替换组件或训练目标|`scripts/ablation/`、`scripts/experiments/`|
|鲁棒性与效率|输入／标签污染评估、门控分析及延迟测试|第 5–6 节列出的评估与 profiling 脚本|
|冻结骨干预测头对照|冻结 TTM、Chronos-Bolt-Mini、Timer-84M，训练 Linear、MLP 或 FCD；使用 MSE|`experiments/frozen/SACQ_TSFM/`|
|原生预测对照|直接使用预训练模型的预测方案，不训练新头|冻结工程 `scripts/native_eval.py`|
|位置编码验证|比较 future PE 开关，保留配对验证与确认代码|`experiments/frozen/SACQ_TSFM/`|

冻结实验主入口包括 `scripts/train.py`、`scripts/run_weather_head_tuning.py` 和 `scripts/run_mlp_weather_electricity.py`。ETTh1 的确认与 MLP 运行文件还包括同目录中的 `scripts/run_v2_confirmation.py`、`scripts/run_mlp_control.py`。

冻结表的配置及结果文件列于 [frozen_table.json](configs/paper/frozen_table.json)。新头使用 2026、2027、2028 三个随机种子；native 每个设置评估一次。具体学习率、weight decay、dropout 和 batch 以各项配置为准。

```bash
# 列出配置
python tools/paper_recipe.py --list

# 查看一项实验的配置、工作目录与运行命令
python tools/paper_recipe.py --id weather/ttm/96/sacq --seed 2026
python tools/paper_recipe.py --id electricity/timer/720/mlp --seed 2027 --gpu 2
```

配置查询工具不执行任务。运行命令前需准备资源，并为新的输出安排独立目录，避免覆盖参考结果。

## 4. 检查与结果索引

```bash
python tools/check_repository.py
python tools/smoke_head.py --profile v1
```

静态检查验证 Python 语法和文件完整性；CPU smoke 使用合成输入检查预测头的前向与梯度，不依赖数据集或预训练权重。位置编码验证另见`scripts/validate_v2.py`。

- [实验入口与配置](docs/PAPER_EXPERIMENTS.md)
- [冻结预测结果来源](docs/FROZEN_TABLE_MAPPING.md)
- [监督实验与图表来源](docs/SUPERVISED_SOURCE_MAPPING.md)
- [补充实现细节](docs/RECORDED_IMPLEMENTATION_DETAILS.md)

### 4.1 论文实验入口

下表区分模型运行入口、已完成结果的查询入口和历史图表的绘制入口。冻结
实验均在 `experiments/frozen/SACQ_TSFM/` 运行；其他模型入口在仓库根目录运行。

|论文内容|入口／配置来源|使用范围|
|---|---|---|
|表 1：监督预测|`run_longExp.py`、`scripts/PatchTST/`、`configs/recorded/`|按任务配置运行；不是所有任务共用一个默认 L/k|
|表 2：跨骨干|`scripts/experiments/run_backbone_flatten_vs_sacq.py`|Flatten+MSE 与 FCD+scaled log-cosh 的配方对照|
|表 3：冻结骨干|`tools/paper_recipe.py --list`，再以 `--id ... --seed ...` 查询单项命令|72 组方案、180 份记录；查询工具本身不执行训练|
|表 4：组件消融|本 README 第 11 节、`docs/SUPERVISED_SOURCE_MAPPING.md`|可定位分支与未确认的行分别注明，不将整个修正栈关闭等同于仅去掉 cross-attention|
|表 5：损失消融|`reproduction/historical_runs/` 下的历史完整代码快照与 `configs/recorded/index.json`|Weather 720 使用匹配历史 median-scale loss 的复现及对照记录；根目录 IRSE/EMA 分支不作为等价替代|
|表 6：L/k 敏感性|`scripts/experiments/run_sacq_Lk_sensitivity.py`、`paper_results/supervised/`|配置和已记录结果可查询；历史 metrics.json 汇总接口不作为已验证的一键重跑保证|
|表 7：测试输入污染|`TSRBench_official/`；`docs/EXTERNAL_BASELINES.md`|官方工具与历史生成版本区分，不用最新版自动替换已有基准|
|图 3–5：污染／门控分析|`reproduction/source_relations/figure_relations.json`、本 README 第 5 节|绘图、输入扰动评估与训练污染协议分别标识|
|图 6：历史长度|`scripts/experiments/` 中 seqlen/lookback 相关入口；图表来源索引|按各曲线固定模型配置改变历史长度|
|图 7：效率|`scripts/experiments/plot_sacq_latency_vs_dmodel.py`，本 README 第 6 节|现存 profiling 入口及历史 CSV；原图未记录的次数不以默认值补写|

### 4.2 实验采用的实现与复现范围

- **FCD 建模粒度**是 future patch。先生成粗预测 F，再构造内容查询 C，
  最后输出门控修正 `F + alpha*Delta`。独立 future PE 是头部参数，
  不扩展或复用 TTM、Chronos、Timer 的 encoder 位置表。
- **冻结实验**使用双向未来 self-attention。默认 `models/factory.py` 不加
  独立 future PE；`models/factory_pe.py` 提供 PE 开关。ETTh1 最终 SACQ
  记录包含 PE-on，Weather/Electricity 对应默认配方，以结果索引为准。
- **内容融合的区别**：冻结默认路径用 C 初始化查询，但没有在每层
  cross-attention 后显式执行论文公式（2）的 `[updated_query || C] -> Linear`。
  监督实现中的可选 AR 上下文适配，也不应直接视作该重复融合公式的等价实现。
  所保留结果对应实际代码变体，不单独证明该公式在冻结条件下的贡献。
- **损失版本**：本文 median-scale scaled log-cosh 的历史复现快照与根目录
  后续 IRSE/EMA 实现分开保留。冻结表统一使用 MSE。
- **污染协议**：已定位的 `eval_etth1_train_corruption_sl512.py` 是对指定 split
  的历史输入加扰动并评估，不等于污染训练标签后重新训练；tail-shift 脚本
  包含历史输入尾部 shift 和 near/far 指标，不作为正文“训练监督尾部污染”的
  同义入口。对应历史协议的缺失记录、12 个展示值与原始评估值的差异见来源索引。
- **效率协议**：现存 profiling 代码采用 pred_len=336，论文效率节写 96；
  历史 CSV batch=64，但 warm-up/repeats 未记录。当前可运行默认配置不能
  自动等同于原图配置。

代码、配置和已报告结果分别保存；来源不完整的实验不标注为已完整重跑验证。
以下说明不改写论文、历史结果或模型实现，也不将 README 的说明视为消除
正文与实现差异。GPU 端到端复现不在本次发布检查范围内。

下文给出污染采样、效率测量、参数匹配、adapter、损失、训练配置和组件消融的实现说明。

## 5. 数据污染与采样

监督预测实验使用 `[B,T,C]`，冻结接口使用 `[B,C,T]`。下面的污染代码用于监督预测实验，不默认用于冻结骨干表。

|内容|代码|采样与作用位置|
|---|---|---|
|固定污染 seed|`scripts/experiments/eval_etth1_train_corruption_sl512.py` 的 `corruption_seed` / `eval_split`|`base + int(md5(corruption_name)[:8],16) % 10000`；评估使用独立 `np.random.default_rng(seed)`，不使用 Python 不稳定的 `hash()`|
|input masking|同文件 `corrupt_batch` 的 `mask` 分支|每个样本无放回抽取 `max(1,round(T*ratio))` 个时间位置，将这些位置所有变量置零；clean 分支不置零|
|输入 OT spike|同文件 `spike` 分支|每个样本抽一个时间位置、随机正负号，在历史 OT 通道添加 `±sigma_mult*ch_std[OT]`；不改未来标签|
|训练数据／target spike|`scripts/experiments/outlier_utils.py` 的 `_inject_spikes_block`、`inject_spike_outliers`、`inject_spike_outliers_bilateral`|对 CSV 数值块的标量位置无放回采样，数量为 `round(block.size*ratio)`，添加随机正负号的 `noise_mult*训练列标准差`；被污染位置之后可能进入输入或预测目标，不能称为只污染输入|
|tail shift|`scripts/experiments/eval_sar_vs_sacq_spike_train_shift.py` 的 `apply_shift_x` / `eval_shift_paired` / `eval_all_shifts_paired`|历史 OT 最后 50%（`TAIL_FRAC=0.5`）添加 `±pct*ch_std[OT]`；batch 独立 seed，配对模型复用同一受扰输入；未来标签保持原值|

CSV spike 生成器记录 `.meta.json`，包含 seed、比例、幅度、训练列标准差、位置总数与污染数量。训练/测试双侧污染分别接受 `train_seed` 和 `test_seed`。标准差使用 `ddof=0`，小于 `1e-8` 的列替换为 1。只污染训练段和双侧污染是不同入口，选择实际实验对应版本。

另一条历史评估入口 `eval_etth1_train_corruption_sl512.py` 的 shift 分支直接在模型输入空间增加 `sign*uniform(low,high)`，并不再乘标准差；不能与上表 tail-shift 入口的幅度定义混用。污染位置依赖 seed、采样顺序及 batch 划分，复现时也应固定数据顺序和 batch。

门控图使用 `plot_spike_outlier_alpha_curves.py::inject_input_spikes`：OT 上多位置注入，`collect_alpha_curves` 以 `2021+batch序号` 作为输入污染 seed。图数据和入口见 [监督图表来源索引](docs/SUPERVISED_SOURCE_MAPPING.md)。当前默认 seed 不替代没有记录的历史命令。

## 6. Latency profiling

计时代码为 `scripts/experiments/plot_sacq_latency_vs_dmodel.py` 的 `bench_forward` 与 `bench_sacq_breakdown`：

- eval / no_grad；先 warm-up，再计时；warm-up 结束和每次计时前后执行 CUDA synchronization。
- 使用 `time.perf_counter()`，排序后取中间位置的测量值；统计方式以函数返回值为准。
- 分别测完整 forward 和 encoder+coarse；额外部分以两者延迟之差估计，并截断到非负值。
- 当前可运行默认配置为 `warmup=30`、`repeats=80`、`batch=32`，随机输入 `[B,512,7]`、预测长度 336。以脚本 `--help` 查看并显式指定所需设置。

论文延迟图的数据 `reproduction/source_relations/original/logs/sacq_param_sweep/sacq_latency_sweep.csv` 记录 batch=64，未保存 warm-up 与重复次数。上面的 30/80 是现存实现默认值，对应的历史运行参数未记录。完整绘图入口为 `scripts/experiments/plot_sacq_params_latency_combined.py`。

## 7. Parameter-matched MLP

MLP 是每个变量共享参数的 `Flatten → Linear → GELU → Dropout → Linear`。输入 `I=D*P`，隐藏宽度 `W`，输出长度 `H`，参数量为 `W*(I+H+1)+H`。`width_for` 在理想宽度的 floor/ceil 中选择与对应 SACQ 预测头可训练参数最接近的整数，并核对实际计数。

|骨干|ETTh1 W，96 / 720|Weather W，96 / 720|Electricity W，96 / 720|
|---|---:|---:|---:|
|TTM|386 / 681|375 / 627|375 / 627|
|Chronos-Bolt-Mini|352 / 943|348 / 921|348 / 921|
|Timer-84M|3652 / 3972|3636 / 3864|3636 / 3864|

逐项参数量和匹配误差见 [mlp_parameter_matching.csv](paper_results/mlp_parameter_matching.csv)，18 组最大绝对误差为 0.087383%。匹配对象是预测头，不含冻结骨干。

- ETTh1：PE 配方 `scripts/run_mlp_control.py` 的 `width_for` / `build_mlp`。
- Electricity 固定协议：默认配方 `scripts/run_mlp_weather_electricity.py` 的同名函数。
- Weather 最终调参：默认配方 `scripts/run_weather_head_tuning.py` 的 `build_mlp`。

ETTh1 对应 PE-on SACQ 参数量；另外两个数据集对应 PE-off 配方。它们的隐藏宽度不同是匹配目标不同，不能任意统一。

## 8. Pretrained adapters

冻结实验的适配器位于 `experiments/frozen/SACQ_TSFM/adapters/`，预测头由 `models/factory.py` 构建，并通过 `models/sacq_forecaster.py` 连接骨干。PE 配方共用这些接口。设 B 为 batch、C 为变量数、T 为历史长度、D 为特征维度、P 为 patch 数：

```text
values [B,C,T] → adapter → memory [B,C,D,P]
                         + loc/scale [B,C,1]
               → Linear / MLP / SACQ → normalized forecast [B,C,H]
               → forecast*scale+loc  → prediction [B,C,H]
```

|Adapter|张量转换与实际特征提取|
|---|---|
|`ttm_adapter.py`|转为 `[B,T,C]` 及对应 observed mask，调用预训练模型，取 `decoder_hidden_state`，排列为 `[B,C,D,P]`；保留 loc/scale 与频率 token 处理|
|`chronos_bolt_adapter.py`|合并 batch 与变量为 `[B*C,T]`，调用 `model.encode`；若有 REG token 则移除，历史 hidden `[B*C,P,D]` 重排为统一 memory|
|`timer_adapter.py`|合并为 `[B*C,T]`，基于 observed mask 做归一化，左侧补齐 token 长度整倍数；调用 `model.model` 取 `last_hidden_state`，重排为统一 memory|

`base.py::BackboneFeatures.denormalize` 负责尺度恢复。这里恢复的是 adapter 的内部归一化尺度，不应与数据加载器的训练集标准化混淆。骨干 `requires_grad=False`、保持 eval、特征提取处 `torch.no_grad()`；`HeadForecaster.train()` 只让头进入训练状态。P 和 D 由实际模型确定，不强行相同。

验证入口：PE 配方 `scripts/validate_v2.py --head-only`、`--all-backbones --gpu <id>`；默认配方 `scripts/batch_memory_test.py`。

## 9. Scaled log-cosh

历史 Weather 720 对应的完整代码快照在：

```text
reproduction/historical_runs/weather_legacy_loss_verification/
20261114_105719_1606233/code/
```

该快照 `utils/losses.py` 的 auto_mad 分支为：

```text
e = prediction - target
beta = clamp(k * median(abs(e)) / 0.6745, min=1e-6, max=1e6).detach()
z = e / beta
loss = mean(beta**2 * (abs(z) + log1p(exp(-2*abs(z)))))
```

非正指数避免直接计算 cosh 溢出；beta 下界避免除零，detach 不通过尺度估计反传。fixed 模式另走固定 beta 分支。历史代码省略 `-log(2)`，实现保持这一约定；beta 随 batch 变化时，不把这个省略简单视为整个训练过程中的固定常数。

根目录 `utils/losses.py` 还包含后来 IRSE/EMA 分支，与历史版本分别保留。复现特定历史表应使用匹配的完整快照，不能只覆盖单个 losses.py 而保留不兼容的 factory 调用。冻结预训练骨干实验使用 MSE，不使用 scaled log-cosh。

## 10. 优化器与训练配置

|体系|优化器与训练设置|
|---|---|
|监督预测实验|`exp/exp_main.py::_select_optimizer` 使用 Adam；支持 backbone/coarse/decoder/alpha 参数组及学习率比例；scheduler 和早停见同文件训练循环及实际 Namespace|
|冻结 `frozen_v1`|AdamW，仅更新头；通常 lr=1e-4、weight_decay=0.01、最多 50 轮、patience=7；ReduceLROnPlateau(mode=min,factor=0.5,patience=1,min_lr=1e-6)|
|冻结 `paper` 分支|默认配方 `scripts/train.py` 使用 Adam、无该 scheduler；不要与 frozen_v1 混用|
|调参后确认|最终 lr、weight_decay、dropout 及 batch 按 `configs/paper/frozen_table.json` 各项实际配置，不用上一行的常见值替换|

优化器调用未覆盖的 PyTorch 默认项包括 `betas=(0.9,0.999)`、`eps=1e-8`、`amsgrad=False`。Adam 未传 weight_decay 时为 0；AdamW 和冻结 Adam 分支使用入口传入的 weight_decay。其余运行选项按所用 PyTorch 版本解析；可用 `python tools/show_runtime_defaults.py` 打印完整默认字典、软件版本和 CUDA 信息，不训练模型。监督预测 Adam 参数组的 lr 比例需同时保存，不能只报告一个标量 lr。

完整软件版本见第 2 节的环境文件；优化器配置需要与所用 PyTorch 版本一并记录。

冻结最终命令由 `tools/paper_recipe.py` 显示；监督预测实验的运行参数见 `configs/recorded/` 及 `reproduction/source_relations/table_sources.json` 的 Namespace。离线模型、CUDA_VISIBLE_DEVICES、PYTHONPATH 与工作目录也是复现配置的一部分。batch 固定为实际记录，不因显存不足自动更改。

## 11. 组件消融

|机制|可定位代码／分支|使用说明|
|---|---|---|
|预测头实现入口|`layers/PatchTST_backbone.py::SACQCrossAttentionHead` / `SACQDecoderStack`|实现粗预测、注意力修正和 alpha 门控|
|固定门控（−G）|`sacq_alpha_mode='fixed'` 与 `sacq_alpha_value=1`|对照可学习 alpha，保留其他设置|
|仅粗预测 Coarse|`cross_attn_layers==0` 返回零修正；`scripts/experiments/eval_*_coarse_only.py` 另有直接评估粗预测路径|后者从已有 checkpoint 提取 coarse，不等于重训一个 coarse-only 模型|
|cross-attention|`SACQDecoderStack.forward` 的 cross-layer 调用|`cross_attn_layers=0` 同时取消整个修正栈，不能直接称为只去掉 cross-attention 的 −CA|
|causal attention|`sacq_autoregressive`、`sacq_future_causal` 与 `_resolve_self_attn_mask`|按实际配置选取，不从 Full 标签反推开关|
|位置编码|`models/factory_pe.py` 与独立 future PE，`scripts/validate_v2.py`|按各实验配置显式启用|
|历史 −CA / −Fus 行|现有组件表来源索引与继承脚本|现有材料未提供已确认对应这两行的独立运行配方；详见组件表来源索引|

`scripts/ablation/align_component_ablation_etth1_pl96.sh` 实际是 ETTh1→ETTh2、预测长度 96 的探索配置，不作为论文 720 组件表的通用复现命令。该脚本的适用范围与预测长度应按其配置使用。

## 12. 第三方来源与许可

第三方许可和修改说明见 [THIRD_PARTY](THIRD_PARTY.md) 与 [NOTICE](NOTICE.md)。
LGA 及来源未确认的旧版污染工具不随仓库捆绑；官方地址、外部路径配置和
独立 TSRBench 的使用说明见 [EXTERNAL_BASELINES](docs/EXTERNAL_BASELINES.md)。
FCD 作者贡献按 [Apache License 2.0](LICENSE) 发布；第三方组件继续适用各自许可，
详见 [THIRD_PARTY](THIRD_PARTY.md) 与 [NOTICE](NOTICE.md)。许可证不自动覆盖
数据集、模型权重或单独发布的论文文本。
