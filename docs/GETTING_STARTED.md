# 环境、资源与运行

## 原项目监督训练

在仓库根目录、原 `forecast` 环境运行。保留 `run_longExp.py` 和原目录布局，因此原有相对导入不变。

数据默认位于 `dataset/`；也可通过原参数 `--root_path` 指定其他位置。ETT 文件为 `ETTh1.csv`、`ETTh2.csv`、`ETTm1.csv`、`ETTm2.csv`；Weather/Electricity/Traffic 分别为同名小写 CSV，ILI 为 `national_illness.csv`。

```bash
conda activate forecast
python run_longExp.py --help
```

主实验、消融和探索脚本都仍保留在原 `scripts/` 中。先按照 `PAPER_EXPERIMENTS.md` 确认目标表和配置，不能把任意脚本当成统一正式入口。

## 论文对应的历史 scaled log-cosh

根目录 `utils/losses.py` 是导出时的原项目版本，未在整理中改写。已经复现 Weather 720 历史分数的代码另存于：

```text
reproduction/historical_runs/weather_legacy_loss_verification/
  20261114_105719_1606233/code/
reproduction/historical_runs/weather_loss_remaining_verification/
  20261117_151148_347137/code/
```

复现原论文损失对照时，应使用这里的匹配快照及对应日志/manifest 配置，不直接拿根目录后来版本替代。历史尺度为 `beta = k * median(abs(error)) / 0.6745`，停止通过尺度反传；具体下界与稳定实现以该快照的损失文件为准。代码的旧损失名、模型任务名保留。

快照完整参数已从日志抽取到 `configs/recorded/`；其中仍记录原服务器路径以保留来源。正式运行前需要明确替换数据和输出路径，而不是不加检查地执行历史绝对命令。整理过程不启动这些任务。

## 冻结预训练骨干

使用 `sacq_tsfm` 环境。在对应工程目录运行，避免根目录的 `models` 等同名包被误导入：

```bash
conda activate sacq_tsfm
cd experiments/frozen/SACQ_TSFM
PYTHONPATH="$PWD" python scripts/train.py --help
```

v1 的 `train.py`、`batch_memory_test.py`、`smoke_test.py`、`run_remaining_matrix.py` 已将硬编码项目 ROOT 改成由脚本位置推导。这里只改变文件定位，不改变训练参数。其他历史 shell 队列可能仍含原服务器路径；作为原命令记录保留，跨机器运行前应检查，不能声称它们都已完成可移植化。

冻结目录分别需要 `datasets/` 与 `models_cache/`，并使用以下模型缓存子目录：

```text
ttm-512-96-ft-r2.1/
ttm-512-720-r2/
chronos-bolt-mini/
timer-base-84m/
```

TTM 运行依赖 `granite-tsfm`；服务器环境记录版本为 0.3.1。vendor 代码保存在 v1 的 `repos/`，保留运行包和元数据，去除第三方 notebooks、生成示例和其测试文档；未执行安装。已有环境继续复用，不强制重装。新的环境应对照导出文件和第三方许可建立。

```bash
# 保持在 experiments/frozen/SACQ_TSFM 中运行
PYTHONPATH="$PWD" python scripts/validate_v2.py --head-only
```

上面是原迁移验证入口，可能导入全部 adapter 依赖。如果只想验证打包后的 head 代码，使用仓库根目录的 `tools/smoke_head.py`，无需下载模型。

## 记录与新运行

冻结工程下保留原 `results/` 中的 JSON/CSV，用于已有对照脚本的引用；不要为了整理而覆盖或伪装成新运行。部分原队列带来源 fingerprint 检查，跨目录/路径改动后可能要求新的运行标识或拒绝直接复用。不得删除或绕过这些检查来假称复现。

本地合成输入检查不等于预训练骨干端到端测试。正式数据、GPU 和全部依赖齐备后，再选单任务做前向/反向检查，不需要重跑全部论文结果。
