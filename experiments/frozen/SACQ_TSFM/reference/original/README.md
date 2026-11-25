# SACQ

长时序预测的 **plug-in 预测头**：不改骨干，只替换 Flatten Head，用「粗预测 + 条件查询残差 + 门控融合」显式建模历史到未来的对应关系。

论文草稿见 `paper.tex`：*SACQ: Memory-Conditioned Structural Decoding for Long-term Time Series Forecasting*。

## 方法

现有 LTSF 模型（如 PatchTST）常用 Flatten Head：把历史 patch 一次性映射到整个未来窗口，长 horizon 下信息瓶颈更明显。SACQ 分三步：

1. **Coarse**：线性粗预测，得到 scaffold \(F\)
2. **Cross-Self Attention**：未来 query 对历史 memory 做交叉（及可选因果自）注意力，学习残差
3. **Gated Residual**：可学习门控 \(\alpha\) 融合 coarse 与 correction

常见变体：

| 名称 | 开关 |
|------|------|
| Flatten | `--head_type flatten`（默认，原版 PatchTST 头） |
| SACQ | `--head_type sacq_cross_attention` |
| SAR | SACQ + `--sacq_autoregressive 1`（未来 query 因果自注意力） |
| S2 | `--sacq_pe_mode relative_bias` |
| DSACQ | `--head_type dsacq_cross_attention`，分布偏移感知的 coarse 分解与门控 |

## 目录

```
run_longExp.py          # 训练 / 测试入口
exp/                    # Exp_Main：训练循环、分组学习率
models/                 # PatchTST / PatchTSTMamba / DLinear / Linear
layers/                 # 骨干、SACQ / DSACQ head、RevIN
data_provider/          # 数据集与 DataLoader
utils/                  # mse / logcosh / logcosh_scaled 损失与指标
dataset/                # CSV（扁平放置，见下方）
scripts/
  PatchTST/             # Flatten 基线
  Linear/               # DLinear / Linear 基线
  ablation/             # DSACQ 与跨域消融
  experiments/          # SACQ / SAR 主实验与绘图
LGA_official/           # 捆绑：Local Geometry Attention（ICLR 2026）
TSRBench/               # 捆绑：EVT 污染 benchmark
```

## 环境

建议使用已有 conda 环境 `forecast`，或：

```bash
cd /home/bkai/CG2/SACQ
pip install -r requirements.txt
```

`requirements.txt` 列出 numpy / pandas / matplotlib / scikit-learn / torch。本机实验常用 `/home/bkai/miniconda3/envs/forecast/bin/python`。

PatchTSTMamba 另需 `pip install mamba-ssm`。

## 数据

CSV 放在 `./dataset/` 根下（**没有** `ETT-small/` 子目录）：

| 文件 | `--data` | `--enc_in` |
|------|----------|------------|
| `ETTh1.csv` / `ETTh2.csv` | `ETTh1` / `ETTh2` | 7 |
| `ETTm1.csv` / `ETTm2.csv` | `ETTm1` / `ETTm2` | 7 |
| `weather.csv` | `weather` | 21 |
| `electricity.csv` | `electricity` | 321 |
| `traffic.csv` | `traffic` | 862 |
| `national_illness.csv` | `national_illness` 或 `ili` | 7 |

格式与 [PatchTST](https://github.com/yuqinie98/PatchTST) 相同：首列 `date`，其余为变量。ETT 按 12/4/4 月划分；electricity / traffic / weather / ILI 按 70/10/20。可从 [Autoformer 数据盘](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) 下载。

目录里还有若干 `*_spike_*.csv`、`*_shift_*.csv` 以及 `*_noise/`，用于鲁棒性评估。

## 快速开始

在仓库根目录执行。

### Flatten 基线（ETTh1）

```bash
bash scripts/PatchTST/etth1.sh
```

或：

```bash
python -u run_longExp.py \
  --random_seed 2021 --is_training 1 \
  --root_path ./dataset/ --data_path ETTh1.csv \
  --model_id ETTh1_512_96 --model PatchTST --data ETTh1 \
  --features M --seq_len 512 --pred_len 96 --enc_in 7 \
  --e_layers 3 --n_heads 4 --d_model 16 --d_ff 128 \
  --dropout 0.3 --fc_dropout 0.3 --head_dropout 0 \
  --patch_len 16 --stride 8 \
  --head_type flatten \
  --train_epochs 30 --patience 12 --batch_size 128 \
  --learning_rate 0.0001 --itr 1 --des Exp
```

### SACQ / SAR + S2（ETTh1，seq_len=512）

批量（四档 pred_len、多卡队列）：

```bash
python -u scripts/experiments/run_etth1_sar2_sl512.py --gpus 0,1,2,3
```

单任务核心参数：

```bash
python -u run_longExp.py \
  --is_training 1 --short_setting 1 --skip_visual 1 \
  --root_path ./dataset/ --data_path ETTh1.csv \
  --model_id H1s512p96SAR2 --model PatchTST --data ETTh1 \
  --features M --freq h --seq_len 512 --pred_len 96 --enc_in 7 \
  --e_layers 3 --n_heads 4 --d_model 16 --d_ff 128 \
  --dropout 0.3 --fc_dropout 0.3 --patch_len 16 --stride 8 \
  --head_type sacq_cross_attention --cross_attn_layers 2 \
  --sacq_autoregressive 1 --sacq_pe_mode relative_bias \
  --dsacq_enable 0 \
  --loss mse --lradj TST --pct_start 0.2 \
  --train_epochs 30 --patience 12 --batch_size 128 \
  --learning_rate 0.0001 --des hSAR2 --itr 1
```

Weather / Electricity / Traffic 的 SAR 队列见 `scripts/experiments/run_wet_sar_sl512.py`。高通道数据请把 `batch_size` 降到 8（Electricity）或 4（Traffic），`d_model=128`、`n_heads=16`。

### 仅测试

```bash
python -u run_longExp.py --is_training 0 --skip_visual 1 \
  --root_path ./dataset/ --data_path ETTh1.csv \
  --model PatchTST --data ETTh1 --features M \
  --seq_len 512 --pred_len 96 --enc_in 7 \
  --head_type sacq_cross_attention --cross_attn_layers 2 \
  --model_id <与训练时相同>
```

跨数据集评估可用 `--checkpoint_setting`、`--scaler_data_path`、`--target_data_path`、`--eval_tag`。

## 关键参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--head_type` | `flatten` | `flatten` / `sacq_cross_attention` / `dsacq_cross_attention` |
| `--cross_attn_layers` | `2` | SACQ 交叉注意力层数 |
| `--sacq_autoregressive` | `0` | `1` = SAR |
| `--sacq_pe_mode` | `default` | `relative_bias` 为 S2 |
| `--dsacq_enable` | `1` | DSACQ 总开关；跑纯 SACQ/SAR 时请显式设 `0` |
| `--loss` | `mse` | 也可 `logcosh` / `logcosh_scaled`（评估仍用 MSE/MAE） |
| `--lradj` | `type3` | SACQ 实验常用 `TST` + OneCycle（`--pct_start 0.2`） |
| `--short_setting` | `0` | `1` 时 checkpoint 目录用短名 |
| `--skip_visual` | `0` | `1` 跳过测试 PDF |

SACQ 优化器可对 coarse / decoder / alpha / backbone 使用不同学习率（`--sacq_*_lr_ratio`）。两阶段训练见 `--init_checkpoint_setting`、`--bootstrap_flatten_head`、`--freeze_backbone`。

## 输出

| 路径 | 内容 |
|------|------|
| `checkpoints/<setting>/checkpoint.pth` | 验证集最优权重 |
| `results/<setting>/` | `pred.npy`、`true.npy`；SACQ 另存 `alpha.npy` |
| `result.txt` | 测试 MSE/MAE 等追加日志 |
| `logs/` | 各实验脚本写入的训练日志 |

## 捆绑子项目

- **LGA**：`LGA_official/`，入口 `run.py`，说明见该目录 [README](LGA_official/README.md)。
- **TSRBench**：`TSRBench/`，EVT 注入 level-shift / spike。仓库数据是扁平 `dataset/*.csv`，与部分脚本里的 `dataset/ETT-small/` 布局不同，生成污染数据时注意改路径。

## 致谢

骨干与数据协议来自 [PatchTST](https://github.com/yuqinie98/PatchTST)、[Autoformer](https://github.com/thuml/Autoformer)、[Time-Series-Library](https://github.com/thuml/Time-Series-Library)。
