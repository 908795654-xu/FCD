# 冻结预训练骨干实验

此目录统一提供 TTM、Chronos-Bolt-Mini、Timer-84M 的 adapter、预测头和全部实验记录。

- `models/factory.py`：固定协议使用的默认入口，不额外添加 future PE。
- `models/factory_pe.py`：ETTh1 确认与位置编码对照使用的显式 PE 开关入口；保留此前配方默认设置。
- 两个入口共用 `adapters/`、`sacq_core/` 和数据加载器，不改变既有训练协议。
- `scripts/run_v2_confirmation.py`、`run_mlp_control.py`、`run_v2_pilot.py` 为保留名称的迁入入口，并非独立工程。
- `results/confirmation_fixed_v1/`、`mlp_control_v1/` 和 `pilot_pe_control_v1/` 保留结果原始字节；其中的旧服务器绝对路径是历史元数据。

配置与单项命令见仓库根目录 `tools/paper_recipe.py`。历史目录与安装回执保存在根目录 作者本地保存的合并归档（不随代码分发）。
