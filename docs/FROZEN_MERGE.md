# 冻结实验统一目录

运行目录：`experiments/frozen/SACQ_TSFM/`。

- adapters、data_provider 和 FCD 核心共用；合并前两个核心类的 AST 相同。
- `models/factory.py` 保留默认 PE-off 行为，Weather/Electricity 入口不变。
- `models/factory_pe.py` 保留独立 future PE 开关。迁入的 `run_v2_confirmation.py`、`run_v2_pilot.py`、`run_mlp_control.py`、`validate_v2.py` 显式从此入口导入。
- ETTh1 确认、MLP、PE pilot 结果迁入同一 results 目录，结果内容未重写。
- 历史文件内的服务器绝对路径与 fingerprint 作为记录保留。合并后的新运行生成新 fingerprint，不能伪装成旧运行。
- `installation_manifest.json` 验证合并后的运行文件；旧回执及源码在 作者本地保存的合并归档（不随代码分发） 中保留。
- 主表索引、资源关联、命令查询与 smoke 工具指向统一目录。源清单中旧路径通过 ZIP 校验，仍保留来源证据。

合并未改变 FCD 核心、损失、batch、学习率或随机种子，未执行训练或 GPU 评估。
