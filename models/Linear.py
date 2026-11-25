"""
模型名 Linear：预测头之前的主干仅为多层 MLP（无 Transformer、无序列分解）。

- 主干：共享 MLP 将每个 patch 映射到 d_model（或「direct」模式下整段序列一次 MLP 到 pred_len）。
- 预测头：head_type 为 flatten或 sacq_cross_attention，与 PatchTST 一致。
- head_type=direct：不设 patch 预测头，主干 MLP 直接从 [B,C,seq_len] 映射到 [B,C,pred_len]（最弱端到端基线）。

（历史别名 head_type=linear 等价于 direct，仍会被接受。）
"""

import torch
import torch.nn as nn

from models.PatchTST import Model as PatchTSTShell


def _mlp_stack(in_dim, out_dim, hidden_dim, n_hidden, dropout, act_layer=nn.GELU):
    """n_hidden: 隐层个数；0 表示单层 Linear(in_dim, out_dim)。"""
    if n_hidden <= 0:
        return nn.Linear(in_dim, out_dim)
    layers = [
        nn.Linear(in_dim, hidden_dim),
        act_layer(),
        nn.Dropout(dropout),
    ]
    for _ in range(n_hidden - 1):
        layers += [
            nn.Linear(hidden_dim, hidden_dim),
            act_layer(),
            nn.Dropout(dropout),
        ]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class LinearPatchEncoder(nn.Module):
    """与 TSTiEncoder 相同接口：输入 [bs, nvars, patch_len, patch_num] → [bs, nvars, d_model, patch_num]。"""

    def __init__(self, configs):
        super().__init__()
        self.patch_len = int(getattr(configs, "patch_len", 16))
        d_model = int(getattr(configs, "d_model", 128))
        d_ff = int(getattr(configs, "d_ff", 128))
        depth = int(getattr(configs, "linear_mlp_depth", 2))
        dropout = float(getattr(configs, "linear_mlp_dropout", 0.1))
        self.mlp = _mlp_stack(self.patch_len, d_model, d_ff, depth, dropout)

    def forward(self, x):
        x = x.permute(0, 1, 3, 2)
        x = self.mlp(x)
        x = x.permute(0, 1, 3, 2)
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.individual = bool(getattr(configs, "individual", 0))
        raw_head = getattr(configs, "head_type", "direct")
        # 「linear」易与「主干叫 Linear」混淆，仅作 direct 的别名
        self.head_type = "direct" if raw_head in ("linear", "direct") else raw_head

        # 主干 MLP 深度 / dropout（patch 嵌入与 direct 模式共用）
        self.linear_mlp_depth = int(getattr(configs, "linear_mlp_depth", 2))
        self.linear_mlp_dropout = float(getattr(configs, "linear_mlp_dropout", 0.1))
        self.d_ff = int(getattr(configs, "d_ff", 128))
        self.d_model = int(getattr(configs, "d_model", 128))
        self.n_heads = int(getattr(configs, "n_heads", 4))
        self.cross_attn_layers = int(getattr(configs, "cross_attn_layers", 1))
        self.head_dropout = float(getattr(configs, "head_dropout", 0.0))
        self.save_attention_map = bool(getattr(configs, "save_attention_map", False))
        self.sacq_overlap_avg = bool(getattr(configs, "sacq_overlap_avg", 0))
        self.sacq_alpha_mode = getattr(configs, "sacq_alpha_mode", "learnable")
        self.sacq_alpha_value = float(getattr(configs, "sacq_alpha_value", 0.5))

        self.use_patch_head = self.head_type in (
            "flatten",
            "sacq_cross_attention",
        )

        if not self.use_patch_head:
            # direct：主干即全部，无 Flatten/SACQ 等预测头
            if self.individual:
                self.backbone_direct_mlps = nn.ModuleList()
                for _ in range(self.channels):
                    self.backbone_direct_mlps.append(
                        _mlp_stack(
                            self.seq_len,
                            self.pred_len,
                            self.d_ff,
                            self.linear_mlp_depth,
                            self.linear_mlp_dropout,
                        )
                    )
            else:
                self.backbone_direct_mlp = _mlp_stack(
                    self.seq_len,
                    self.pred_len,
                    self.d_ff,
                    self.linear_mlp_depth,
                    self.linear_mlp_dropout,
                )
            return

        shell = PatchTSTShell(configs)
        self.trunk = shell.model
        self.trunk.backbone = LinearPatchEncoder(configs)
        del shell

    def forward(self, x):
        # x: [Batch, seq_len, Channel]
        x = x.permute(0, 2, 1)
        if not self.use_patch_head:
            if self.individual:
                outs = []
                for i in range(self.channels):
                    outs.append(self.backbone_direct_mlps[i](x[:, i, :]))
                y = torch.stack(outs, dim=1)
            else:
                y = self.backbone_direct_mlp(x)
            return y.permute(0, 2, 1)

        out = self.trunk(x)
        if isinstance(out, tuple):
            y, aux = out
            return y.permute(0, 2, 1), aux
        return out.permute(0, 2, 1)

    def get_aux_loss(self):
        if hasattr(self, "trunk") and hasattr(self.trunk, "get_aux_loss"):
            return self.trunk.get_aux_loss()
        return torch.tensor(0.0, device=next(self.parameters()).device)
