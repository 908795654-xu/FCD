__all__ = ["MambaPatchTST_backbone"]

from typing import Optional
import numpy as np
import torch
from torch import nn, Tensor
from layers.PatchTST_layers import positional_encoding
from layers.PatchTST_backbone import Flatten_Head, SACQCrossAttentionHead
from layers.RevIN import RevIN


class MambaPatchTST_backbone(nn.Module):
    def __init__(
        self,
        c_in: int,
        context_window: int,
        target_window: int,
        patch_len: int,
        stride: int,
        max_seq_len: Optional[int] = 1024,
        n_layers: int = 3,
        cross_attn_layers: int = 1,
        d_model: int = 128,
        n_heads: int = 16,
        d_ff: int = 256,
        dropout: float = 0.0,
        pe: str = "zeros",
        learn_pe: bool = True,
        fc_dropout: float = 0.0,
        head_dropout: float = 0.0,
        padding_patch=None,
        pretrain_head: bool = False,
        head_type: str = "flatten",
        individual: bool = False,
        revin: bool = True,
        affine: bool = True,
        subtract_last: bool = False,
        affine_group_size: int = 0,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(
                c_in,
                affine=affine,
                subtract_last=subtract_last,
                affine_group_size=affine_group_size,
                max_steps=max(context_window, target_window),
            )

        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.target_window = target_window
        patch_num = int((context_window - patch_len) / stride + 1)
        if padding_patch == "end":
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1
        self.out_patch_num = int(np.ceil(target_window / patch_len))

        self.backbone = MambaTSTiEncoder(
            c_in=c_in,
            patch_num=patch_num,
            patch_len=patch_len,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            pe=pe,
            learn_pe=learn_pe,
            verbose=verbose,
            **kwargs,
        )

        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual
        self.sacq_overlap_avg = bool(kwargs.pop("sacq_overlap_avg", False))
        self.save_attention_map = bool(kwargs.pop("save_attention_map", False))
        self.sacq_alpha_mode = kwargs.pop("sacq_alpha_mode", "learnable")
        self.sacq_alpha_value = float(kwargs.pop("sacq_alpha_value", 0.5))

        self.xatt_query_pos_emb = positional_encoding(pe, learn_pe, patch_num + target_window, d_model)

        if self.pretrain_head:
            self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout)
        elif head_type == "flatten":
            self.head = Flatten_Head(
                self.individual,
                self.n_vars,
                self.head_nf,
                target_window,
                head_dropout=head_dropout,
            )
        elif head_type == "sacq_cross_attention":
            self.head = SACQCrossAttentionHead(
                n_vars=self.n_vars,
                nf=self.head_nf,
                d_model=d_model,
                target_window=target_window,
                patch_len=patch_len,
                cross_attn_layers=cross_attn_layers,
                n_heads=n_heads,
                head_dropout=head_dropout,
                individual=self.individual,
                query_pos_emb=self.xatt_query_pos_emb,
                return_attention_map=self.save_attention_map,
                patch_stride=self.stride,
                overlap_avg=self.sacq_overlap_avg,
                sacq_alpha_mode=self.sacq_alpha_mode,
                sacq_alpha_value=self.sacq_alpha_value,
            )
        else:
            raise ValueError(
                f"Unsupported head_type: {head_type}. "
                "Mamba backbone currently supports ['flatten', 'sacq_cross_attention']."
            )

    def _encode(self, z):
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, "norm")
            z = z.permute(0, 2, 1)

        if self.padding_patch == "end":
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z = z.permute(0, 1, 3, 2)

        z = self.backbone(z)
        return z

    def _decode(self, z):
        decoded = self.head(z)
        if isinstance(decoded, tuple):
            z, aux = decoded
            return z, aux
        return decoded, None

    def _denorm(self, z):
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, "denorm")
            z = z.permute(0, 2, 1)
        return z

    def forward(self, z):
        enc_feat = self._encode(z)
        z, aux = self._decode(enc_feat)
        z = self._denorm(z)
        if aux is None:
            return z
        return z, aux

    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout), nn.Conv1d(head_nf, vars, 1))


class MambaTSTiEncoder(nn.Module):
    def __init__(
        self,
        c_in,
        patch_num,
        patch_len,
        max_seq_len=1024,
        n_layers=3,
        d_model=128,
        d_ff=256,
        dropout=0.0,
        pe="zeros",
        learn_pe=True,
        verbose=False,
        **kwargs,
    ):
        super().__init__()
        self.patch_num = patch_num
        self.patch_len = patch_len

        self.W_P = nn.Linear(patch_len, d_model)
        self.W_pos = positional_encoding(pe, learn_pe, patch_num, d_model)
        self.dropout = nn.Dropout(dropout)
        self.encoder = MambaEncoder(d_model=d_model, d_ff=d_ff, dropout=dropout, n_layers=n_layers)

    def forward(self, x) -> Tensor:
        n_vars = x.shape[1]
        x = x.permute(0, 1, 3, 2)
        x = self.W_P(x)
        u = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        u = self.dropout(u + self.W_pos)
        z = self.encoder(u)
        z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))
        z = z.permute(0, 1, 3, 2)
        return z


class MambaEncoder(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [MambaBlock(d_model=d_model, d_ff=d_ff, dropout=dropout) for _ in range(n_layers)]
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float, expand: int = 2, conv_kernel: int = 3):
        super().__init__()
        inner_dim = d_model * expand
        self.in_norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, inner_dim * 2)
        self.dw_conv = nn.Conv1d(
            in_channels=inner_dim,
            out_channels=inner_dim,
            kernel_size=conv_kernel,
            groups=inner_dim,
            padding=conv_kernel - 1,
        )
        self.dt_proj = nn.Linear(inner_dim, inner_dim)
        self.out_proj = nn.Linear(inner_dim, d_model)
        self.dropout = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        u = self.in_norm(x)
        u = self.in_proj(u)
        x_proj, gate = torch.chunk(u, 2, dim=-1)

        # Mamba-style token mixing: depthwise conv + selective cumsum scan.
        x_proj = self.dw_conv(x_proj.transpose(1, 2)).transpose(1, 2)
        x_proj = x_proj[:, : x.shape[1], :]
        delta = torch.sigmoid(self.dt_proj(x_proj))
        scanned = torch.cumsum(delta * x_proj, dim=1)
        mixed = scanned * torch.sigmoid(gate)
        x = residual + self.dropout(self.out_proj(mixed))

        x = x + self.ffn(self.ffn_norm(x))
        return x
