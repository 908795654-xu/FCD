__all__ = ["PatchTSTMamba"]

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from layers.Mamba_backbone import MambaTSTiEncoder
from layers.PatchTST_layers import series_decomp
from models.PatchTST import Model as PatchTSTShell


class Model(nn.Module):
    def __init__(
        self,
        configs,
        max_seq_len: Optional[int] = 1024,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        act: str = "gelu",
        key_padding_mask: bool = "auto",
        padding_var: Optional[int] = None,
        attn_mask: Optional[Tensor] = None,
        res_attention: bool = True,
        pre_norm: bool = False,
        store_attn: bool = False,
        pe: str = "zeros",
        learn_pe: bool = True,
        pretrain_head: bool = False,
        head_type: str = "flatten",
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__()

        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len

        n_layers = configs.e_layers
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size
        head_type = kwargs.pop("head_type", getattr(configs, "head_type", head_type))

        self.decomposition = decomposition

        def _swap_patchtst_encoder_for_mamba(bb):
            old_enc = bb.backbone
            bb.backbone = MambaTSTiEncoder(
                c_in=c_in,
                patch_num=old_enc.patch_num,
                patch_len=old_enc.patch_len,
                max_seq_len=max_seq_len,
                n_layers=n_layers,
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout,
                pe=pe,
                learn_pe=learn_pe,
                verbose=verbose,
            )
            del old_enc

        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            shell_res = PatchTSTShell(
                configs,
                max_seq_len=max_seq_len,
                d_k=d_k,
                d_v=d_v,
                norm=norm,
                attn_dropout=attn_dropout,
                act=act,
                key_padding_mask=key_padding_mask,
                padding_var=padding_var,
                attn_mask=attn_mask,
                res_attention=res_attention,
                pre_norm=pre_norm,
                store_attn=store_attn,
                pe=pe,
                learn_pe=learn_pe,
                pretrain_head=pretrain_head,
                head_type=head_type,
                verbose=verbose,
                **kwargs,
            )
            self.model_res = shell_res.model
            _swap_patchtst_encoder_for_mamba(self.model_res)
            del shell_res

            shell_trend = PatchTSTShell(
                configs,
                max_seq_len=max_seq_len,
                d_k=d_k,
                d_v=d_v,
                norm=norm,
                attn_dropout=attn_dropout,
                act=act,
                key_padding_mask=key_padding_mask,
                padding_var=padding_var,
                attn_mask=attn_mask,
                res_attention=res_attention,
                pre_norm=pre_norm,
                store_attn=store_attn,
                pe=pe,
                learn_pe=learn_pe,
                pretrain_head=pretrain_head,
                head_type=head_type,
                verbose=verbose,
                **kwargs,
            )
            self.model_trend = shell_trend.model
            _swap_patchtst_encoder_for_mamba(self.model_trend)
            del shell_trend
        else:
            shell = PatchTSTShell(
                configs,
                max_seq_len=max_seq_len,
                d_k=d_k,
                d_v=d_v,
                norm=norm,
                attn_dropout=attn_dropout,
                act=act,
                key_padding_mask=key_padding_mask,
                padding_var=padding_var,
                attn_mask=attn_mask,
                res_attention=res_attention,
                pre_norm=pre_norm,
                store_attn=store_attn,
                pe=pe,
                learn_pe=learn_pe,
                pretrain_head=pretrain_head,
                head_type=head_type,
                verbose=verbose,
                **kwargs,
            )
            self.model = shell.model
            _swap_patchtst_encoder_for_mamba(self.model)
            del shell

    def forward(self, x):
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            if isinstance(res, tuple):
                res = res[0]
            if isinstance(trend, tuple):
                trend = trend[0]
            x = res + trend
            x = x.permute(0, 2, 1)
        else:
            x = x.permute(0, 2, 1)
            out = self.model(x)
            if isinstance(out, tuple):
                x, aux = out
                x = x.permute(0, 2, 1)
                return x, aux
            x = out.permute(0, 2, 1)
        return x

    def get_aux_loss(self):
        if self.decomposition:
            loss = 0.0
            if hasattr(self.model_trend, "get_aux_loss"):
                loss = loss + self.model_trend.get_aux_loss()
            if hasattr(self.model_res, "get_aux_loss"):
                loss = loss + self.model_res.get_aux_loss()
            if isinstance(loss, torch.Tensor):
                return loss
            return torch.tensor(0.0, device=next(self.parameters()).device)
        if hasattr(self.model, "get_aux_loss"):
            return self.model.get_aux_loss()
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def set_gs_assign_temp(self, temp: float):
        if self.decomposition:
            if hasattr(self.model_trend, "set_gs_assign_temp"):
                self.model_trend.set_gs_assign_temp(temp)
            if hasattr(self.model_res, "set_gs_assign_temp"):
                self.model_res.set_gs_assign_temp(temp)
            return
        if hasattr(self.model, "set_gs_assign_temp"):
            self.model.set_gs_assign_temp(temp)

    def set_gs_interaction_epoch(self, epoch_1based: int):
        if self.decomposition:
            if hasattr(self.model_trend, "set_gs_interaction_epoch"):
                self.model_trend.set_gs_interaction_epoch(epoch_1based)
            if hasattr(self.model_res, "set_gs_interaction_epoch"):
                self.model_res.set_gs_interaction_epoch(epoch_1based)
            return
        if hasattr(self.model, "set_gs_interaction_epoch"):
            self.model.set_gs_interaction_epoch(epoch_1based)
