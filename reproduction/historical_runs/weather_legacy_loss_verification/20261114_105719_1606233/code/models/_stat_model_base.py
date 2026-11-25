import torch
import torch.nn as nn

from models.PatchTST import Model as PatchTSTShell
from models._stat_backbones import (
    ARIMAPatchEncoder,
    ETSPatchEncoder,
    NBEATSPatchEncoder,
    ProphetPatchEncoder,
    ThetaPatchEncoder,
)


_ENCODER_REGISTRY = {
    "ARIMA": ARIMAPatchEncoder,
    "ETS": ETSPatchEncoder,
    "Theta": ThetaPatchEncoder,
    "Prophet": ProphetPatchEncoder,
    "NBEATS": NBEATSPatchEncoder,
}


class StatHeadModel(nn.Module):
    def __init__(self, configs, encoder_name):
        super().__init__()
        self.encoder_name = encoder_name
        self.head_type = getattr(configs, "head_type", "flatten")
        if self.head_type not in ("flatten", "sacq_cross_attention"):
            raise ValueError(
                f"{encoder_name} unsupported head_type={self.head_type!r}; "
                "use 'flatten' or 'sacq_cross_attention'."
            )

        shell = PatchTSTShell(configs)
        self.trunk = shell.model
        del shell

        encoder_cls = _ENCODER_REGISTRY[encoder_name]
        self.trunk.backbone = encoder_cls(configs)

    def forward(self, x):
        # x: [Batch, seq_len, Channel]
        out = self.trunk(x.permute(0, 2, 1))
        if isinstance(out, tuple):
            y, aux = out
            return y.permute(0, 2, 1), aux
        return out.permute(0, 2, 1)

    def get_aux_loss(self):
        if hasattr(self.trunk, "get_aux_loss"):
            return self.trunk.get_aux_loss()
        return torch.tensor(0.0, device=next(self.parameters()).device)

