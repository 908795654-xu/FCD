# RevIN-derived layer with project extensions; ETRI (2022), MIT; see NOTICE.md and LICENSES/RevIN-MIT.txt.
import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps=1e-5,
        affine=True,
        subtract_last=False,
        affine_group_size: int = 0,
        max_steps: int = 0,
        affine_share_groups: str = "",
        conditional_affine: bool = False,
        conditional_affine_scale: float = 0.2,
        conditional_affine_hidden: int = 0,
    ):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.affine_group_size = int(affine_group_size)
        self.max_steps = int(max_steps)
        self.affine_share_groups = str(affine_share_groups or "").strip()
        if self.affine_group_size < 0:
            raise ValueError(f"affine_group_size must be >= 0, got {self.affine_group_size}.")
        if self.affine_group_size > 0 and self.max_steps <= 0:
            raise ValueError(
                f"max_steps must be > 0 when affine_group_size > 0, got {self.max_steps}."
            )
        self.grouped_affine = self.affine and self.affine_group_size > 0
        self.share_affine_by_channel_group = self.affine and bool(self.affine_share_groups)
        if bool(conditional_affine) and not self.share_affine_by_channel_group:
            raise ValueError(
                "RevIN conditional_affine requires affine_share_groups (group-shared affine base)."
            )
        self.conditional_affine = self.affine and bool(conditional_affine)
        self.conditional_affine_scale = float(conditional_affine_scale)
        hid = int(conditional_affine_hidden)
        self._group_index = None
        self.cond_affine_mlp = None
        self._runtime_affine_weight = None
        self._runtime_affine_bias = None
        if self.share_affine_by_channel_group:
            self._group_index = self._build_group_index(self.affine_share_groups)
        if self.affine:
            self._init_params()
        if self.conditional_affine:
            n_groups = int(torch.max(self._group_index).item()) + 1
            in_dim = 2 * n_groups
            hidden = hid if hid > 0 else max(32, 4 * n_groups)
            self.cond_affine_mlp = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, in_dim),
            )

    def _build_group_index(self, spec: str):
        groups = []
        for block in spec.split(";"):
            block = block.strip()
            if not block:
                continue
            ids = [int(x.strip()) for x in block.split(",") if x.strip()]
            if not ids:
                continue
            groups.append(ids)
        if not groups:
            raise ValueError("affine_share_groups is provided but parsed empty groups.")
        seen = set()
        group_index = torch.empty(self.num_features, dtype=torch.long)
        for gi, g in enumerate(groups):
            for idx in g:
                if idx < 0 or idx >= self.num_features:
                    raise ValueError(
                        f"affine_share_groups index out of range: {idx} (num_features={self.num_features})."
                    )
                if idx in seen:
                    raise ValueError(f"affine_share_groups has duplicate index: {idx}.")
                seen.add(idx)
                group_index[idx] = gi
        missing = sorted(set(range(self.num_features)) - seen)
        if missing:
            raise ValueError(
                f"affine_share_groups must cover all channels exactly once; missing {missing}."
            )
        return group_index

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        if self.share_affine_by_channel_group:
            n_groups = int(torch.max(self._group_index).item()) + 1
            self.affine_weight = nn.Parameter(torch.ones(n_groups))
            self.affine_bias = nn.Parameter(torch.zeros(n_groups))
        elif self.grouped_affine:
            max_groups = (self.max_steps + self.affine_group_size - 1) // self.affine_group_size
            self.affine_weight = nn.Parameter(torch.ones(max_groups, self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(max_groups, self.num_features))
        else:
            # initialize RevIN params: (C,)
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _resolve_affine_params(self, x):
        if self.share_affine_by_channel_group:
            if x.ndim != 3:
                raise ValueError(
                    f"group-shared RevIN affine expects x.ndim == 3 [B, T, C], got {x.ndim}."
                )
            group_index = self._group_index.to(device=x.device)
            weight = self.affine_weight[group_index]
            bias = self.affine_bias[group_index]
            return weight.view(1, 1, -1), bias.view(1, 1, -1)
        if not self.grouped_affine:
            return self.affine_weight, self.affine_bias
        if x.ndim != 3:
            raise ValueError(
                f"grouped RevIN affine currently expects x.ndim == 3 [B, T, C], got {x.ndim}."
            )
        steps = x.size(1)
        if steps > self.max_steps:
            raise ValueError(
                f"input steps {steps} exceed configured max_steps {self.max_steps} for grouped affine."
            )
        group_count = self.affine_weight.size(0)
        if group_count == 1:
            step_weight = self.affine_weight.expand(steps, -1)
            step_bias = self.affine_bias.expand(steps, -1)
            return step_weight.unsqueeze(0), step_bias.unsqueeze(0)

        device = x.device
        # Soft grouping: each time step smoothly interpolates two adjacent groups.
        # group_pos in [0, +inf): t=0 -> group 0, t=N -> group 1, etc.
        group_pos = torch.arange(steps, device=device, dtype=torch.float32) / float(self.affine_group_size)
        left_idx = torch.floor(group_pos).long().clamp(max=group_count - 1)
        right_idx = (left_idx + 1).clamp(max=group_count - 1)

        right_w = (group_pos - left_idx.to(group_pos.dtype)).unsqueeze(-1)  # [T, 1]
        left_w = 1.0 - right_w

        left_weight = self.affine_weight[left_idx]   # [T, C]
        right_weight = self.affine_weight[right_idx] # [T, C]
        left_bias = self.affine_bias[left_idx]       # [T, C]
        right_bias = self.affine_bias[right_idx]     # [T, C]

        step_weight = left_w * left_weight + right_w * right_weight
        step_bias = left_w * left_bias + right_w * right_bias
        return step_weight.unsqueeze(0), step_bias.unsqueeze(0)

    def _conditional_group_affine_params(self, x):
        B, _, _ = x.shape
        G = self.affine_weight.shape[0]
        device = x.device
        gi = self._group_index.to(device)
        base_mean = self.last if self.subtract_last else self.mean
        parts = []
        for g in range(G):
            idx = torch.where(gi == g)[0]
            m_g = base_mean[:, :, idx].mean(dim=-1, keepdim=True)
            s_g = self.stdev[:, :, idx].mean(dim=-1, keepdim=True)
            parts.append(m_g)
            parts.append(s_g)
        h = torch.cat(parts, dim=-1).squeeze(1)
        delta = self.cond_affine_mlp(h)
        d_w, d_b = delta[:, :G], delta[:, G:]
        sc = self.conditional_affine_scale
        tw = torch.tanh(d_w) * sc
        tb = torch.tanh(d_b) * sc
        w_g = self.affine_weight.unsqueeze(0) * (1.0 + tw)
        b_g = self.affine_bias.unsqueeze(0) + tb
        gather_idx = gi.view(1, -1).expand(B, -1)
        w_ch = torch.gather(w_g, 1, gather_idx)
        b_ch = torch.gather(b_g, 1, gather_idx)
        return w_ch.unsqueeze(1), b_ch.unsqueeze(1)

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            if self.conditional_affine:
                affine_weight, affine_bias = self._conditional_group_affine_params(x)
                self._runtime_affine_weight = affine_weight
                self._runtime_affine_bias = affine_bias
                x = x * affine_weight + affine_bias
            else:
                self._runtime_affine_weight = None
                self._runtime_affine_bias = None
                affine_weight, affine_bias = self._resolve_affine_params(x)
                x = x * affine_weight + affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            if self.conditional_affine and self._runtime_affine_weight is not None:
                affine_weight = self._runtime_affine_weight
                affine_bias = self._runtime_affine_bias
            else:
                affine_weight, affine_bias = self._resolve_affine_params(x)
            x = x - affine_bias
            x = x / (affine_weight + self.eps * self.eps)
        stdev = self.stdev
        x = x * stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x