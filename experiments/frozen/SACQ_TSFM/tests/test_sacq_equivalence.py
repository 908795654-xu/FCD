"""Equivalence test for the standalone SACQ extraction."""

from pathlib import Path
import sys

import torch

ORIGINAL_ROOT = Path("/home/bkai/CG2/SACQ")
if str(ORIGINAL_ROOT) not in sys.path:
    sys.path.insert(0, str(ORIGINAL_ROOT))

from layers.PatchTST_backbone import (
    SACQCrossAttentionHead as OriginalSACQHead,
)
from sacq_core.head import (
    SACQCrossAttentionHead as StandaloneSACQHead,
)


def build_kwargs():
    batch_size = 2
    channels = 3
    hidden_dim = 16
    history_patches = 8
    prediction_length = 24
    future_patch_length = 8

    kwargs = dict(
        n_vars=channels,
        nf=hidden_dim * history_patches,
        d_model=hidden_dim,
        target_window=prediction_length,
        patch_len=future_patch_length,
        cross_attn_layers=2,
        n_heads=4,
        head_dropout=0.0,
        individual=False,
        query_pos_emb=None,
        return_attention_map=False,
        patch_stride=None,
        overlap_avg=False,
        sacq_alpha_mode="learnable",
        sacq_alpha_value=0.5,
        sacq_drift_alpha_enable=True,
        sacq_alpha_temperature=0.5,
        sacq_alpha_granularity="patch",
        sacq_alpha_softplus=False,
        sacq_autoregressive=False,
        sacq_future_causal=False,
        sacq_pe_mode="default",
        hist_patch_num=history_patches,

        # Pure SACQ: these must always be explicit.
        dsacq_enable=False,
        dsacq_mem_gate_enable=False,
        dsacq_drift_gate_enable=False,
    )
    shape = (batch_size, channels, hidden_dim, history_patches)
    return kwargs, shape


def assert_tensor_equal(name, actual, expected):
    torch.testing.assert_close(
        actual,
        expected,
        rtol=0.0,
        atol=0.0,
        msg=lambda msg: f"{name} differs:\n{msg}",
    )


def test_original_and_standalone_are_identical():
    torch.manual_seed(2026)
    kwargs, input_shape = build_kwargs()

    original = OriginalSACQHead(**kwargs).cpu().eval()
    standalone = StandaloneSACQHead(**kwargs).cpu().eval()

    standalone.load_state_dict(original.state_dict(), strict=True)

    torch.manual_seed(2027)
    x = torch.randn(*input_shape)

    with torch.no_grad():
        original_y, original_aux = original(x)
        standalone_y, standalone_aux = standalone(x)

    assert original_y.shape == (2, 3, 24)
    assert standalone_y.shape == original_y.shape
    assert_tensor_equal("prediction", standalone_y, original_y)

    assert standalone_aux.keys() == original_aux.keys()
    for key in original_aux:
        assert_tensor_equal(
            f"aux[{key!r}]",
            standalone_aux[key],
            original_aux[key],
        )

    original_params = sum(p.numel() for p in original.parameters())
    standalone_params = sum(p.numel() for p in standalone.parameters())
    assert standalone_params == original_params

    assert original.dsacq_enable is False
    assert standalone.dsacq_enable is False
    assert standalone.dsacq_mem_gate is False
    assert standalone.dsacq_drift_gate is False

    print("input_shape:", tuple(x.shape))
    print("output_shape:", tuple(standalone_y.shape))
    print("aux_keys:", sorted(standalone_aux))
    print("parameter_count:", standalone_params)
    print("max_abs_diff:", (standalone_y - original_y).abs().max().item())
    print("EQUIVALENCE_OK")


if __name__ == "__main__":
    test_original_and_standalone_are_identical()
