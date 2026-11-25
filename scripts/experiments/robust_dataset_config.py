"""Shared config for SAR vs SACQ spike-robustness experiments."""
from __future__ import annotations

from argparse import Namespace

import pandas as pd


def spike_tag(spike_pct: int, noise_mult: float = 3.0) -> str:
    if noise_mult == 3.0:
        return f"spp{spike_pct}"
    mult = int(noise_mult) if noise_mult == int(noise_mult) else noise_mult
    return f"sp{mult}xp{spike_pct}"


def train_rows_for_csv(csv_path: str) -> int:
    df = pd.read_csv(csv_path)
    return int(len(df) * 0.7)


DATASETS = {
    "ETTh1": dict(
        data="ETTh1",
        data_path_clean="ETTh1.csv",
        enc_in=7,
        ot_ch=6,
        id_prefix="H1",
        d_model=16,
        n_heads=4,
        d_ff=128,
        dropout=0.3,
        fc_dropout=0.3,
        train_rows=8640,
        spike_csv=lambda pct, nm: (
            f"ETTh1_spike_{int(nm) if nm == int(nm) else nm}x_p{int(pct)}.csv"
        ),
    ),
    "weather": dict(
        data="weather",
        data_path_clean="weather.csv",
        enc_in=21,
        ot_ch=20,
        id_prefix="W",
        d_model=128,
        n_heads=16,
        d_ff=256,
        dropout=0.2,
        fc_dropout=0.2,
        train_rows=36887,
        spike_csv=lambda pct, nm: (
            f"weather_spike_{int(nm) if nm == int(nm) else nm}x_p{int(pct)}.csv"
        ),
    ),
}


def pair_spike_ds(
    dataset: str,
    spike_pct: int,
    noise_mult: float = 3.0,
    pred_len: int = 720,
) -> list:
    cfg = DATASETS[dataset]
    tag = spike_tag(spike_pct, noise_mult)
    mid = f"{cfg['id_prefix']}s512p{pred_len}"
    data = cfg["data"]
    return [
        (
            "SAR (spike-train)",
            f"{mid}SARlc_{tag}_PatchTST_{data}_sl512_pl{pred_len}_hSARlc_{tag}_0",
            dict(
                model_id=f"{mid}SARlc_{tag}",
                sacq_autoregressive=1,
                loss="logcosh",
                des=f"hSARlc_{tag}",
            ),
            "#2ca02c",
        ),
        (
            "SACQ (spike-train)",
            f"{mid}SACQslc_{tag}_PatchTST_{data}_sl512_pl{pred_len}_hSACQslc_{tag}_0",
            dict(
                model_id=f"{mid}SACQslc_{tag}",
                sacq_autoregressive=0,
                loss="logcosh_scaled",
                des=f"hSACQslc_{tag}",
            ),
            "#9467bd",
        ),
    ]


def exp_base(dataset: str, pred_len: int, **overrides) -> Namespace:
    cfg = DATASETS[dataset]
    base = dict(
        random_seed=2021,
        is_training=0,
        model="PatchTST",
        data=cfg["data"],
        root_path="./dataset/",
        data_path=cfg["data_path_clean"],
        scaler_data_path="",
        features="M",
        target="OT",
        freq="h",
        checkpoints="./checkpoints/",
        checkpoint_setting="",
        init_checkpoint_setting="",
        skip_visual=1,
        eval_tag="robust_eval",
        seq_len=512,
        label_len=48,
        pred_len=pred_len,
        fc_dropout=cfg["fc_dropout"],
        head_dropout=0.0,
        patch_len=16,
        stride=8,
        padding_patch="end",
        revin=1,
        affine=0,
        subtract_last=0,
        individual=0,
        decomposition=0,
        kernel_size=25,
        enc_in=cfg["enc_in"],
        dec_in=cfg["enc_in"],
        c_out=cfg["enc_in"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        e_layers=3,
        d_layers=1,
        d_ff=cfg["d_ff"],
        moving_avg=25,
        factor=1,
        distil=False,
        dropout=cfg["dropout"],
        embed="timeF",
        activation="gelu",
        num_workers=4,
        itr=1,
        train_epochs=50,
        batch_size=128,
        patience=20,
        learning_rate=1e-4,
        des="Exp",
        short_setting=1,
        drift_stat_loss_weight=0.0,
        sacq_gate_loss_weight=0.0,
        sacq_corr_sign_loss_weight=0.0,
        sacq_drift_alpha_enable=1,
        sacq_alpha_temperature=0.5,
        sacq_alpha_granularity="patch",
        sacq_alpha_softplus=0,
        sacq_shift_boost_loss_weight=0.0,
        lradj="TST",
        pct_start=0.2,
        use_amp=False,
        use_gpu=True,
        gpu=0,
        use_multi_gpu=False,
        devices="0",
        test_flop=False,
        loss="mse",
        logcosh_beta=1.0,
        logcosh_beta_mode="auto_mad",
        head_type="sacq_cross_attention",
        cross_attn_layers=2,
        sacq_scheme="default",
        sacq_pe_mode="default",
        sacq_autoregressive=0,
        sacq_alpha_mode="learnable",
        dsacq_enable=0,
        dsacq_mem_gate_enable=0,
        dsacq_drift_gate_enable=0,
    )
    base.update(overrides)
    return Namespace(**base)
