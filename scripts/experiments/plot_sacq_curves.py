"""Plot ground truth, prediction, and SACQ alpha on the test set."""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from exp.exp_main import Exp_Main

DEFAULT_PYTHON = "/home/bkai/miniconda3/envs/forecast/bin/python"

# Best alpha-boost checkpoint per pred_len (α×2)
DEFAULT_SETTINGS = {
    96: "TwS2Zc001A20s512p96_PatchTST_ETTh2_sl512_pl96_hA20_0",
    192: "TwS2Zc01A20s512p192_PatchTST_ETTh2_sl512_pl192_hA20_0",
    336: "TwS2Zc05A20s512p336_PatchTST_ETTh2_sl512_pl336_hA20_0",
    720: "TwS2Zc01A20s512p720_PatchTST_ETTh2_sl512_pl720_hA20_0",
}


def make_args(pred_len, setting, gpu=0):
    from argparse import Namespace

    return Namespace(
        random_seed=2021,
        is_training=0,
        model_id=setting.split("_")[0],
        model="PatchTST",
        data="ETTh2",
        root_path="./dataset/",
        data_path="ETTh2.csv",
        features="M",
        target="OT",
        freq="h",
        checkpoints="./checkpoints/",
        checkpoint_setting="",
        init_checkpoint_setting="",
        skip_visual=1,
        eval_tag="",
        seq_len=512,
        label_len=48,
        pred_len=pred_len,
        fc_dropout=0.3,
        head_dropout=0.0,
        patch_len=16,
        stride=8,
        head_type="sacq_cross_attention",
        cross_attn_layers=2,
        sacq_autoregressive=0,
        sacq_scheme="default",
        sacq_dual_struct_gate=0,
        sacq_dual_res_fuse=0,
        sacq_dual_detach=1,
        sacq_dual_scaffold_ln=0,
        sacq_pe_mode="relative_bias",
        sacq_pe_rel_buckets=32,
        padding_patch="end",
        revin=1,
        affine=0,
        subtract_last=0,
        sacq_overlap_avg=0,
        sacq_alpha_mode="learnable",
        sacq_alpha_granularity="step" if "A01gSt" in setting else "patch",
        sacq_alpha_value=0.1 if "A01g" in setting else 0.0001,
        sacq_coarse_lr_ratio=0.1,
        sacq_backbone_lr_ratio=0.0,
        sacq_alpha_lr_ratio=2.0,
        bootstrap_flatten_head=0,
        freeze_backbone=1,
        freeze_coarse_head=0,
        dsacq_enable=0,
        dsacq_mem_gate_enable=0,
        dsacq_drift_gate_enable=0,
        dsacq_mem_gate_hidden=32,
        dsacq_mem_gate_mode="stats",
        mem_shift_aug_std=0.15,
        mem_shift_aug_prob=0.5,
        sacq_alpha_temperature=0.5,
        sacq_gate_loss_weight=0.3 if "A01g" in setting else 0.0,
        sacq_corr_sign_loss_weight=0.15 if "A01g" in setting else 0.0,
        sacq_shift_boost_loss_weight=0.0,
        drift_stat_loss_weight=0.0,
        mem_coral_loss_weight=0.0,
        align_ablate_shift=0,
        align_ablate_dual_bank=0,
        align_ablate_token_gate=0,
        align_ablate_shift_drift=0,
        lradj="TST",
        pct_start=0.2,
        use_amp=False,
        use_gpu=True,
        gpu=gpu,
        use_multi_gpu=False,
        devices="0",
        device_ids=[0],
        test_flop=False,
        enc_in=7,
        dec_in=7,
        c_out=7,
        e_layers=3,
        d_layers=1,
        n_heads=4,
        d_model=16,
        d_ff=128,
        dropout=0.3,
        factor=1,
        embed="timeF",
        distil=True,
        individual=0,
        num_workers=4,
        batch_size=128,
        learning_rate=1e-4,
        train_epochs=1,
        patience=10,
        itr=1,
        des="plot",
        short_setting=1,
        decomposition=0,
        kernel_size=25,
        sacq_drift_alpha_enable=1,
        save_attention_map=0,
    )


def collect_series(exp, max_batches=None):
    _, test_loader = exp._get_data(flag="test")
    preds, trues, alphas, inputs = [], [], [], []
    f_dim = -1 if exp.args.features == "MS" else 0

    exp.model.eval()
    with torch.no_grad():
        for bi, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            if max_batches is not None and bi >= max_batches:
                break
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            outputs, aux = exp._split_model_output(exp._forward(batch_x))
            outputs = outputs[:, -exp.args.pred_len :, f_dim:]
            batch_y = batch_y[:, -exp.args.pred_len :, f_dim:]

            pred = outputs.detach().cpu().numpy()
            true = batch_y.detach().cpu().numpy()
            inp = batch_x.detach().cpu().numpy()
            preds.append(pred)
            trues.append(true)
            inputs.append(inp)

            alpha_t = aux.get("sacq_alpha_seq") if aux else None
            if alpha_t is not None:
                alpha_btc = exp._to_btc(alpha_t, batch_y)
                if alpha_btc is None:
                    if alpha_t.shape[1] == batch_y.shape[2]:
                        alpha_btc = alpha_t.permute(0, 2, 1)
                    else:
                        alpha_btc = alpha_t
                alphas.append(alpha_btc.detach().cpu().numpy())
            else:
                alphas.append(np.zeros_like(pred))

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    inputs = np.concatenate(inputs, axis=0)
    alphas = np.concatenate(alphas, axis=0)
    return inputs, preds, trues, alphas


def alpha_to_patch(alpha_fut, patch_len):
    """SACQ α is constant within each patch; collapse to one value per patch."""
    n_patch = len(alpha_fut) // patch_len
    return np.array([float(alpha_fut[i * patch_len]) for i in range(n_patch)])


def find_high_alpha_sample(alphas, channel, min_std=0.0):
    stds = alphas[:, :, channel].std(axis=1)
    idx = int(stds.argmax())
    if stds[idx] < min_std:
        return 0, float(stds[idx])
    return idx, float(stds[idx])


def plot_one_sample(
    inp_hist,
    true_fut,
    pred_fut,
    alpha_fut,
    out_path,
    title,
    channel,
    patch_len=16,
):
    """inp_hist: [seq_len], true/pred/alpha: [pred_len]."""
    seq_len = len(inp_hist)
    pred_len = len(true_fut)
    t_in = np.arange(seq_len)
    t_fut = np.arange(seq_len, seq_len + pred_len)
    alpha_patch = alpha_to_patch(alpha_fut, patch_len)
    n_patch = len(alpha_patch)
    t_patch = np.linspace(seq_len, seq_len + pred_len, n_patch, endpoint=False) + (
        (pred_len / n_patch) / 2.0
    )

    a_min, a_max = float(alpha_fut.min()), float(alpha_fut.max())
    a_mean = float(alpha_fut.mean())

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(12, 6), gridspec_kw={"height_ratios": [2.2, 1.0]}, sharex=False
    )
    ax_top.plot(t_in, inp_hist, color="#888888", linewidth=1.0, alpha=0.85, label="history")
    ax_top.plot(t_fut, true_fut, color="#1f77b4", linewidth=1.8, label="ground truth")
    ax_top.plot(t_fut, pred_fut, color="#ff7f0e", linewidth=1.8, linestyle="--", label="prediction")
    ax_top.axvline(seq_len - 0.5, color="black", linestyle=":", linewidth=1.0, alpha=0.5)
    ax_top.set_ylabel("value")
    ax_top.set_title(title + f"\nα∈[{a_min:.2e}, {a_max:.2e}], mean={a_mean:.2e} (patch-wise gate)")
    ax_top.legend(loc="upper left", fontsize=9)

    # Per-timestep α (step-wise constant inside each patch)
    ax_mid = ax_top.twinx()
    ax_mid.plot(t_fut, alpha_fut, color="#2ca02c", linewidth=1.0, alpha=0.55, drawstyle="steps-mid")
    ax_mid.set_ylabel(r"$\alpha$ (per step)", color="#2ca02c")
    if a_max > a_min * 1.01:
        margin = (a_max - a_min) * 0.15 + 1e-12
        ax_mid.set_ylim(a_min - margin, a_max + margin)
    else:
        ax_mid.set_ylim(0, max(a_max * 1.5, 1e-7))
    ax_mid.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax_mid.tick_params(axis="y", labelcolor="#2ca02c", labelsize=8)

    # Patch-level α (native resolution of the gate)
    ax_bot.step(
        range(n_patch),
        alpha_patch,
        where="mid",
        color="#2ca02c",
        linewidth=1.8,
        label=r"$\alpha$ per future patch",
    )
    ax_bot.set_xlabel(f"future patch index (patch_len={patch_len})")
    ax_bot.set_ylabel(r"$\alpha$")
    if a_max > a_min * 1.01:
        margin = (a_max - a_min) * 0.15 + 1e-12
        ax_bot.set_ylim(a_min - margin, a_max + margin)
    else:
        ax_bot.set_ylim(0, max(a_max * 1.5, 1e-7))
    ax_bot.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax_bot.grid(True, alpha=0.3)
    ax_bot.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_alpha_heatmap(alphas, out_path, title, n_show=200, patch_len=16):
    """Patch-level α heatmap [N, n_patch]."""
    n_show = min(n_show, len(alphas))
    n_patch = alphas.shape[1] // patch_len
    a = np.stack([alpha_to_patch(alphas[i].mean(axis=-1), patch_len) for i in range(n_show)])
    vmax = float(np.percentile(a, 99)) if a.size else 1.0
    vmax = max(vmax, 1e-12)
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(a, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
    ax.set_xlabel("forecast step")
    ax.set_ylabel("test sample index")
    ax.set_title(title + " (α mean over channels)")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_setting(setting, pred_len, out_dir, sample_idx, channel, gpu, out_basename=None):
    ckpt = os.path.join(PROJECT_ROOT, "checkpoints", setting, "checkpoint.pth")
    if not os.path.isfile(ckpt):
        print(f"skip missing ckpt: {ckpt}")
        return None

    args = make_args(pred_len, setting, gpu=gpu)
    exp = Exp_Main(args)
    exp.model.load_state_dict(torch.load(ckpt, map_location=exp.device))

    inputs, preds, trues, alphas = collect_series(exp)
    os.makedirs(out_dir, exist_ok=True)
    np.savez(
        os.path.join(out_dir, f"pl{pred_len}_series.npz"),
        inputs=inputs,
        preds=preds,
        trues=trues,
        alphas=alphas,
    )

    ch = min(channel, preds.shape[-1] - 1)
    if sample_idx < 0:
        si, a_std = find_high_alpha_sample(alphas, ch)
        sample_note = f"auto sample={si} (max α std={a_std:.2e})"
    else:
        si = min(sample_idx, len(preds) - 1)
        sample_note = f"sample={si}"
    f_dim = ch
    inp_ch = inputs[si, :, f_dim]
    base = out_basename or f"pl{pred_len}_sample{si}_ch{ch}"
    curve_path = os.path.join(out_dir, f"{base}.png")
    plot_one_sample(
        inp_ch,
        trues[si, :, ch],
        preds[si, :, ch],
        alphas[si, :, ch],
        curve_path,
        f"ETTh2 pl={pred_len} | {setting}\n({sample_note}, channel={ch})",
        ch,
        patch_len=args.patch_len,
    )
    heat_suffix = out_basename or f"pl{pred_len}"
    plot_alpha_heatmap(
        alphas,
        os.path.join(out_dir, f"{heat_suffix}_alpha_heatmap.png"),
        f"ETTh2 pl={pred_len}",
        patch_len=args.patch_len,
    )
    print(f"saved curves -> {curve_path}")
    return {"mse": float(np.mean((preds - trues) ** 2))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="logs/etth2_alpha_lr_boost/curves")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=-1,
        help="test sample index; -1 = pick sample with largest α std",
    )
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--pred_lens", type=str, default="96,192,336,720")
    parser.add_argument("--setting", type=str, default="", help="override single setting")
    parser.add_argument("--pred_len", type=int, default=96)
    args = parser.parse_args()

    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    if args.setting:
        run_setting(args.setting, args.pred_len, out_dir, args.sample_idx, args.channel, args.gpu)
        return

    for pl in [int(x) for x in args.pred_lens.split(",") if x.strip()]:
        setting = DEFAULT_SETTINGS.get(pl)
        if setting:
            run_setting(setting, pl, out_dir, args.sample_idx, args.channel, args.gpu)


if __name__ == "__main__":
    main()
