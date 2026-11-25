import argparse
import random
import sys

import numpy as np
import torch

from exp.exp_main import Exp_Main


parser = argparse.ArgumentParser(description='SACQ / PatchTST long-term forecasting')

parser.add_argument('--random_seed', type=int, default=2021, help='random seed')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument(
    '--model',
    type=str,
    required=True,
    default='PatchTST',
    help='model name: PatchTST, PatchTSTMamba, DLinear, Linear, ARIMA, ETS, Theta, Prophet, NBEATS',
)

parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument(
    '--scaler_data_path',
    type=str,
    default='',
    help='optional CSV for scaler fit only (train segment); defaults to data_path',
)
parser.add_argument(
    '--target_data_path',
    type=str,
    default='',
    help='optional CSV for prediction targets only; defaults to data_path',
)
parser.add_argument(
    '--features',
    type=str,
    default='M',
    help='M: multivariate predict multivariate; MS: multivariate predict univariate',
)
parser.add_argument('--target', type=str, default='OT', help='target feature for MS task')
parser.add_argument(
    '--freq',
    type=str,
    default='h',
    help='time feature frequency: s,t,h,d,b,w,m or finer like 15min',
)
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='checkpoint directory')
parser.add_argument(
    '--checkpoint_setting',
    type=str,
    default='',
    help='checkpoint folder name (training setting); empty = use current setting',
)
parser.add_argument(
    '--init_checkpoint_setting',
    type=str,
    default='',
    help='optional checkpoint folder name to initialize training weights',
)
parser.add_argument('--skip_visual', type=int, default=0, choices=[0, 1], help='skip test PDF plots')
parser.add_argument(
    '--eval_tag',
    type=str,
    default='',
    help='tag for cross-dataset eval lines in result file (e.g. ETTh1->ETTh2)',
)

parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length (unused by PatchTST)')
parser.add_argument('--pred_len', type=int, default=96, help='prediction length')

parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='patch stride')
parser.add_argument(
    '--head_type',
    type=str,
    default='flatten',
    choices=['flatten', 'sacq_cross_attention', 'dsacq_cross_attention'],
    help='prediction head type (dsacq_cross_attention = SACQ + level/trend coarse split)',
)
parser.add_argument(
    '--dsacq_coarse_split',
    type=int,
    default=1,
    choices=[0, 1],
    help='DSACQ head: 1=level+trend+residual coarse branches; 0=original SACQ coarse',
)
parser.add_argument(
    '--dsacq_decomp_kernel',
    type=int,
    default=25,
    help='DSACQ trend/residual moving-average kernel (default: --kernel_size, DLinear series_decomp=25)',
)
parser.add_argument(
    '--dsacq_level_slope_k',
    type=int,
    default=25,
    help='DSACQ level slope history window (default: --kernel_size, aligned with DLinear decomp)',
)
parser.add_argument('--cross_attn_layers', type=int, default=2, help='SACQ cross-attention depth')
parser.add_argument(
    '--sacq_mem_film_enable',
    type=int,
    default=0,
    choices=[0, 1],
    help='enable coarse-conditioned FiLM modulation on history memory tokens',
)
parser.add_argument(
    '--sacq_mem_film_hidden',
    type=int,
    default=64,
    help='hidden size for coarse-conditioned memory FiLM MLP',
)
parser.add_argument(
    '--sacq_mem_film_scale',
    type=float,
    default=0.25,
    help='max absolute FiLM scale/shift factor via tanh bounding',
)
parser.add_argument(
    '--sacq_autoregressive',
    type=int,
    default=0,
    choices=[0, 1],
    help='SACQ head: causal self-attention over future patch queries (alias for sacq_future_causal)',
)
parser.add_argument(
    '--sacq_future_causal',
    type=int,
    default=0,
    choices=[0, 1],
    help='Causal mask on future-patch self-attention (prevents future-to-future leakage)',
)
parser.add_argument(
    '--sacq_ar_verify',
    type=int,
    default=0,
    choices=[0, 1],
    help='Light AR branch: feed causal fused patch beliefs into later decoder queries',
)
parser.add_argument(
    '--sacq_coarse_init',
    type=str,
    default='flatten',
    choices=['flatten', 'routing', 'routing_blend'],
    help='Coarse scaffold F: flatten | history routing | blend with flatten',
)
parser.add_argument(
    '--sacq_routing_topk',
    type=int,
    default=0,
    help='routing coarse init: top-k history keys per future patch (0 = full softmax)',
)
parser.add_argument(
    '--sacq_routing_window',
    type=int,
    default=0,
    help='routing coarse init: local history window radius (0 = disabled)',
)
parser.add_argument(
    '--sacq_scheme',
    type=str,
    default='default',
    choices=['default', 'dual_residual', 'template_ar'],
    help='SACQ head variant: default | dual_residual (scheme1) | template_ar (scheme6)',
)
parser.add_argument(
    '--sacq_dual_struct_gate',
    type=int,
    default=0,
    choices=[0, 1],
    help='dual_residual: learnable gate on struct branch',
)
parser.add_argument(
    '--sacq_dual_res_fuse',
    type=int,
    default=0,
    choices=[0, 1],
    help='dual_residual: blend-fuse query with scaffold embedding (see res_fuse_logit)',
)
parser.add_argument(
    '--sacq_dual_detach',
    type=int,
    default=1,
    choices=[0, 1],
    help='dual_residual: stop-gradient on scaffold before residual branch',
)
parser.add_argument(
    '--sacq_dual_scaffold_ln',
    type=int,
    default=0,
    choices=[0, 1],
    help='dual_residual: LayerNorm on scaffold patches before residual branch',
)
parser.add_argument(
    '--sacq_pe_mode',
    type=str,
    default='default',
    choices=['default', 'shared', 'continuation', 'relative_bias'],
    help='SACQ positional encoding: default | shared (6) | continuation (4) | relative_bias (2)',
)
parser.add_argument(
    '--sacq_pe_rel_buckets',
    type=int,
    default=32,
    help='relative_bias: number of time-lag buckets for cross-attention bias',
)
parser.add_argument(
    '--sacq_time_token_enable',
    type=int,
    default=0,
    choices=[0, 1],
    help='inject patch-level time feature embedding into encoder tokens',
)
parser.add_argument(
    '--sacq_time_token_hidden',
    type=int,
    default=64,
    help='hidden size of time-token MLP when sacq_time_token_enable=1',
)
parser.add_argument(
    '--sacq_time_token_dropout',
    type=float,
    default=0.0,
    help='dropout on time-token embedding before encoder attention',
)
parser.add_argument(
    '--sacq_time_token_pool',
    type=str,
    default='last',
    choices=['last', 'mean'],
    help='pooling over per-patch time marks: last step or mean',
)
parser.add_argument(
    '--sacq_time_token_gate_init',
    type=float,
    default=0.05,
    help="initial gate value g for time-token injection: u' = u + g * time_emb",
)
parser.add_argument('--padding_patch', default='end', help='None or end')
parser.add_argument('--revin', type=int, default=1, help='RevIN on/off')
parser.add_argument('--affine', type=int, default=0, help='RevIN affine on/off')
parser.add_argument('--subtract_last', type=int, default=0, help='RevIN: 0 subtract mean, 1 subtract last')
parser.add_argument('--sacq_overlap_avg', type=int, default=0, choices=[0, 1], help='overlap patch merge for SACQ')
parser.add_argument(
    '--sacq_alpha_mode',
    type=str,
    default='learnable',
    choices=['learnable', 'fixed'],
    help='SACQ gate mode',
)
parser.add_argument('--sacq_alpha_value', type=float, default=0.5, help='initial/fixed SACQ gate')
parser.add_argument(
    '--sacq_coarse_lr_ratio',
    type=float,
    default=1.0,
    help='stage-2 SACQ: LR multiplier for query_flatten_head vs base learning_rate',
)
parser.add_argument(
    '--sacq_decoder_lr_ratio',
    type=float,
    default=1.0,
    help='SACQ: LR multiplier for cross-attention decoder head parameters',
)
parser.add_argument(
    '--sacq_backbone_lr_ratio',
    type=float,
    default=0.3,
    help='stage-2 SACQ: LR multiplier for encoder backbone (non-head) parameters',
)
parser.add_argument(
    '--sacq_alpha_lr_ratio',
    type=float,
    default=1.0,
    help='SACQ: LR multiplier for alpha gate (alpha_proj / alpha_feat_norm) vs decoder',
)
parser.add_argument(
    '--bootstrap_flatten_head',
    type=int,
    default=0,
    choices=[0, 1],
    help='after init_checkpoint load, copy flatten head weights into SACQ query_flatten_head',
)
parser.add_argument(
    '--freeze_backbone',
    type=int,
    default=0,
    choices=[0, 1],
    help='freeze encoder backbone (all non-head parameters)',
)
parser.add_argument(
    '--freeze_coarse_head',
    type=int,
    default=0,
    choices=[0, 1],
    help='freeze SACQ query_flatten_head (coarse flatten branch)',
)
parser.add_argument('--dsacq_enable', type=int, default=1, choices=[0, 1], help='master switch for DSACQ modules')
parser.add_argument('--dsacq_mem_gate_enable', type=int, default=1, choices=[0, 1], help='distribution-aware memory gate (requires dsacq_enable=1)')
parser.add_argument('--dsacq_drift_gate_enable', type=int, default=1, choices=[0, 1], help='drift-adaptive residual gate (requires dsacq_enable=1)')
parser.add_argument('--dsacq_mem_gate_hidden', type=int, default=32, help='hidden dim of DSACQ memory gate MLP')
parser.add_argument(
    '--dsacq_mem_gate_mode',
    type=str,
    default='stats',
    choices=['stats', 'e2e', 'align'],
    help='DSACQ memory gate: stats, e2e, or align (cross-domain)',
)
parser.add_argument('--mem_shift_aug_std', type=float, default=0.15, help='shift augmentation std for align gate (train only)')
parser.add_argument('--mem_shift_aug_prob', type=float, default=0.5, help='probability of shift augmentation per batch')
parser.add_argument(
    '--mem_coral_loss_weight',
    type=float,
    default=0.05,
    help='CORAL aux loss weight between memory and query tokens (align mode)',
)
parser.add_argument('--align_ablate_shift', type=int, default=0, choices=[0, 1], help='align ablation: disable shift encoding')
parser.add_argument('--align_ablate_dual_bank', type=int, default=0, choices=[0, 1], help='align ablation: disable dual memory bank')
parser.add_argument('--align_ablate_token_gate', type=int, default=0, choices=[0, 1], help='align ablation: disable query-aware token gate')
parser.add_argument('--align_ablate_shift_drift', type=int, default=0, choices=[0, 1], help='align ablation: use heuristic drift gate')
parser.add_argument('--decomposition', type=int, default=0, help='series decomposition for PatchTST')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition kernel size')
parser.add_argument('--individual', type=int, default=0, help='per-channel head')

parser.add_argument('--enc_in', type=int, default=7, help='encoder input size / channel count')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='model dimension')
parser.add_argument('--n_heads', type=int, default=8, help='attention heads')
parser.add_argument('--e_layers', type=int, default=2, help='encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='decoder layers (unused)')
parser.add_argument('--d_ff', type=int, default=2048, help='FFN dimension')
parser.add_argument('--moving_avg', type=int, default=25, help='moving average window')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_true', help='legacy flag kept for run naming')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF', help='time features: timeF, fixed, learned')
parser.add_argument('--activation', type=str, default='gelu', help='activation')

parser.add_argument('--num_workers', type=int, default=10, help='data loader workers')
parser.add_argument('--itr', type=int, default=1, help='experiment repeats')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('--des', type=str, default='test', help='experiment description')
parser.add_argument(
    '--short_setting',
    type=int,
    default=0,
    choices=[0, 1],
    help='use compact run naming to avoid too-long paths',
)
parser.add_argument(
    '--loss',
    type=str,
    default='mse',
    choices=['mse', 'logcosh', 'logcosh_scaled'],
    help='training loss: mse | logcosh | logcosh_scaled (evaluation metrics still use MSE)',
)
parser.add_argument(
    '--logcosh_beta',
    type=float,
    default=1.0,
    help='Scaled LogCosh transition scale beta (>0). Only used when loss=logcosh_scaled.',
)
parser.add_argument(
    '--logcosh_beta_mode',
    type=str,
    default='auto_mad',
    choices=['auto_mad', 'irse', 'fixed'],
    help=(
        'Scaled LogCosh beta mode: auto_mad/irse (IRSE during training, EMA beta at eval) '
        'or fixed. Only used when loss=logcosh_scaled.'
    ),
)
parser.add_argument(
    '--logcosh_irse_iters',
    type=int,
    default=3,
    help='IRSE refinement iterations per training batch (2-3 recommended).',
)
parser.add_argument(
    '--logcosh_beta_ema_momentum',
    type=float,
    default=0.01,
    help='EMA momentum for training beta; used as fixed scale at eval/inference.',
)
parser.add_argument(
    '--drift_stat_loss_weight',
    type=float,
    default=0.0,
    help='aux loss weight for matching prediction/target mean-std statistics',
)
parser.add_argument(
    '--sacq_gate_loss_weight',
    type=float,
    default=0.0,
    help='aux loss weight to supervise SACQ alpha with target-aware proxy gate',
)
parser.add_argument(
    '--sacq_corr_sign_loss_weight',
    type=float,
    default=0.0,
    help='aux loss weight to enforce correction branch direction consistency',
)
parser.add_argument(
    '--sacq_drift_alpha_enable',
    type=int,
    default=1,
    help='use drift/misalign features in basic SACQ alpha head (independent of DSACQ)',
)
parser.add_argument(
    '--sacq_alpha_temperature',
    type=float,
    default=0.5,
    help='temperature for SACQ alpha sigmoid; lower => sharper gate response',
)
parser.add_argument(
    '--sacq_alpha_granularity',
    type=str,
    default='patch',
    choices=['patch', 'step', 'global'],
    help='SACQ gate resolution: per patch, per timestep in patch, or one scalar for all patches',
)
parser.add_argument(
    '--sacq_alpha_softplus',
    type=int,
    default=0,
    choices=[0, 1],
    help='0: α=sigmoid(·) in (0,1); 1: α=softplus(·) >= 0 (no upper bound)',
)
parser.add_argument(
    '--sacq_shift_boost_loss_weight',
    type=float,
    default=0.0,
    help='aux loss to raise alpha under high input drift (shift sensitivity)',
)
parser.add_argument(
    '--sacq_grad_firewall_enable',
    type=int,
    default=0,
    choices=[0, 1],
    help='enable SACQ gradient firewall gate on KV memory (detach-mix)',
)
parser.add_argument(
    '--sacq_grad_firewall_hidden',
    type=int,
    default=64,
    help='hidden dim of SACQ gradient firewall MLP',
)
parser.add_argument(
    '--sacq_grad_firewall_temp',
    type=float,
    default=1.0,
    help='temperature for SACQ gradient firewall sigmoid gate',
)
parser.add_argument(
    '--sacq_grad_firewall_mode',
    type=str,
    default='learnable',
    choices=['learnable', 'fixed_stat'],
    help='gradient firewall gate mode: learnable MLP or fixed statistic function',
)
parser.add_argument(
    '--sacq_grad_firewall_sup_weight',
    type=float,
    default=0.0,
    help='supervision loss weight for firewall gate using paired noise mask',
)
parser.add_argument(
    '--sacq_grad_firewall_sparse_weight',
    type=float,
    default=0.0,
    help='sparsity regularization weight on firewall gate mean',
)
parser.add_argument('--lradj', type=str, default='type3', help='learning rate schedule')
parser.add_argument('--pct_start', type=float, default=0.3, help='OneCycleLR pct_start when lradj=TST')
parser.add_argument('--use_amp', action='store_true', help='automatic mixed precision', default=False)

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu id')
parser.add_argument('--use_multi_gpu', action='store_true', help='multi gpu', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='multi-gpu device ids')
parser.add_argument('--test_flop', action='store_true', default=False, help='profile FLOPs and exit')
parser.add_argument(
    '--paired_noise_enable',
    type=int,
    default=0,
    choices=[0, 1],
    help='train with 50/50 paired clean+noisy mini-batches',
)
parser.add_argument(
    '--paired_noise_sigma',
    type=float,
    default=7.0,
    help='outlier scale for noisy half-batch (single random input timestep): +-sigma*std',
)
parser.add_argument(
    '--grad_probe_enable',
    type=int,
    default=0,
    choices=[0, 1],
    help='enable hook-based gradient probe on noisy-only vs total gradients',
)
parser.add_argument(
    '--grad_probe_epoch',
    type=int,
    default=5,
    help='epoch index (1-based) used for gradient probe',
)
parser.add_argument(
    '--grad_probe_max_batches',
    type=int,
    default=0,
    help='max batches to probe at probe epoch (0 means all batches)',
)

args = parser.parse_args()


def _cli_flag_present(flag: str) -> bool:
    return any(a == flag or a.startswith(flag + '=') for a in sys.argv[1:])


if not _cli_flag_present('--dsacq_decomp_kernel'):
    args.dsacq_decomp_kernel = args.kernel_size
if not _cli_flag_present('--dsacq_level_slope_k'):
    args.dsacq_level_slope_k = args.kernel_size

fix_seed = args.random_seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.is_training:
    for ii in range(args.itr):
        if args.short_setting:
            setting = (
                f'{args.model_id}_{args.model}_{args.data}_'
                f'sl{args.seq_len}_pl{args.pred_len}_{args.des}_{ii}'
            )
        else:
            setting = (
                f'{args.model_id}_{args.model}_{args.data}_ft{args.features}_'
                f'sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_'
                f'dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_'
                f'df{args.d_ff}_fc{args.factor}_eb{args.embed}_dt{args.distil}_{args.des}_{ii}'
            )
        exp = Exp(args)
        print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
        exp.train(setting)
        print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test(setting)
        torch.cuda.empty_cache()
else:
    ii = 0
    if args.short_setting:
        setting = (
            f'{args.model_id}_{args.model}_{args.data}_'
            f'sl{args.seq_len}_pl{args.pred_len}_{args.des}_{ii}'
        )
    else:
        setting = (
            f'{args.model_id}_{args.model}_{args.data}_ft{args.features}_'
            f'sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_'
            f'dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_'
            f'df{args.d_ff}_fc{args.factor}_eb{args.embed}_dt{args.distil}_{args.des}_{ii}'
        )
    exp = Exp(args)
    print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
