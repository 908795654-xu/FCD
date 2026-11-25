from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import DLinear, Linear, PatchTST, PatchTSTMamba
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler

import os
import time

import warnings

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'DLinear': DLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'PatchTSTMamba': PatchTSTMamba,
            'MambaPatchTST': PatchTSTMamba,
        }
        model = model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    @staticmethod
    def _unwrap_model(model):
        return model.module if isinstance(model, nn.DataParallel) else model

    def _bootstrap_flatten_into_sacq(self, ckpt_sd=None):
        """Copy Flatten_Head weights into SACQ query_flatten_head (individual or shared)."""
        model = self._unwrap_model(self.model)
        head = getattr(model, 'model', model)
        if not hasattr(head, 'head') or not hasattr(head.head, 'query_flatten_head'):
            return
        qfh = head.head.query_flatten_head
        src_sd = ckpt_sd if ckpt_sd is not None else model.state_dict()
        prefixes = ('model.head.', 'head.')
        copied = 0

        def copy_linear(dst_linear, src_key_w, src_key_b):
            nonlocal copied
            if src_key_w in src_sd and src_sd[src_key_w].shape == dst_linear.weight.shape:
                dst_linear.weight.data.copy_(src_sd[src_key_w])
                copied += 1
            if src_key_b in src_sd and dst_linear.bias is not None and src_sd[src_key_b].shape == dst_linear.bias.shape:
                dst_linear.bias.data.copy_(src_sd[src_key_b])
                copied += 1

        if qfh.individual:
            for i, lin in enumerate(qfh.linears):
                for pfx in prefixes:
                    copy_linear(lin, f'{pfx}linears.{i}.weight', f'{pfx}linears.{i}.bias')
        else:
            for pfx in prefixes:
                copy_linear(qfh.linear, f'{pfx}linear.weight', f'{pfx}linear.bias')
        if copied:
            print(f'bootstrap_flatten_head: copied {copied} tensors into query_flatten_head')

    def _apply_train_freeze(self):
        freeze_bb = bool(int(getattr(self.args, 'freeze_backbone', 0)))
        freeze_coarse = bool(int(getattr(self.args, 'freeze_coarse_head', 0)))
        if not freeze_bb and not freeze_coarse:
            return
        model = self._unwrap_model(self.model)
        n_bb = n_coarse = 0
        for name, param in model.named_parameters():
            if freeze_bb and not name.startswith('model.head.'):
                if param.requires_grad:
                    n_bb += 1
                param.requires_grad = False
            if freeze_coarse and name.startswith('model.head.query_flatten_head'):
                if param.requires_grad:
                    n_coarse += 1
                param.requires_grad = False
        print(f'train freeze: backbone={freeze_bb} ({n_bb} tensors), coarse_head={freeze_coarse} ({n_coarse} tensors)')

    @staticmethod
    def _is_sacq_alpha_param(param_name: str) -> bool:
        if not param_name.startswith('model.head.'):
            return False
        tail = param_name[len('model.head.') :]
        return (
            'alpha_proj' in tail
            or 'alpha_feat_norm' in tail
            or tail in ('fusion_logit', 'fusion_gain', 'global_alpha_logit')
        )

    def _select_optimizer(self):
        model = self._unwrap_model(self.model)
        self._apply_train_freeze()
        lr = float(self.args.learning_rate)
        head_type = getattr(self.args, 'head_type', 'flatten')
        coarse_ratio = float(getattr(self.args, 'sacq_coarse_lr_ratio', 1.0))
        decoder_ratio = float(getattr(self.args, 'sacq_decoder_lr_ratio', 1.0))
        backbone_ratio = float(getattr(self.args, 'sacq_backbone_lr_ratio', 1.0))
        alpha_ratio = float(getattr(self.args, 'sacq_alpha_lr_ratio', 1.0))
        use_sacq_groups = (
            head_type in ('sacq_cross_attention', 'dsacq_cross_attention')
            and hasattr(model, 'model')
            and hasattr(model.model, 'head')
            and hasattr(model.model.head, 'query_flatten_head')
        )
        if not use_sacq_groups:
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            return optim.Adam(trainable, lr=lr)

        coarse_params, alpha_params, decoder_params, backbone_params = [], [], [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith('model.head.query_flatten_head'):
                coarse_params.append(param)
            elif self._is_sacq_alpha_param(name):
                alpha_params.append(param)
            elif name.startswith('model.head.'):
                decoder_params.append(param)
            else:
                backbone_params.append(param)

        if coarse_params or alpha_params or decoder_params or backbone_params:
            param_groups = []
            max_lrs = []
            if coarse_params:
                param_groups.append(
                    {'params': coarse_params, 'lr': lr * coarse_ratio, 'lr_scale': coarse_ratio}
                )
                max_lrs.append(lr * coarse_ratio)
            if decoder_params:
                param_groups.append(
                    {'params': decoder_params, 'lr': lr * decoder_ratio, 'lr_scale': decoder_ratio}
                )
                max_lrs.append(lr * decoder_ratio)
            if alpha_params:
                param_groups.append(
                    {'params': alpha_params, 'lr': lr * alpha_ratio, 'lr_scale': alpha_ratio}
                )
                max_lrs.append(lr * alpha_ratio)
            if backbone_params:
                param_groups.append(
                    {'params': backbone_params, 'lr': lr * backbone_ratio, 'lr_scale': backbone_ratio}
                )
                max_lrs.append(lr * backbone_ratio)
            print(
                'SACQ optimizer groups: '
                f'coarse={len(coarse_params)} decoder={len(decoder_params)} '
                f'alpha={len(alpha_params)} backbone={len(backbone_params)} '
                f'(lr ratios coarse={coarse_ratio}, decoder={decoder_ratio}, '
                f'alpha={alpha_ratio}, backbone={backbone_ratio})'
            )
            self._optimizer_max_lrs = max_lrs
            return optim.Adam(param_groups)

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError('No trainable parameters after freeze settings.')
        print(f'SACQ optimizer: single group, {len(trainable)} parameter tensors')
        self._optimizer_max_lrs = lr
        return optim.Adam(trainable, lr=lr)

    def _select_criterion(self):
        from utils.losses import build_training_loss

        loss_name = getattr(self.args, "loss", "mse")
        return build_training_loss(
            loss_name,
            beta=getattr(self.args, "logcosh_beta", 1.0),
            beta_mode=getattr(self.args, "logcosh_beta_mode", "auto_mad"),
            irse_iters=getattr(self.args, "logcosh_irse_iters", 3),
            beta_ema_momentum=getattr(self.args, "logcosh_beta_ema_momentum", 0.01),
        )

    def _split_model_output(self, model_out):
        if isinstance(model_out, tuple):
            pred, aux = model_out
            return pred, aux if aux is not None else {}
        return model_out, {}

    def _forward(self, batch_x, batch_x_mark=None, batch_y_mark=None):
        if getattr(self.args, 'model', '') == 'PatchTST':
            return self.model(batch_x, batch_x_mark=batch_x_mark, batch_y_mark=batch_y_mark)
        if 'Linear' in self.args.model or 'TST' in self.args.model or 'Mamba' in self.args.model:
            return self.model(batch_x)
        raise ValueError(f"Unsupported model: {self.args.model}")

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        total_mse = []
        self.model.eval()
        if isinstance(criterion, nn.Module):
            criterion.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                outputs, _ = self._split_model_output(
                    self._forward(batch_x, batch_x_mark=batch_x_mark, batch_y_mark=batch_y_mark)
                )
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())
                total_mse.append(float(F.mse_loss(outputs, batch_y).item()))
        self.model.train()
        if isinstance(criterion, nn.Module):
            criterion.train()
        return np.average(total_loss), np.average(total_mse)

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        if isinstance(criterion, nn.Module):
            criterion = criterion.to(self.device)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        max_lr = getattr(self, '_optimizer_max_lrs', None) or self.args.learning_rate
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=max_lr,
        )

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_mse_batches = []
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, _ = self._split_model_output(
                            self._forward(batch_x, batch_x_mark=batch_x_mark, batch_y_mark=batch_y_mark)
                        )
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                        loss = criterion(outputs, batch_y)
                else:
                    outputs, _ = self._split_model_output(
                        self._forward(batch_x, batch_x_mark=batch_x_mark, batch_y_mark=batch_y_mark)
                    )
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                    loss = criterion(outputs, batch_y)

                train_loss.append(loss.item())
                train_mse_batches.append(float(F.mse_loss(outputs, batch_y).item()))

                if (i + 1) % 100 == 0:
                    print(
                        f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}"
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.4f}")
            train_loss = np.average(train_loss)
            train_mse = float(np.average(train_mse_batches)) if train_mse_batches else float('nan')
            vali_loss, vali_mse = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_mse = self.vali(test_data, test_loader, criterion)
            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} "
                f"Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f} | "
                f"MSE Train {train_mse:.7f} Vali {vali_mse:.7f} Test {test_mse:.7f}"
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print(f'Updating learning rate to {scheduler.get_last_lr()[0]}')

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    @staticmethod
    def _to_btc(x, ref_btc):
        """Normalize tensor shape to [B, T, C] when possible."""
        if x is None or not torch.is_tensor(x):
            return None
        if x.dim() != 3:
            return None
        b, t, c = ref_btc.shape
        if x.shape == (b, t, c):
            return x
        if x.shape == (b, c, t):
            return x.permute(0, 2, 1)
        return None

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        ckpt_setting = getattr(self.args, 'checkpoint_setting', '') or setting
        if ckpt_setting != setting:
            print(f'load checkpoint from training run: {ckpt_setting}')
            print(f'evaluate on dataset: {self.args.data} (setting={setting})')

        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join(self.args.checkpoints, ckpt_setting, 'checkpoint.pth'))
            )

        preds = []
        trues = []
        inputx = []
        alphas_all = []
        has_alpha = False
        folder_path = os.path.join('./test_results/', setting)
        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                outputs, aux = self._split_model_output(
                    self._forward(batch_x, batch_x_mark=batch_x_mark, batch_y_mark=batch_y_mark)
                )
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                alpha_t = aux.get('sacq_alpha_seq') if aux else None
                if alpha_t is not None:
                    has_alpha = True
                    alpha_btc = self._to_btc(alpha_t, batch_y)
                    if alpha_btc is None:
                        alpha_btc = (
                            alpha_t.permute(0, 2, 1)
                            if alpha_t.shape[1] == batch_y.shape[2]
                            else alpha_t
                        )
                    alphas_all.append(alpha_btc.detach().cpu().numpy())

                if not getattr(self.args, 'skip_visual', 0) and i % 20 == 0:
                    inp = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((inp[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((inp[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1], batch_x.shape[2]))
            return

        preds = np.array(preds).reshape(-1, preds[0].shape[-2], preds[0].shape[-1])
        trues = np.array(trues).reshape(-1, trues[0].shape[-2], trues[0].shape[-1])
        inputx = np.array(inputx).reshape(-1, inputx[0].shape[-2], inputx[0].shape[-1])

        folder_path = os.path.join('./results/', setting)
        os.makedirs(folder_path, exist_ok=True)
        mae, mse, rmse, mape, mspe, rse, corr, smape, mase, ql50, pwmae = metric(preds, trues, inputx=inputx)
        print(
            f'mse:{mse}, mae:{mae}, rse:{rse}, rmse:{rmse}, mape:{mape}, mspe:{mspe}, corr:{corr}, '
            f'smape:{smape}, mase:{mase}, ql50:{ql50}, pwmae:{pwmae}'
        )
        eval_tag = getattr(self.args, 'eval_tag', '') or ''
        with open('result.txt', 'a') as f:
            header = setting
            if eval_tag:
                header = f'[{eval_tag}] {setting}'
            f.write(header + "  \n")
            f.write(
                f'mse:{mse}, mae:{mae}, rse:{rse}, rmse:{rmse}, mape:{mape}, mspe:{mspe}, corr:{corr}, '
                f'smape:{smape}, mase:{mase}, ql50:{ql50}, pwmae:{pwmae}\n\n'
            )
        cross_path = 'cross_dataset_results.txt'
        if eval_tag:
            with open(cross_path, 'a') as f:
                f.write(f'{eval_tag}\tckpt={ckpt_setting}\t{setting}\n')
                f.write(
                    f'mse:{mse}, mae:{mae}, rse:{rse}, rmse:{rmse}, mape:{mape}, mspe:{mspe}, corr:{corr}, '
                    f'smape:{smape}, mase:{mase}, ql50:{ql50}, pwmae:{pwmae}\n\n'
                )

        np.save(os.path.join(folder_path, 'pred.npy'), preds)
        np.save(os.path.join(folder_path, 'true.npy'), trues)
        np.save(os.path.join(folder_path, 'x.npy'), inputx)
        if has_alpha and alphas_all:
            alphas_all = np.array(alphas_all).reshape(-1, alphas_all[0].shape[-2], alphas_all[0].shape[-1])
            np.save(os.path.join(folder_path, 'alpha.npy'), alphas_all)
