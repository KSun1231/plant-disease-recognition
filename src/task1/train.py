#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
try:  # support new/old AMP API, prefer torch.amp.autocast('cuda', ...)
    from torch.amp import autocast as _autocast_new  # type: ignore

    def _amp_autocast(enabled: bool):
        return _autocast_new('cuda', enabled=enabled)
except Exception:  # fallback to old API
    from torch.cuda.amp import autocast as _autocast_old  # type: ignore

    def _amp_autocast(enabled: bool):
        return _autocast_old(enabled=enabled)

# GradScaler compatibility (new/old)
try:
    from torch.amp import GradScaler as _GradScalerNew  # type: ignore

    def _create_grad_scaler(enabled: bool):
        return _GradScalerNew('cuda', enabled=enabled)
except Exception:
    from torch.cuda.amp import GradScaler as _GradScalerOld  # type: ignore

    def _create_grad_scaler(enabled: bool):
        return _GradScalerOld(enabled=enabled)

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

from src.dataloaders.build import DataConfig, build_dataloaders, compute_class_weights
from src.models.build_model import build_model
from src.utils.ema import ModelEma
from src.utils.metrics import top1_accuracy, f1_macro
from src.utils.logger import CsvLogger
from src.utils.saver import save_checkpoint, ensure_dir


def load_config(path: str | None) -> Dict[str, Any]:
    if path is None:
        return {}
    if yaml is None:
        raise RuntimeError('PyYAML not installed; cannot read YAML config')
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_scheduler(optimizer, total_epochs: int, warmup_epochs: int, base_lr: float, min_lr: float):
    def lr_lambda(curr_epoch: int):
        if curr_epoch < warmup_epochs:
            return max(1e-8, (curr_epoch + 1) / max(1, warmup_epochs))
        # cosine from base_lr to min_lr
        t = (curr_epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
        import math

        cos = 0.5 * (1 + math.cos(math.pi * t))
        scale = (min_lr / base_lr) + (1 - (min_lr / base_lr)) * cos
        return float(scale)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, mixup_fn=None, grad_clip: float | None = None, *, logit_adjust_bias: torch.Tensor | None = None):
    model.train()
    loss_meter = 0.0
    acc_meter = 0.0
    n = 0
    for xb, yb in loader:
        if xb.numel() == 0:
            continue
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        use_amp = scaler is not None
        optimizer.zero_grad(set_to_none=True)
        if mixup_fn is not None:
            xb, yb_mix = mixup_fn(xb, yb)
        with _amp_autocast(enabled=use_amp):
            logits = model(xb)
            if logit_adjust_bias is not None:
                logits = logits + logit_adjust_bias
            if mixup_fn is not None:
                loss = criterion(logits, yb_mix)
            else:
                loss = criterion(logits, yb)
        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        with torch.no_grad():
            loss_meter += float(loss.detach().item()) * xb.size(0)
            acc_meter += top1_accuracy(logits, yb) * xb.size(0)
            n += xb.size(0)

    return loss_meter / max(1, n), acc_meter / max(1, n)


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes: int, *, logit_adjust_bias: torch.Tensor | None = None):
    model.eval()
    loss_meter = 0.0
    acc_meter = 0.0
    n = 0
    all_logits = []
    all_targets = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        if logit_adjust_bias is not None:
            logits = logits + logit_adjust_bias
        loss = criterion(logits, yb)
        loss_meter += float(loss.detach().item()) * xb.size(0)
        acc_meter += top1_accuracy(logits, yb) * xb.size(0)
        n += xb.size(0)
        all_logits.append(logits.cpu())
        all_targets.append(yb.cpu())
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    f1m = f1_macro(logits, targets, num_classes)
    return loss_meter / max(1, n), acc_meter / max(1, n), f1m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/task1.yaml')
    ap.add_argument('--out-dir', default=None, help='override output directory in config')
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Dataset & DataLoader
    ds_cfg = cfg.get('dataset', {})
    data_cfg = DataConfig(
        root=ds_cfg.get('root', 'data/ProblemB-Data'),
        train_list=ds_cfg.get('train_list', 'datasets/train_list.txt'),
        val_list=ds_cfg.get('val_list', 'datasets/val_list.txt'),
        img_size=int(ds_cfg.get('img_size', 224)),
        resize_short=int(ds_cfg.get('resize_short', 256)),
        batch_size=int(ds_cfg.get('batch_size', 64)),
        num_workers=int(ds_cfg.get('num_workers', 8)),
        class_balancing=bool(ds_cfg.get('class_balancing', False)),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last_train=True,
        min_dim=16,
        strict=False,
        seed=int(cfg.get('train', {}).get('seed', 42)),
    )
    (ds_train, ds_val), (dl_train, dl_val), info = build_dataloaders(data_cfg)

    num_classes = len(info['train_class_counts']) if info['train_class_counts'] else 61

    # Model
    m_cfg = cfg.get('model', {})
    model_name = m_cfg.get('name', 'efficientnetv2_s')
    pretrained = bool(m_cfg.get('pretrained', True))
    drop_rate = float(m_cfg.get('drop_rate', 0.2))
    model = build_model(model_name=model_name, num_classes=num_classes, pretrained=pretrained, drop_rate=drop_rate)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Loss / Optim / Sched
    t_cfg = cfg.get('train', {})
    epochs = int(t_cfg.get('epochs', 100))
    base_lr = float(t_cfg.get('lr', 5e-4))
    weight_decay = float(t_cfg.get('weight_decay', 1e-4))
    label_smoothing = float(t_cfg.get('label_smoothing', 0.1))
    amp = bool(t_cfg.get('amp', True))
    ema_enable = bool(t_cfg.get('ema', True))
    grad_clip = float(t_cfg.get('grad_clip_norm', 1.0))
    warmup_epochs = int(cfg.get('scheduler', {}).get('warmup_epochs', 5))
    min_lr = float(cfg.get('scheduler', {}).get('min_lr', 1e-6))

    mixup_alpha = float(t_cfg.get('mixup_alpha', 0.0))
    cutmix_alpha = float(t_cfg.get('cutmix_alpha', 0.0))
    # class weighting
    class_weighting = str(t_cfg.get('class_weighting', 'none')).lower()
    class_weight_log_alpha = float(t_cfg.get('class_weight_log_alpha', 1.02))
    class_weight_normalize = bool(t_cfg.get('class_weight_normalize_mean', True))
    # DRW and Logit-Adjusted config
    drw_start_epoch = int(t_cfg.get('drw_start_epoch', 1))
    la_enable = bool(t_cfg.get('la_enable', False))
    la_tau = float(t_cfg.get('la_tau', 1.0))
    eval_la = bool(t_cfg.get('eval_la', la_enable))
    mixup_fn = None
    if (mixup_alpha > 1e-8 or cutmix_alpha > 1e-8):
        try:
            from timm.data.mixup import Mixup  # type: ignore
            from timm.loss import SoftTargetCrossEntropy  # type: ignore

            mixup_fn = Mixup(mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha, prob=1.0)
            if class_weighting in {'sqrt', 'log', 'inverse'}:
                print('[w] MixUp/CutMix enabled; class weighting will be ignored (SoftTargetCrossEntropy does not support weight)')
            criterion = SoftTargetCrossEntropy()
        except Exception:
            mixup_fn = None
            criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    else:
        # if class weighting needed, build weights from train distribution
        ce_weight = None
        if class_weighting in {'sqrt', 'log', 'inverse'}:
            try:
                labels = [lab for _, lab in ds_train.items]
                scheme = 'sqrt' if class_weighting == 'sqrt' else ('log' if class_weighting == 'log' else 'inverse')
                w_dict = compute_class_weights(labels, scheme=scheme, log_alpha=class_weight_log_alpha)
                w = torch.ones(num_classes, dtype=torch.float)
                for c, val in w_dict.items():
                    if 0 <= int(c) < num_classes:
                        w[int(c)] = float(val)
                ce_weight = w.to(device)
                if class_weighting != 'none':
                    print(f'[i] Using class weighting ({scheme}); mean weight≈{float(w.mean()):.3f}')
            except Exception as e:  # noqa: BLE001
                print('[w] Failed to compute class weights; falling back to none.', e)
                ce_weight = None
        # build both weighted/plain CE to support DRW
        criterion_weighted = nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=ce_weight)
        criterion_plain = nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=None)
        criterion = criterion_weighted if drw_start_epoch <= 1 else criterion_plain

    # compute LA bias (based on train class prior)
    la_bias = None
    if la_enable:
        try:
            from collections import Counter

            labels = [lab for _, lab in ds_train.items]
            cnt = Counter(labels)
            prior = torch.full((num_classes,), 1e-12, dtype=torch.float)
            for k, v in cnt.items():
                if 0 <= int(k) < num_classes:
                    prior[int(k)] = float(v)
            prior = prior / prior.sum()
            la_bias = (la_tau * torch.log(prior + 1e-12)).to(device)
            print('[i] Enable Logit-Adjusted, tau=', la_tau)
        except Exception as e:  # noqa: BLE001
            print('[w] Failed to compute LA bias; ignore.', e)
            la_bias = None

    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    scheduler = build_scheduler(optimizer, total_epochs=epochs, warmup_epochs=warmup_epochs, base_lr=base_lr, min_lr=min_lr)
    scaler = _create_grad_scaler(amp)

    ema = ModelEma(model, decay=0.9998, device=device) if ema_enable else None

    # Output & logger
    s_cfg = cfg.get('save', {})
    out_dir = args.out_dir or s_cfg.get('out_dir', 'outputs/task1_effv2s')
    ensure_dir(out_dir)
    logger = CsvLogger(os.path.join(out_dir, 'train_log.csv'), fieldnames=['epoch', 'lr', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_f1_macro', 'best_metric'])

    best_metric_name = s_cfg.get('save_best_metric', 'val_f1_macro')
    best_metric = -1.0

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        # DRW: after start epoch, switch to weighted CE (if available and no MixUp)
        if mixup_fn is None and 'criterion_weighted' in locals() and 'criterion_plain' in locals():
            if (epoch + 1) >= drw_start_epoch:
                criterion = criterion_weighted
            else:
                criterion = criterion_plain
        train_loss, train_acc = train_one_epoch(
            model, dl_train, criterion, optimizer, device, scaler, mixup_fn, grad_clip,
            logit_adjust_bias=(la_bias if la_enable else None),
        )
        if ema is not None:
            ema.update(model)

        if ema is not None:
            eval_model = ema.ema
        else:
            eval_model = model
        val_loss, val_acc, val_f1 = evaluate(
            eval_model, dl_val, criterion, device, num_classes,
            logit_adjust_bias=(la_bias if eval_la and la_enable else None),
        )

        scheduler.step()

        lr_curr = optimizer.param_groups[0]['lr']
        metric_curr = val_f1 if best_metric_name.lower().endswith('f1_macro') else val_acc
        if metric_curr > best_metric:
            best_metric = metric_curr
            save_checkpoint(os.path.join(out_dir, 'best.pth'), {
                'epoch': epoch + 1,
                'model': (ema.ema.state_dict() if ema is not None else model.state_dict()),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict() if scaler is not None else None,
                'config': cfg,
            })

        logger.log({
            'epoch': epoch + 1,
            'lr': lr_curr,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1_macro': val_f1,
            'best_metric': best_metric,
        })
        t1 = time.time()
        print(f"Epoch {epoch+1:03d}/{epochs} | lr={lr_curr:.6f} | train loss={train_loss:.4f} acc={train_acc:.4f} | val loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f} | best={best_metric:.4f} | {t1-t0:.1f}s")

    logger.close()
    print('Training finished. Best metric:', best_metric_name, best_metric)


if __name__ == '__main__':
    main()

