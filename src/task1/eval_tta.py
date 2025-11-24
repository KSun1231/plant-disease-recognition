#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn

from src.dataloaders.build import DataConfig, build_dataloaders
from src.models.build_model import build_model
from src.utils.metrics import top1_accuracy, f1_macro


@torch.no_grad()
def evaluate_tta(model, loader, device, num_classes: int, hflip: bool = True):
    model.eval()
    all_logits = []
    all_targets = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        if hflip:
            logits_flip = model(torch.flip(xb, dims=[3]))
            logits = (logits + logits_flip) * 0.5
        all_logits.append(logits.cpu())
        all_targets.append(yb.cpu())
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    acc = top1_accuracy(logits, targets)
    f1m = f1_macro(logits, targets, num_classes)
    return acc, f1m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True, help='path to best.pth')
    ap.add_argument('--root', default='data/ProblemB-Data')
    ap.add_argument('--val-list', default='datasets/val_list.txt')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--img-size', type=int, default=224)
    ap.add_argument('--model', default='efficientnetv2_s')
    ap.add_argument('--num-classes', type=int, default=61)
    args = ap.parse_args()

    cfg = DataConfig(
        root=args.root,
        train_list='datasets/train_list.txt',
        val_list=args.val_list,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.workers,
        class_balancing=False,
    )
    (_, ds_val), (_, dl_val), info = (None, None), (None, None), None
    (_, ds_val), (dl_train, dl_val), info = build_dataloaders(cfg)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(model_name=args.model, num_classes=args.num_classes, pretrained=False, drop_rate=0.0)
    state = torch.load(args.weights, map_location='cpu')
    model.load_state_dict(state['model'], strict=False)
    model.to(device)

    acc, f1m = evaluate_tta(model, dl_val, device, args.num_classes)
    print(f'TTA validation: acc={acc:.4f} f1_macro={f1m:.4f}')


if __name__ == '__main__':
    main()

