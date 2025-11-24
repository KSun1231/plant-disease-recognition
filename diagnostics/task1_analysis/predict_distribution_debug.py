#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Validation prediction distribution diagnostic (no CLI args).

Purpose:
- Load weights and run a val forward pass; compute prediction histogram and per-class accuracy.
- Also print: classes present in val, missing classes, overall Acc/F1 (F1 computed only on present classes).
- Quickly judge whether the model collapses to a few classes or missing classes affect evaluation.

Per AGENTS, parameters are at the top; run directly.
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from src.dataloaders.build import DataConfig, build_dataloaders
from src.models.build_model import build_model
from src.utils.metrics import top1_accuracy


# ===== Global parameters (edit as needed) =====
DATA_ROOT = 'data/ProblemB-Data'
VAL_LIST = 'datasets/val_list_rebalanced.txt'  # or switch to datasets/val_list.txt
MODEL_NAME = 'efficientnetv2_s'
NUM_CLASSES = 61
WEIGHTS_PATH = 'outputs/task1_effv2s/best.pth'
BATCH_SIZE = 128
NUM_WORKERS = -1  # -1 means use all CPU cores
IMG_SIZE = 224
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


@torch.no_grad()
def evaluate_distribution():
    cfg = DataConfig(
        root=DATA_ROOT,
        train_list='datasets/train_list.txt',  # unused here, placeholder
        val_list=VAL_LIST,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        class_balancing=False,
    )
    (_, _), (_, dl_val), info = build_dataloaders(cfg)

    model = build_model(model_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=False, drop_rate=0.0)
    if not os.path.isfile(WEIGHTS_PATH):
        raise FileNotFoundError(f'Weights not found: {WEIGHTS_PATH}')
    state = torch.load(WEIGHTS_PATH, map_location='cpu')
    sd = state.get('model', state)
    model.load_state_dict(sd, strict=False)
    model.to(DEVICE)
    model.eval()

    all_logits: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    for xb, yb in dl_val:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        logits = model(xb)
        all_logits.append(logits.cpu())
        all_targets.append(yb.cpu())
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # Macro-F1 computed only on classes present in val (avoid depressing F1 due to missing classes)
    try:
        from sklearn.metrics import f1_score

        present = sorted(set(int(x) for x in targets.tolist()))
        f1m = float(f1_score(targets.numpy(), logits.argmax(1).numpy(), average='macro', labels=present, zero_division=0))
    except Exception:
        f1m = -1.0

    acc = top1_accuracy(logits, targets)

    # Prediction histogram & per-class accuracy
    pred = logits.argmax(1).tolist()
    tgt = targets.tolist()
    pred_hist = Counter(pred)
    tgt_hist = Counter(tgt)
    per_class_acc: Dict[int, float] = {}
    right = defaultdict(int)
    for p, t in zip(pred, tgt):
        if p == t:
            right[t] += 1
    for c, n in tgt_hist.items():
        per_class_acc[c] = (right[c] / n) if n > 0 else 0.0

    # Print key info
    print('== Val coverage ==')
    present_classes = sorted(tgt_hist.keys())
    missing = sorted(set(range(NUM_CLASSES)) - set(present_classes))
    print('num samples:', sum(tgt_hist.values()), 'covered classes:', len(present_classes), '/ expected', NUM_CLASSES)
    print('missing classes:', missing)
    print('\n== Metrics (classes present in val) ==')
    print(f'Acc={acc:.4f}', f'F1(macro,present-only)={f1m:.4f}')
    print('\n== Prediction histogram Top-10 ==')
    for cls, cnt in pred_hist.most_common(10):
        print(f'class {cls}: {cnt}')
    print('\n== Per-class acc, worst Top-10 ==')
    worst = sorted(per_class_acc.items(), key=lambda x: x[1])[:10]
    for c, a in worst:
        print(f'class {c}: acc={a:.4f} (n={tgt_hist[c]})')


if __name__ == '__main__':
    evaluate_distribution()
