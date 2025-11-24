#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Label permutation suspicion check (no CLI args).

Idea:
- Build confusion matrix cm[true, pred] on validation.
- Greedy one-to-one matching to find an upper bound accuracy for a possible label permutation.
- If the remapped upper bound is significantly higher than baseline, labels/logit order may be inconsistent.

Per AGENTS, parameters are at top; run directly.
"""

from __future__ import annotations

from typing import List, Tuple
from collections import Counter
import os

import torch

from src.dataloaders.build import DataConfig, build_dataloaders
from src.models.build_model import build_model
from src.utils.metrics import top1_accuracy


# ===== Global parameters =====
DATA_ROOT = 'data/ProblemB-Data'
VAL_LIST = 'data/ProblemB-Data/AgriculturalDisease_validationset/ttest_list.txt'
MODEL_NAME = 'efficientnetv2_s'
NUM_CLASSES = 61
WEIGHTS_PATH = 'outputs/task1_effv2s/best.pth'
IMG_SIZE = 224
BATCH_SIZE = 128
NUM_WORKERS = -1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def greedy_bipartite_match(cm: torch.Tensor) -> Tuple[List[int], float]:
    """Greedy one-to-one mapping on the confusion matrix; return pred->true map and upper-bound accuracy.

    Notes:
    - Greedy: for each true row (sorted by max value), assign the best unused pred column.
    - An approximation to Hungarian; used as an upper bound for quick suspicion checks.
    """
    cm = cm.clone().float()
    n = cm.size(0)
    # record (row_idx, max_val) per row
    row_order = sorted(range(n), key=lambda r: float(cm[r].max()), reverse=True)
    used_pred = set()
    mapping_pred_to_true = [-1] * n
    total = float(cm.sum())
    agree = 0.0
    for r in row_order:
        # choose the largest unused column for this row
        row = cm[r]
        vals = row.tolist()
        # columns sorted by descending value
        cols = sorted(range(n), key=lambda c: vals[c], reverse=True)
        chosen = None
        for c in cols:
            if c not in used_pred:
                chosen = c
                break
        if chosen is None:
            continue
        used_pred.add(chosen)
        mapping_pred_to_true[chosen] = r
        agree += float(cm[r, chosen])
    acc_upper = agree / max(1.0, total)
    return mapping_pred_to_true, acc_upper


@torch.no_grad()
def main():
    print('== Label permutation suspicion check ==')
    cfg = DataConfig(
        root=DATA_ROOT,
        train_list='datasets/train_list.txt',
        val_list=VAL_LIST,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        class_balancing=False,
    )
    (_, _), (_, dl_val), _ = build_dataloaders(cfg)

    if not os.path.isfile(WEIGHTS_PATH):
        raise FileNotFoundError(f'Weights not found: {WEIGHTS_PATH}')
    model = build_model(model_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=False, drop_rate=0.0)
    state = torch.load(WEIGHTS_PATH, map_location='cpu')
    sd = state.get('model', state)
    model.load_state_dict(sd, strict=False)
    device = DEVICE
    model.to(device)
    model.eval()

    all_logits = []
    all_targets = []
    for xb, yb in dl_val:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        all_logits.append(logits.cpu())
        all_targets.append(yb.cpu())
    logits = torch.cat(all_logits, 0)
    targets = torch.cat(all_targets, 0)
    acc = top1_accuracy(logits, targets)
    print(f'[baseline] Acc = {acc:.4f}')

    # build confusion matrix (true x pred)
    preds = logits.argmax(1)
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    for t, p in zip(targets.tolist(), preds.tolist()):
        if 0 <= t < NUM_CLASSES and 0 <= p < NUM_CLASSES:
            cm[t, p] += 1

    mapping, acc_upper = greedy_bipartite_match(cm)
    used = sum(1 for m in mapping if m >= 0)
    print(f'[perm upper] Acc_upper ≈ {acc_upper:.4f} (covers {used} classes)')
    if acc_upper - acc >= 0.10:
        print('[hint] Upper bound is much higher than baseline; labels/logit order may be inconsistent. Verify label mapping and list consistency.')
    else:
        print('[hint] No strong permutation signal; likely generalization or validation split issue.')


if __name__ == '__main__':
    main()
