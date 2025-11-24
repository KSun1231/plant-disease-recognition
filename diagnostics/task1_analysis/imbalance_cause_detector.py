#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Diagnostic for poor generalization due to class imbalance (no CLI args).

Capabilities:
- Read train/val lists and summarize coverage and distribution.
- Load current weights, run val forward pass, compute:
  - Prediction histogram, normalized entropy, Top‑1/Top‑5 concentration.
  - Macro/micro metrics (including present‑only macro‑F1).
  - Per-class accuracy and correlation with train frequencies.
- Based on thresholds, judge if imbalance/missing‑class is a highly likely cause and suggest actions.

Per AGENTS, all tunables are top-level globals; run this file directly.
"""

from __future__ import annotations

import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from src.dataloaders.build import DataConfig, build_dataloaders
from src.models.build_model import build_model


# ===== Configurable globals =====
# Data & lists
DATA_ROOT = 'data/ProblemB-Data'
TRAIN_LIST = 'datasets/train_list_rebalanced.txt'  # or original datasets/train_list.txt
VAL_LIST = 'datasets/val_list_rebalanced.txt'      # or original datasets/val_list.txt
NUM_CLASSES = 61

# Model & weights
MODEL_NAME = 'efficientnetv2_s'
WEIGHTS_PATH = 'outputs/task1_effv2s/best.pth'
IMG_SIZE = 224
BATCH_SIZE = 128
NUM_WORKERS = -1  # -1 means use all CPU cores
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Thresholds (tune as needed)
TOP1_SHARE_COLLAPSE = 0.30      # Top-1 share above threshold → collapse
TOP5_SHARE_COLLAPSE = 0.65      # Top-5 share above threshold → highly concentrated
NORM_ENTROPY_LOW = 0.70         # normalized entropy below threshold → narrow distribution
MACRO_MINUS_MICRO_GAP = 0.20    # macro - micro gap threshold (long-tail drag)
PRESENT_ONLY_F1_BOOST = 0.10    # present-only macro-F1 much higher than full → missing-class effect
FREQ_ACC_SPEARMAN = 0.30        # Spearman corr threshold of train freq vs per-class acc


# ===== Utilities =====
def read_list(path: str) -> List[Tuple[str, int]]:
    items: List[Tuple[str, int]] = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            items.append((parts[0].replace('\\', '/'), int(parts[1])))
    if not items:
        raise RuntimeError(f'List is empty: {path}')
    return items


def entropy(counts: Counter, n_classes: int) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in range(n_classes):
        p = counts.get(c, 0) / total
        if p > 0:
            h -= p * math.log(p + 1e-12)
    return h


def normalized_entropy(counts: Counter, n_classes: int) -> float:
    h = entropy(counts, n_classes)
    h_max = math.log(max(1, n_classes))
    return float(h / h_max) if h_max > 0 else 0.0


def spearman_rank_corr(xs: List[float], ys: List[float]) -> float:
    # Simplified Spearman: rank values, then Pearson
    if len(xs) != len(ys) or len(xs) == 0:
        return 0.0
    def rank(vs: List[float]) -> List[float]:
        order = sorted(range(len(vs)), key=lambda i: vs[i])
        r = [0]*len(vs)
        for rnk, i in enumerate(order):
            r[i] = rnk
        return r
    rx, ry = rank(xs), rank(ys)
    import numpy as np
    rx, ry = np.array(rx, dtype=float), np.array(ry, dtype=float)
    rx = (rx - rx.mean())/(rx.std()+1e-12)
    ry = (ry - ry.mean())/(ry.std()+1e-12)
    return float((rx*ry).mean())


def f1_macro_full_and_present_only(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> Tuple[float, float]:
    try:
        from sklearn.metrics import f1_score
        y_true = targets.numpy()
        y_pred = logits.argmax(1).numpy()
        f1_full = float(f1_score(y_true, y_pred, average='macro', labels=list(range(num_classes)), zero_division=0))
        present = sorted(set(int(x) for x in y_true.tolist()))
        f1_present = float(f1_score(y_true, y_pred, average='macro', labels=present, zero_division=0))
    except Exception:
        # Fallback: use utils.metrics implementation (full only)
        from src.utils.metrics import f1_macro
        f1_full = f1_macro(logits, targets, num_classes)
        f1_present = -1.0
    return f1_full, f1_present


# ===== Main flow =====
def main() -> None:
    print('== Imbalance-caused generalization diagnostic ==')
    train_entries = read_list(TRAIN_LIST)
    val_entries = read_list(VAL_LIST)
    train_counts = Counter(lab for _, lab in train_entries)
    val_counts = Counter(lab for _, lab in val_entries)
    print(f'[lists] train samples: {len(train_entries):,}, classes: {len(train_counts)}')
    print(f'[lists] val samples: {len(val_entries):,}, classes: {len(val_counts)}')
    missing_val = sorted(set(range(NUM_CLASSES)) - set(val_counts.keys()))
    if missing_val:
        print('[lists] val missing classes:', missing_val)
    else:
        print('[lists] val has all classes.')

    # DataLoader
    cfg = DataConfig(
        root=DATA_ROOT,
        train_list=TRAIN_LIST,
        val_list=VAL_LIST,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        class_balancing=False,
    )
    (_, _), (_, dl_val), _ = build_dataloaders(cfg)

    # Model
    if not os.path.isfile(WEIGHTS_PATH):
        raise FileNotFoundError(f'Weights not found: {WEIGHTS_PATH}')
    model = build_model(model_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=False, drop_rate=0.0)
    state = torch.load(WEIGHTS_PATH, map_location='cpu')
    sd = state.get('model', state)
    model.load_state_dict(sd, strict=False)
    model.to(DEVICE)
    model.eval()

    # Collect
    all_logits: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    with torch.no_grad():
        for xb, yb in dl_val:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            logits = model(xb)
            all_logits.append(logits.cpu())
            all_targets.append(yb.cpu())
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # Metrics
    from src.utils.metrics import top1_accuracy
    acc = top1_accuracy(logits, targets)
    f1_full, f1_present = f1_macro_full_and_present_only(logits, targets, NUM_CLASSES)
    preds = logits.argmax(1)
    pred_hist = Counter(int(x) for x in preds.tolist())
    top1_share = max(pred_hist.values())/max(1, sum(pred_hist.values()))
    top5_share = sum(v for _, v in pred_hist.most_common(5))/max(1, sum(pred_hist.values()))
    ne = normalized_entropy(pred_hist, NUM_CLASSES)

    # per-class acc
    tgt_hist = Counter(int(x) for x in targets.tolist())
    right = defaultdict(int)
    for p, t in zip(preds.tolist(), targets.tolist()):
        if p == t:
            right[int(t)] += 1
    per_class_acc: Dict[int, float] = {}
    for c, n in tgt_hist.items():
        per_class_acc[c] = right.get(c, 0) / n if n > 0 else 0.0

    # Correlation (train frequency vs per-class accuracy)
    common_classes = sorted(set(train_counts.keys()) & set(per_class_acc.keys()))
    if common_classes:
        xs = [float(train_counts[c]) for c in common_classes]
        ys = [float(per_class_acc[c]) for c in common_classes]
        rho = spearman_rank_corr(xs, ys)
    else:
        rho = 0.0

    # Report
    print('\n== Metrics ==')
    print(f'Acc={acc:.4f}', f'F1_full={f1_full:.4f}', f'F1_present-only={f1_present:.4f}')
    print(f'Top1-share={top1_share:.3f}', f'Top5-share={top5_share:.3f}', f'Norm-Entropy={ne:.3f}')
    print(f'Spearman(train_freq vs per-class-acc)={rho:.3f}')

    print('\n== Prediction concentration Top-10 ==')
    for cls, cnt in pred_hist.most_common(10):
        print(f'pred class {cls}: {cnt}')
    print('\n== Per-class accuracy (worst Top-10) ==')
    worst = sorted(per_class_acc.items(), key=lambda x: x[1])[:10]
    for c, a in worst:
        print(f'class {c}: acc={a:.4f} (val_n={tgt_hist[c]}) train_n={train_counts.get(c,0)}')

    # Attribution
    flags = []
    if missing_val:
        flags.append('missing class in val')
    if top1_share >= TOP1_SHARE_COLLAPSE or top5_share >= TOP5_SHARE_COLLAPSE:
        flags.append('prediction collapse')
    if ne <= NORM_ENTROPY_LOW:
        flags.append('low prediction entropy')
    micro = acc  # approximation as micro reference (note only)
    if f1_full >= 0 and (micro - f1_full) >= MACRO_MINUS_MICRO_GAP:
        flags.append('large macro/micro gap')
    if f1_present >= 0 and (f1_present - f1_full) >= PRESENT_ONLY_F1_BOOST:
        flags.append('present-only F1 >> full (missing class effect)')
    if rho >= FREQ_ACC_SPEARMAN:
        flags.append('per-class acc correlates with freq (long-tail)')

    print('\n== Conclusion ==')
    if flags:
        print('Highly likely generalization issues due to imbalance/missing classes:', ', '.join(flags))
    else:
        print('No strong imbalance signals; check other factors (aug, LR, regularization, etc.).')

    print('\n== Suggestions ==')
    print('- Keep balanced sampling and class weighting; if collapse persists, switch to log weighting and raise log_alpha.')
    print('- Ensure at least N samples per class in val (e.g., 15); re-balance splits if possible.')
    print('- Inspect worst Top-10 classes: targeted augmentation/oversampling; Focal loss if necessary.')
    print('- Enable MixUp/CutMix only after basics stabilize (incompatible with weighted CE).')


if __name__ == '__main__':
    main()
