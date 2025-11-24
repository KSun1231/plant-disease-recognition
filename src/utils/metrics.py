#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch


def top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    with torch.no_grad():
        pred = logits.argmax(dim=1)
        correct = (pred == targets).sum().item()
        total = targets.numel()
        return correct / max(1, total)


def f1_macro(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    try:
        from sklearn.metrics import f1_score

        y_true = targets.cpu().numpy()
        y_pred = logits.argmax(dim=1).cpu().numpy()
        return float(f1_score(y_true, y_pred, average='macro', labels=list(range(num_classes)), zero_division=0))
    except Exception:
        # Fallback implementation: approximate via confusion matrix
        y_true = targets.cpu().numpy().astype(int)
        y_pred = logits.argmax(dim=1).cpu().numpy().astype(int)
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t, p in zip(y_true, y_pred):
            if 0 <= t < num_classes and 0 <= p < num_classes:
                cm[t, p] += 1
        f1s = []
        for c in range(num_classes):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            f1s.append(f1)
        return float(np.mean(f1s))
