#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def _replace_classifier(module: nn.Module, num_classes: int) -> nn.Module:
    """Try to replace the final classifier layer with `num_classes`.

    Covers common patterns:
    - torchvision EfficientNetV2: `classifier` is Sequential, last layer Linear
    - torchvision ResNet: `fc`
    - Generic fallback: locate the last Linear and replace it on its parent
    """

    # 1) Common direct attributes
    if hasattr(module, 'classifier'):
        clf = getattr(module, 'classifier')
        if isinstance(clf, nn.Linear):
            in_f = clf.in_features
            setattr(module, 'classifier', nn.Linear(in_f, num_classes))
            return module
        # Sequential: replace the last Linear
        if isinstance(clf, nn.Sequential) and len(clf) > 0:
            # scan from the end to find a Linear
            for idx in range(len(clf) - 1, -1, -1):
                if isinstance(clf[idx], nn.Linear):
                    in_f = clf[idx].in_features
                    clf[idx] = nn.Linear(in_f, num_classes)
                    return module

    if hasattr(module, 'head') and isinstance(module.head, nn.Linear):
        in_f = module.head.in_features
        module.head = nn.Linear(in_f, num_classes)
        return module
    if hasattr(module, 'fc') and isinstance(module.fc, nn.Linear):
        in_f = module.fc.in_features
        module.fc = nn.Linear(in_f, num_classes)
        return module

    # 2) Generic fallback: locate the last Linear and replace it on its parent
    last_path = None
    for name, m in module.named_modules():
        if isinstance(m, nn.Linear):
            last_path = name  # record path, e.g., 'classifier.1' or 'head.fc'
    if not last_path:
        raise RuntimeError('Unable to locate classifier layer to replace to num_classes')

    # parse parent module and attr/index
    parent = module
    parts = last_path.split('.')
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]  # type: ignore[index]
        else:
            parent = getattr(parent, p)
    last_key = parts[-1]
    last_mod = getattr(parent, last_key) if not last_key.isdigit() else parent[int(last_key)]  # type: ignore[index]
    if not isinstance(last_mod, nn.Linear):
        raise RuntimeError('Last located layer is not Linear; cannot replace')
    in_f = last_mod.in_features
    new_lin = nn.Linear(in_f, num_classes)
    if last_key.isdigit():
        parent[int(last_key)] = new_lin  # type: ignore[index]
    else:
        setattr(parent, last_key, new_lin)
    return module


def build_model(model_name: str = 'efficientnetv2_s', num_classes: int = 61, *, pretrained: bool = True, drop_rate: float = 0.2) -> nn.Module:
    """Build a model.

    Prefer timm; if unavailable, fall back to torchvision alternatives (not v2-s, but runnable).
    """
    model = None
    try:
        import timm  # type: ignore

        # timm uses arg name drop_rate
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)
    except Exception:
        # Fallback: torchvision EfficientNet_v2_s; if unavailable, ResNet50
        try:
            from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

            weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
            model = efficientnet_v2_s(weights=weights)
            _replace_classifier(model, num_classes)
        except Exception:
            from torchvision.models import resnet50, ResNet50_Weights

            weights = ResNet50_Weights.DEFAULT if pretrained else None
            model = resnet50(weights=weights)
            _replace_classifier(model, num_classes)
    return model
