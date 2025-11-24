#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Tuple

from torchvision import transforms as T


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(
    img_size: int = 224,
    *,
    scale: Tuple[float, float] = (0.7, 1.0),
    ratio: Tuple[float, float] = (3 / 4, 4 / 3),
    jitter: float = 0.2,
    hflip_p: float = 0.5,
    rotation: float = 15.0,
    erase_p: float = 0.25,
    imagenet_norm: bool = True,
):
    ops = [
        T.RandomResizedCrop(img_size, scale=scale, ratio=ratio),
        T.RandomHorizontalFlip(hflip_p),
        T.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter, hue=0.05),
        T.RandomRotation(degrees=rotation),
        T.ToTensor(),
    ]
    if imagenet_norm:
        ops.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    # RandomErasing operates on tensors
    if erase_p and erase_p > 0:
        ops.append(T.RandomErasing(p=erase_p, value='random'))
    return T.Compose(ops)


def get_val_transforms(
    img_size: int = 224,
    *,
    resize_short: int = 256,
    imagenet_norm: bool = True,
):
    ops = [
        T.Resize(resize_short),
        T.CenterCrop(img_size),
        T.ToTensor(),
    ]
    if imagenet_norm:
        ops.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return T.Compose(ops)
