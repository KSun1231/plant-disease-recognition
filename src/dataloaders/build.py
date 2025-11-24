#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.datamodules.list_dataset import ListDataset, ListDatasetConfig, safe_collate
from src.transforms.build import get_train_transforms, get_val_transforms


@dataclass
class DataConfig:
    # paths
    root: str = 'data/ProblemB-Data'
    train_list: str = 'datasets/train_list.txt'
    val_list: str = 'datasets/val_list.txt'

    # runtime
    img_size: int = 224
    resize_short: int = 256
    batch_size: int = 64
    num_workers: int = 4  # if <=0, use os.cpu_count()
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    drop_last_train: bool = True

    # robustness
    min_dim: int = 16
    strict: bool = False

    # sampling
    class_balancing: bool = False
    weight_scheme: str = 'sqrt'  # 'sqrt' | 'log'
    weight_log_alpha: float = 1.02  # for 'log'

    # seed
    seed: int = 42


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(labels: Iterable[int], *, scheme: str = 'sqrt', log_alpha: float = 1.02) -> Dict[int, float]:
    from collections import Counter

    cnt = Counter(labels)
    if not cnt:
        return {}
    weights: Dict[int, float] = {}
    if scheme == 'sqrt':
        for c, n in cnt.items():
            weights[c] = 1.0 / math.sqrt(max(1, n))
    elif scheme == 'log':
        alpha = max(1.0001, float(log_alpha))
        for c, n in cnt.items():
            weights[c] = 1.0 / math.log(alpha + max(1, n))
    else:
        for c, n in cnt.items():
            weights[c] = 1.0 / max(1, n)
    # normalize to mean=1 for sampler stability
    mean_w = sum(weights.values()) / len(weights)
    for k in list(weights.keys()):
        weights[k] /= mean_w
    return weights


def build_datasets(cfg: DataConfig):
    t_train = get_train_transforms(img_size=cfg.img_size)
    t_val = get_val_transforms(img_size=cfg.img_size, resize_short=cfg.resize_short)

    ds_train = ListDataset(
        cfg.root,
        cfg.train_list,
        transform=t_train,
        min_dim=cfg.min_dim,
        strict=cfg.strict,
    )
    ds_val = ListDataset(
        cfg.root,
        cfg.val_list,
        transform=t_val,
        min_dim=cfg.min_dim,
        strict=True,  # be strict on validation
    )
    return ds_train, ds_val


def _worker_init_fn(worker_id: int):  # noqa: D401
    """Derive per-worker RNG seed for reproducible augmentations."""
    base_seed = torch.initial_seed() % 2**31
    random.seed(base_seed + worker_id)
    np.random.seed(base_seed + worker_id)


def build_dataloaders(cfg: DataConfig):
    seed_everything(cfg.seed)
    ds_train, ds_val = build_datasets(cfg)

    # auto workers: if <=0, use all CPU cores
    workers_eff = cfg.num_workers if cfg.num_workers and cfg.num_workers > 0 else max(1, os.cpu_count() or 1)

    sampler = None
    if cfg.class_balancing:
        # compute class weights and map to per-sample weights
        labels = [lab for _, lab in ds_train.items]
        class_w = compute_class_weights(labels, scheme=cfg.weight_scheme, log_alpha=cfg.weight_log_alpha)
        sample_w = [class_w.get(lab, 1.0) for _, lab in ds_train.items]
        sampler = WeightedRandomSampler(weights=sample_w, num_samples=len(sample_w), replacement=True)

    dl_train = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=workers_eff,
        pin_memory=cfg.pin_memory,
        persistent_workers=(workers_eff > 0 and cfg.persistent_workers),
        prefetch_factor=(cfg.prefetch_factor if workers_eff > 0 else None),
        drop_last=cfg.drop_last_train,
        collate_fn=safe_collate,
    )

    dl_val = DataLoader(
        ds_val,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=workers_eff,
        pin_memory=cfg.pin_memory,
        persistent_workers=(workers_eff > 0 and cfg.persistent_workers),
        prefetch_factor=(cfg.prefetch_factor if workers_eff > 0 else None),
    )

    info = {
        'train_len': len(ds_train),
        'val_len': len(ds_val),
        'train_class_counts': ds_train.class_counts(),
        'val_class_counts': ds_val.class_counts(),
        'num_workers': workers_eff,
    }
    return (ds_train, ds_val), (dl_train, dl_val), info


if __name__ == '__main__':
    # quick sanity: build only, no iteration (lists must exist)
    cfg = DataConfig()
    print('[i] building dataloaders with config:', cfg)
    _, (_, _), info = build_dataloaders(cfg)
    print('[i] dataset info:', info)
