#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset


def read_list_file(path: str) -> List[Tuple[str, int]]:
    pairs: List[Tuple[str, int]] = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            p, lab = parts
            pairs.append((p.replace('\\', '/'), int(lab)))
    return pairs


@dataclass
class ListDatasetConfig:
    root: str = 'data/ProblemB-Data'
    list_file: str = 'datasets/train_list.txt'
    min_dim: int = 16
    strict: bool = False  # True: raise on bad samples; False: return None and let collate_fn drop


class ListDataset(Dataset):
    """Dataset based on a `path label` list file.

    - Apply EXIF transpose and convert to RGB during loading.
    - If strict=False, return None for bad samples; use with safe_collate.
    """

    def __init__(
        self,
        root: str,
        list_file: str,
        transform: Optional[Callable] = None,
        *,
        min_dim: int = 16,
        strict: bool = False,
    ) -> None:
        super().__init__()
        self.root = root
        self.list_file = list_file
        self.items: List[Tuple[str, int]] = read_list_file(list_file)
        self.t = transform
        self.min_dim = min_dim
        self.strict = strict

    def __len__(self) -> int:
        return len(self.items)

    def _load_image(self, abs_path: str) -> Image.Image:
        img = Image.open(abs_path)
        img.load()
        img = ImageOps.exif_transpose(img)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img

    def __getitem__(self, i: int):
        rel, y = self.items[i]
        abs_path = os.path.join(self.root, rel)
        try:
            img = self._load_image(abs_path)
            w, h = img.size
            if w < self.min_dim or h < self.min_dim:
                raise ValueError(f'small:{w}x{h}')
            if self.t is not None:
                img = self.t(img)
            return img, y
        except Exception as e:  # noqa: BLE001
            if self.strict:
                raise
            return None

    def class_counts(self) -> dict:
        from collections import Counter

        return dict(Counter(lab for _, lab in self.items))


def safe_collate(batch: Sequence[Optional[Tuple[torch.Tensor, int]]]):
    """Collate that safely drops None entries."""
    filtered: List[Tuple[torch.Tensor, int]] = [b for b in batch if b is not None]
    if not filtered:
        # return a minimal empty batch to avoid DataLoader crash
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    xs, ys = zip(*filtered)
    return torch.stack(list(xs), dim=0), torch.tensor(list(ys), dtype=torch.long)
