#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task1 list diagnostics.

This script helps diagnose abnormal validation metrics by focusing on:
1. Sample sizes and class coverage for train/val lists.
2. Extreme per-class imbalance.
3. Whether relative paths exist under the data root.
4. Whether labels match the class prefix in filenames.
5. Cross-duplicates between train/val (leakage risk).

Per AGENTS, parameters live as top-level globals; no CLI args required.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple
import re

# ===== Global parameters =====
DATA_ROOT = Path('data/ProblemB-Data')
TRAIN_LIST_PATH = Path('datasets/train_list.txt')
VAL_LIST_PATH = Path('datasets/val_list.txt')
EXPECTED_NUM_CLASSES = 61
CHECK_IMAGES_EXISTENCE = True
TOP_IMBALANCED_CLASSES = 10
MAX_MISSING_REPORT = 20


@dataclass
class SplitStats:
    name: str
    entries: List[Tuple[str, int]]
    class_counts: Counter
    missing_files: List[Path]
    label_mismatches: List[Tuple[str, int, int]]  # (path, label, inferred_from_name)

    @property
    def num_samples(self) -> int:
        return len(self.entries)

    @property
    def covered_classes(self) -> int:
        return len(self.class_counts)

    @property
    def min_class_size(self) -> int:
        return min(self.class_counts.values()) if self.class_counts else 0

    @property
    def max_class_size(self) -> int:
        return max(self.class_counts.values()) if self.class_counts else 0


NUM_PATTERN = re.compile(r'(\d+)_')


def load_list(list_path: Path) -> List[Tuple[str, int]]:
    items: List[Tuple[str, int]] = []
    if not list_path.is_file():
        raise FileNotFoundError(f'List file not found: {list_path}')
    with list_path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            rel, lab = parts
            items.append((rel.replace('\\', '/'), int(lab)))
    if not items:
        raise RuntimeError(f'{list_path} is empty; cannot diagnose.')
    return items


def infer_label_from_name(rel_path: str) -> int | None:
    name = Path(rel_path).name
    match = NUM_PATTERN.match(name)
    if match:
        return int(match.group(1))
    return None


def verify_entries(entries: Sequence[Tuple[str, int]]) -> Tuple[List[Path], List[Tuple[str, int, int]]]:
    missing: List[Path] = []
    mismatched: List[Tuple[str, int, int]] = []
    root_exists = DATA_ROOT.exists()
    for rel, label in entries:
        rel_path = Path(rel)
        if CHECK_IMAGES_EXISTENCE and root_exists:
            abs_path = DATA_ROOT / rel_path
            if not abs_path.is_file():
                missing.append(rel_path)
        inferred = infer_label_from_name(rel)
        if inferred is not None and inferred != label:
            mismatched.append((rel, label, inferred))
    return missing, mismatched


def build_stats(name: str, entries: List[Tuple[str, int]]) -> SplitStats:
    class_counts = Counter(lab for _, lab in entries)
    missing, mismatched = verify_entries(entries)
    return SplitStats(name=name, entries=entries, class_counts=class_counts, missing_files=missing, label_mismatches=mismatched)


def describe_split(stats: SplitStats) -> None:
    print(f'[{stats.name}] Num samples: {stats.num_samples:,}')
    print(f'[{stats.name}] Covered classes: {stats.covered_classes} / expected {EXPECTED_NUM_CLASSES}')
    print(f'[{stats.name}] Min per-class: {stats.min_class_size}, max: {stats.max_class_size}')
    missing_classes = sorted(set(range(EXPECTED_NUM_CLASSES)) - set(stats.class_counts.keys()))
    if missing_classes:
        print(f'[{stats.name}] Missing classes: {missing_classes}')
    small_classes = sorted(stats.class_counts.items(), key=lambda x: x[1])[:TOP_IMBALANCED_CLASSES]
    print(f'[{stats.name}] Smallest {TOP_IMBALANCED_CLASSES} classes: {small_classes}')
    if stats.missing_files:
        report = stats.missing_files[:MAX_MISSING_REPORT]
        print(f'[{stats.name}] Missing {len(stats.missing_files)} files (data root {DATA_ROOT}). Examples:')
        for rel in report:
            print('    ', rel)
    elif not DATA_ROOT.exists():
        print(f'[{stats.name}] Data root {DATA_ROOT} not found; skip file existence checks.')
    if stats.label_mismatches:
        example = stats.label_mismatches[:MAX_MISSING_REPORT]
        print(f'[{stats.name}] {len(stats.label_mismatches)} filename/label mismatches. Examples:')
        for rel, label, inferred in example:
            print(f'    {rel} -> label {label} / filename prefix {inferred}')


def summarize_overlap(train_entries: Sequence[Tuple[str, int]], val_entries: Sequence[Tuple[str, int]]) -> None:
    train_set = set(train_entries)
    val_set = set(val_entries)
    overlap = train_set & val_set
    if overlap:
        print(f'[overlap] {len(overlap)} identical entries between train and val — leakage risk.')
    else:
        print('[overlap] No path+label duplicates between train and val.')
    shared_paths = set(rel for rel, _ in train_entries) & set(rel for rel, _ in val_entries)
    if shared_paths:
        print(f'[overlap] {len(shared_paths)} paths appear in both lists (labels may differ) — check annotation consistency.')


def summarize_label_gaps(train_stats: SplitStats, val_stats: SplitStats) -> None:
    train_classes = set(train_stats.class_counts.keys())
    val_classes = set(val_stats.class_counts.keys())
    only_train = sorted(train_classes - val_classes)
    only_val = sorted(val_classes - train_classes)
    if only_train:
        print(f'[class diff] Classes only in train: {only_train}')
    if only_val:
        print(f'[class diff] Classes only in val: {only_val}')
    if not only_train and not only_val:
        print('[class diff] Train and val class sets are identical.')


def main() -> None:
    print('== Task1 list diagnostics ==')
    train_entries = load_list(TRAIN_LIST_PATH)
    val_entries = load_list(VAL_LIST_PATH)
    train_stats = build_stats('train', train_entries)
    val_stats = build_stats('val', val_entries)
    describe_split(train_stats)
    print('-' * 60)
    describe_split(val_stats)
    print('-' * 60)
    summarize_label_gaps(train_stats, val_stats)
    summarize_overlap(train_entries, val_entries)
    print('== Done ==')


if __name__ == '__main__':
    main()
