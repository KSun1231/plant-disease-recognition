#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import sys

import torch

from src.dataloaders.build import DataConfig, build_dataloaders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/ProblemB-Data')
    ap.add_argument('--train-list', default='datasets/train_list.txt')
    ap.add_argument('--val-list', default='datasets/val_list.txt')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--img-size', type=int, default=224)
    ap.add_argument('--balanced', action='store_true')
    args = ap.parse_args()

    cfg = DataConfig(
        root=args.root,
        train_list=args.train_list,
        val_list=args.val_list,
        batch_size=args.batch_size,
        num_workers=args.workers,
        img_size=args.img_size,
        class_balancing=args.balanced,
    )
    (ds_tr, ds_va), (dl_tr, dl_va), info = build_dataloaders(cfg)
    print('[i] info:', info)

    # preview a batch
    xb, yb = next(iter(dl_tr))
    print('[i] train batch:', xb.shape, yb.shape, yb[:8])
    xb2, yb2 = next(iter(dl_va))
    print('[i] val   batch:', xb2.shape, yb2.shape, yb2[:8])


if __name__ == '__main__':
    main()
