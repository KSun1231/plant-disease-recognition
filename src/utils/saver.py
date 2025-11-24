#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from typing import Any, Dict

import torch


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def save_checkpoint(path: str, state: Dict[str, Any]):
    ensure_dir(os.path.dirname(path))
    torch.save(state, path)

