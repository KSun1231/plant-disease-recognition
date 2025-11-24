#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import copy
from typing import Iterable

import torch


class ModelEma:
    def __init__(self, model, decay: float = 0.9999, device: str | None = None):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.device = device
        if device is not None:
            self.ema.to(device)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if k in msd:
                v.copy_(v * d + msd[k] * (1.0 - d))

    def to(self, device):
        self.ema.to(device)
        self.device = device
        return self
