#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import os
from typing import Dict


class CsvLogger:
    def __init__(self, path: str, fieldnames):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, 'w', encoding='utf-8', newline='')
        self.writer = csv.DictWriter(self.f, fieldnames=fieldnames)
        self.writer.writeheader()

    def log(self, row: Dict):
        self.writer.writerow(row)
        self.f.flush()

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass

