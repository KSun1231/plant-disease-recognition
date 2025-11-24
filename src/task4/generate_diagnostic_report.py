#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-task diagnostic report generator (load best.pth, sample validation, output CSV and Grad‑CAM visualizations)

AGENTS constraints: no CLI args; configure at top and run.
"""

from __future__ import annotations

import os
import csv
from typing import List, Tuple

# --- Add repo root to sys.path when running directly (fix ModuleNotFoundError: 'src') ---
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn as nn
from torchvision import transforms as T
from PIL import Image, ImageOps
import torch.nn.functional as F

from src.task4.multitask_train import MultiTaskNet, map_to_severity, DATA_ROOT, VAL_LIST, MODEL_NAME, NUM_CLASSES, NUM_SEVERITY
from src.utils.saver import ensure_dir


# ===== Globals =====
WEIGHTS_PATH = 'outputs/task4_multitask/best.pth'
OUT_DIR = 'outputs/task4_multitask/report'
MAX_SAMPLES = 64
IMG_SIZE = 224
RESIZE_SHORT = 256
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def read_list_file(path: str) -> List[Tuple[str, int]]:
    items: List[Tuple[str, int]] = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            items.append((parts[0].replace('\\','/'), int(parts[1])))
    return items


def build_transforms():
    t = T.Compose([
        T.Resize(RESIZE_SHORT), T.CenterCrop(IMG_SIZE), T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return t


@torch.no_grad()
def forward_batch(model, imgs: torch.Tensor):
    model.eval()
    return model(imgs)


def grad_cam(model: nn.Module, x: torch.Tensor, target_layer: nn.Module) -> torch.Tensor:
    feats = None
    grads = None
    def f_hook(_, __, out):
        nonlocal feats
        feats = out
    def b_hook(_, gin, gout):
        nonlocal grads
        grads = gout[0]
    h1 = target_layer.register_forward_hook(f_hook)
    h2 = target_layer.register_full_backward_hook(b_hook)
    try:
        logits_cls, _ = model(x)
        cls = logits_cls.argmax(1)
        score = logits_cls.gather(1, cls.view(-1,1)).sum()
        model.zero_grad(set_to_none=True)
        score.backward()
        w = grads.mean(dim=(2,3), keepdim=True)
        cam = (w*feats).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam -= cam.amin(dim=(2,3), keepdim=True)
        cam /= (cam.amax(dim=(2,3), keepdim=True)+1e-6)
        return cam
    finally:
        h1.remove(); h2.remove()


def find_last_conv(m: nn.Module) -> nn.Module:
    last=None
    for mod in m.modules():
        if isinstance(mod, nn.Conv2d):
            last=mod
    if last is None:
        raise RuntimeError('No Conv2d layer found')
    return last


def main():
    ensure_dir(OUT_DIR)
    # read samples
    entries = read_list_file(VAL_LIST)
    entries = entries[:MAX_SAMPLES]
    t = build_transforms()

    # load model
    model = MultiTaskNet(MODEL_NAME, NUM_CLASSES, NUM_SEVERITY, pretrained=False)
    state = torch.load(WEIGHTS_PATH, map_location='cpu')
    sd = state.get('model', state)
    model.load_state_dict(sd, strict=False)
    model.to(DEVICE)
    model.eval()
    layer = find_last_conv(model)

    # diagnostic CSV
    csv_path = os.path.join(OUT_DIR, 'diagnostic.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['path','label_id','pred_cls','conf_cls','pred_sev','conf_sev'])
        for i, (rel, y) in enumerate(entries):
            # read image
            from torchvision.utils import save_image
            img = Image.open(os.path.join(DATA_ROOT, rel))
            img = ImageOps.exif_transpose(img)
            if img.mode!='RGB':
                img = img.convert('RGB')
            x = t(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits_cls, logits_sev = model(x)
                prob_c = torch.softmax(logits_cls, dim=1)[0]
                prob_s = torch.softmax(logits_sev, dim=1)[0]
                pred_c = int(prob_c.argmax().item())
                pred_s = int(prob_s.argmax().item())
                conf_c = float(prob_c[pred_c].item())
                conf_s = float(prob_s[pred_s].item())
            # write row
            w.writerow([rel, int(y), pred_c, f'{conf_c:.4f}', pred_s, f'{conf_s:.4f}'])
            # save CAM overlay
            cam = grad_cam(model, x, layer)[0]  # [1,Hf,Wf]
            show = x[0].detach().cpu()
            show = (show*torch.tensor([0.229,0.224,0.225]).view(3,1,1) + torch.tensor([0.485,0.456,0.406]).view(3,1,1)).clamp(0,1)
            # upsample to input size before overlay
            try:
                cam = F.interpolate(cam.unsqueeze(0), size=show.shape[1:], mode='bilinear', align_corners=False)[0]
            except Exception:
                pass
            heat = cam.expand_as(show).cpu()
            mix = (show*0.6 + heat*0.4).clamp(0,1)
            save_image(mix, os.path.join(OUT_DIR, f'cam_{i:03d}.png'))
    print('[i] Diagnostic report generated:', csv_path)


if __name__ == '__main__':
    main()
