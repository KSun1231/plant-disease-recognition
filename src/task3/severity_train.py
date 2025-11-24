#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task3: Disease severity grading (4 classes: healthy/mild/moderate/severe)

Requirements & highlights:
- No CLI args; configure at top and run to train.
- Map 61-class labels to 4 severity levels:
  - 0=healthy, 1=mild, 2=moderate, 3=severe.
  - Based on class names (Healthy/General/Serious).
  - Others not explicitly marked (e.g., some Virus/Spot without General/Serious) default to Moderate.
- Data: reuse `datasets/train_list.txt` and `datasets/val_list.txt` path+label.
- Model: reuse Task1 `build_model` (default EfficientNetV2‑S).
- Training: CE (with label smoothing) + AMP + grad clip + save best.
- Evaluation: Acc, macro‑F1, and recall for each severity (4 classes).
- Explainability: save several Grad‑CAM visualizations (last Conv features).
"""

from __future__ import annotations

import os
import random
from typing import Dict, Iterable, List, Sequence, Tuple

# --- Add repo root to sys.path when running directly (fix ModuleNotFoundError: 'src') ---
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T
from PIL import Image, ImageOps
import torch.nn.functional as F

from src.models.build_model import build_model
from src.utils.metrics import top1_accuracy, f1_macro
from src.utils.logger import CsvLogger
from src.utils.saver import ensure_dir, save_checkpoint


# ===== Global parameters (editable) =====
DATA_ROOT = 'data/ProblemB-Data'
TRAIN_LIST = 'datasets/train_list.txt'  # or data/.../train_list.txt
VAL_LIST = 'datasets/val_list.txt'      # or data/.../ttest_list.txt
OUT_DIR = 'outputs/task3_severity'

NUM_SEVERITY = 4  # 0=healthy, 1=mild, 2=moderate, 3=severe

# Model & training
MODEL_NAME = 'efficientnetv2_s'
PRETRAINED = True
EPOCHS = 20
LR = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
AMP = True
GRAD_CLIP_NORM = 1.0
SEED = 2025

# DataLoader
IMG_SIZE = 224
RESIZE_SHORT = 256
BATCH_SIZE = 64
NUM_WORKERS = -1  # -1 means use all CPU cores
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2

# Training balance (severity counts are often imbalanced)
ENABLE_BALANCED_SAMPLER = True
BALANCE_SCHEME = 'sqrt'  # sqrt | log | inverse
BALANCE_LOG_ALPHA = 1.05

# Grad‑CAM visualization
SAVE_CAM = True
NUM_CAM_SAMPLES = 12


# ===== 61->4 mapping =====
# Known Healthy classes (from the official list; adjust if needed):
HEALTHY_IDS = {0, 6, 9, 17, 24, 27, 30, 33, 38, 41}

# Known General/Serious pairs (not exhaustive; others default to Moderate):
GENERAL_IDS = {
    1, 4, 7, 10, 12, 14, 18, 20, 22, 25, 28, 31,
    34, 36, 39, 42, 44, 52, 54, 56, 58,
}
SERIOUS_IDS = {
    2, 5, 8, 11, 13, 15, 19, 21, 23, 26, 29, 32,
    35, 37, 40, 43, 45, 53, 55, 57, 59,
}

# Override mapping here if needed (highest priority), e.g., {60:2}
CUSTOM_SEVERITY_MAP: Dict[int, int] = {
    60: 2,  # Tomato Mosaic Virus (no explicit severity; use Moderate)
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
            rel, lab = parts
            items.append((rel.replace('\\', '/'), int(lab)))
    return items


def map_to_severity(cls61: int) -> int:
    # custom override
    if cls61 in CUSTOM_SEVERITY_MAP:
        return int(CUSTOM_SEVERITY_MAP[cls61])
    if cls61 in HEALTHY_IDS:
        return 0  # healthy
    if cls61 in GENERAL_IDS:
        return 1  # mild
    if cls61 in SERIOUS_IDS:
        return 3  # severe
    return 2      # moderate (default)


class ListDatasetSeverity(Dataset):
    def __init__(self, root: str, entries: Sequence[Tuple[str, int]], transform=None, *, min_dim: int = 16):
        self.root = root
        self.items = list(entries)
        self.t = transform
        self.min_dim = min_dim

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
        rel, y61 = self.items[i]
        abs_path = os.path.join(self.root, rel)
        img = self._load_image(abs_path)
        if img.size[0] < self.min_dim or img.size[1] < self.min_dim:
            raise ValueError('image too small')
        if self.t is not None:
            img = self.t(img)
        y4 = map_to_severity(int(y61))
        return img, y4


def build_transforms(img_size: int = 224, resize_short: int = 256):
    t_train = T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(3/4, 4/3)),
        T.RandomHorizontalFlip(0.5),
        T.ColorJitter(0.2, 0.2, 0.2, 0.05),
        T.RandomRotation(15),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    t_val = T.Compose([
        T.Resize(resize_short),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return t_train, t_val


def compute_class_weights(labels: Iterable[int], *, scheme: str = 'sqrt', log_alpha: float = 1.05) -> Dict[int, float]:
    from collections import Counter
    import math
    cnt = Counter(labels)
    if not cnt:
        return {}
    w: Dict[int, float] = {}
    if scheme == 'sqrt':
        for c, n in cnt.items():
            w[c] = 1.0 / (max(1, n) ** 0.5)
    elif scheme == 'log':
        alpha = max(1.0001, float(log_alpha))
        import math as m
        for c, n in cnt.items():
            w[c] = 1.0 / m.log(alpha + max(1, n))
    else:
        for c, n in cnt.items():
            w[c] = 1.0 / max(1, n)
    mean = sum(w.values())/len(w)
    for k in list(w.keys()):
        w[k] /= mean
    return w


def build_dataloaders():
    set_seed(SEED)
    t_train, t_val = build_transforms(IMG_SIZE, RESIZE_SHORT)
    tr = read_list_file(TRAIN_LIST)
    va = read_list_file(VAL_LIST)
    ds_tr = ListDatasetSeverity(DATA_ROOT, tr, transform=t_train)
    ds_va = ListDatasetSeverity(DATA_ROOT, va, transform=t_val)
    workers = NUM_WORKERS if NUM_WORKERS and NUM_WORKERS > 0 else max(1, os.cpu_count() or 1)

    sampler = None
    if ENABLE_BALANCED_SAMPLER:
        labels = [map_to_severity(y) for _, y in tr]
        cw = compute_class_weights(labels, scheme=BALANCE_SCHEME, log_alpha=BALANCE_LOG_ALPHA)
        sw = [cw.get(map_to_severity(y), 1.0) for _, y in tr]
        sampler = WeightedRandomSampler(weights=sw, num_samples=len(sw), replacement=True)

    dl_tr = DataLoader(
        ds_tr,
        batch_size=BATCH_SIZE,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=workers,
        pin_memory=PIN_MEMORY,
        drop_last=True,
        persistent_workers=(workers > 0 and PERSISTENT_WORKERS),
        prefetch_factor=(PREFETCH_FACTOR if workers > 0 else None),
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=workers,
        pin_memory=PIN_MEMORY,
        persistent_workers=(workers > 0 and PERSISTENT_WORKERS),
        prefetch_factor=(PREFETCH_FACTOR if workers > 0 else None),
    )
    return ds_tr, ds_va, dl_tr, dl_va


def create_amp():
    try:
        from torch.amp import autocast as autocast_new  # type: ignore
        from torch.amp import GradScaler as GradScalerNew  # type: ignore
        return (lambda enabled: autocast_new('cuda', enabled=enabled)), (lambda enabled: GradScalerNew('cuda', enabled=enabled))
    except Exception:
        from torch.cuda.amp import autocast as autocast_old  # type: ignore
        from torch.cuda.amp import GradScaler as GradScalerOld  # type: ignore
        return (lambda enabled: autocast_old(enabled=enabled)), (lambda enabled: GradScalerOld(enabled=enabled))


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, *, grad_clip: float | None = None):
    model.train()
    loss_m = 0.0
    acc_m = 0.0
    n = 0
    autocast_ctx, _ = create_amp()
    use_amp = scaler is not None
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx(enabled=use_amp):
            logits = model(xb)
            loss = criterion(logits, yb)
        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        with torch.no_grad():
            bsz = xb.size(0)
            loss_m += float(loss.detach().item()) * bsz
            acc_m += top1_accuracy(logits, yb) * bsz
            n += bsz
    return loss_m/max(1,n), acc_m/max(1,n)


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes: int):
    model.eval()
    loss_m = 0.0
    acc_m = 0.0
    n = 0
    all_logits = []
    all_targets = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        bsz = xb.size(0)
        loss_m += float(loss.detach().item()) * bsz
        acc_m += top1_accuracy(logits, yb) * bsz
        n += bsz
        all_logits.append(logits.cpu())
        all_targets.append(yb.cpu())
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    f1m = f1_macro(logits, targets, num_classes)
    # per-class recall
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    pred = logits.argmax(1)
    for t, p in zip(targets.tolist(), pred.tolist()):
        cm[t, p] += 1
    recall = []
    for c in range(num_classes):
        tp = cm[c, c].item()
        fn = cm[c, :].sum().item() - tp
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recall.append(rec)
    return loss_m/max(1,n), acc_m/max(1,n), f1m, recall


def grad_cam(model: nn.Module, x: torch.Tensor, target_layer: nn.Module) -> torch.Tensor:
    """Minimal Grad‑CAM; return heatmap at input resolution (normalized to 0..1)."""
    feats = None
    grads = None

    def f_hook(_, __, output):
        nonlocal feats
        feats = output

    def b_hook(_, grad_in, grad_out):
        nonlocal grads
        grads = grad_out[0]

    h1 = target_layer.register_forward_hook(f_hook)
    h2 = target_layer.register_full_backward_hook(b_hook)
    try:
        logits = model(x)
        cls = logits.argmax(1)
        score = logits.gather(1, cls.view(-1,1)).sum()
        model.zero_grad(set_to_none=True)
        score.backward()
        # GAP weights
        w = grads.mean(dim=(2,3), keepdim=True)  # [B,C,1,1]
        cam = (w * feats).sum(dim=1, keepdim=True)  # [B,1,H,W]
        cam = torch.relu(cam)
        # normalize to 0..1
        cam -= cam.amin(dim=(2,3), keepdim=True)
        cam /= (cam.amax(dim=(2,3), keepdim=True) + 1e-6)
        return cam
    finally:
        h1.remove(); h2.remove()


def find_last_conv(m: nn.Module) -> nn.Module:
    last = None
    for mod in m.modules():
        if isinstance(mod, nn.Conv2d):
            last = mod
    if last is None:
        raise RuntimeError('No Conv2d layer found for Grad‑CAM')
    return last


def save_cam_examples(model, dl_val, device, out_dir: str, n_samples: int = 12) -> None:
    ensure_dir(out_dir)
    layer = find_last_conv(model)
    saved = 0
    for xb, yb in dl_val:
        xb = xb.to(device)
        cam = grad_cam(model, xb, layer)  # [B,1,Hf,Wf]
        # upsample CAM to input size for overlay
        try:
            cam = F.interpolate(cam, size=xb.shape[2:], mode='bilinear', align_corners=False)
        except Exception:
            pass
        for i in range(min(xb.size(0), n_samples - saved)):
            import torchvision.utils as vutils
            # overlay heatmap (simple linear blend)
            heat = cam[i].expand_as(xb[i]).cpu()
            show = xb[i].detach().cpu()
            # de-normalize to 0..1 range
            show = (show * torch.tensor([0.229,0.224,0.225]).view(3,1,1) + torch.tensor([0.485,0.456,0.406]).view(3,1,1)).clamp(0,1)
            mix = (show*0.6 + heat*0.4).clamp(0,1)
            vutils.save_image(mix, os.path.join(out_dir, f'cam_{saved:03d}.png'))
            saved += 1
            if saved >= n_samples:
                return


def main():
    set_seed(SEED)
    ensure_dir(OUT_DIR)
    ds_tr, ds_va, dl_tr, dl_va = build_dataloaders()
    # model
    model = build_model(model_name=MODEL_NAME, num_classes=NUM_SEVERITY, pretrained=PRETRAINED, drop_rate=0.2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # optimizer & loss
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    _, create_scaler = create_amp()
    scaler = create_scaler(AMP)

    logger = CsvLogger(os.path.join(OUT_DIR, 'train_log.csv'), fieldnames=['epoch','lr','train_loss','train_acc','val_loss','val_acc','val_f1_macro','recall_healthy','recall_mild','recall_moderate','recall_severe','best_metric'])
    best_f1 = -1.0

    for epoch in range(EPOCHS):
        tr_loss, tr_acc = train_one_epoch(model, dl_tr, criterion, optimizer, device, scaler, grad_clip=GRAD_CLIP_NORM)
        va_loss, va_acc, va_f1, recall = evaluate(model, dl_va, criterion, device, NUM_SEVERITY)
        lr_curr = optimizer.param_groups[0]['lr']
        if va_f1 > best_f1:
            best_f1 = va_f1
            save_checkpoint(os.path.join(OUT_DIR, 'best.pth'), {
                'epoch': epoch+1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            })
        logger.log({
            'epoch': epoch+1,
            'lr': lr_curr,
            'train_loss': tr_loss,
            'train_acc': tr_acc,
            'val_loss': va_loss,
            'val_acc': va_acc,
            'val_f1_macro': va_f1,
            'recall_healthy': recall[0] if len(recall)>0 else 0.0,
            'recall_mild': recall[1] if len(recall)>1 else 0.0,
            'recall_moderate': recall[2] if len(recall)>2 else 0.0,
            'recall_severe': recall[3] if len(recall)>3 else 0.0,
            'best_metric': best_f1,
        })
        print(f"Epoch {epoch+1:03d}/{EPOCHS} | lr={lr_curr:.6f} | train loss={tr_loss:.4f} acc={tr_acc:.4f} | val loss={va_loss:.4f} acc={va_acc:.4f} f1={va_f1:.4f} | rec={recall} | best={best_f1:.4f}")

    logger.close()

    if SAVE_CAM:
        save_cam_examples(model, dl_va, device, os.path.join(OUT_DIR, 'cams'), n_samples=NUM_CAM_SAMPLES)
    print('[i] Training finished. Best F1 =', best_f1)


if __name__ == '__main__':
    main()
