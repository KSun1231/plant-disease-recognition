#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task4: Multi-task joint learning & explainable diagnosis (61-way disease + 4-way severity)

AGENTS alignment:
- Globals at top; no CLI args; run to train.
- Two heads: disease(61) and severity(4).
- Log Acc/macro‑F1 for both tasks and save best.
- Severity mapping aligned with Task3.
- Training: AMP, grad clip, cosine schedule, light label smoothing.
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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image, ImageOps
import torch.nn.functional as F

from src.utils.metrics import top1_accuracy, f1_macro
from src.utils.logger import CsvLogger
from src.utils.saver import ensure_dir, save_checkpoint


# ===== Global parameters =====
DATA_ROOT = 'data/ProblemB-Data'
TRAIN_LIST = 'datasets/train_list.txt'
VAL_LIST = 'datasets/val_list.txt'
OUT_DIR = 'outputs/task4_multitask'

NUM_CLASSES = 61
NUM_SEVERITY = 4  # 0=healthy,1=mild,2=moderate,3=severe

MODEL_NAME = 'efficientnetv2_s'  # timm name preferred; fallback to torchvision
PRETRAINED = True

EPOCHS = 30
LR = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
LAMBDA_SEV = 0.5  # loss weight: loss = loss_cls + lambda*loss_sev
AMP = True
GRAD_CLIP_NORM = 1.0
SEED = 2025

IMG_SIZE = 224
RESIZE_SHORT = 256
BATCH_SIZE = 64
NUM_WORKERS = -1
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2

# Optional: save several Grad-CAM visualizations after training
SAVE_CAM = True
NUM_CAM_SAMPLES = 12


# ===== Severity mapping (aligned with Task3) =====
HEALTHY_IDS = {0, 6, 9, 17, 24, 27, 30, 33, 38, 41}
GENERAL_IDS = {
    1, 4, 7, 10, 12, 14, 18, 20, 22, 25, 28, 31,
    34, 36, 39, 42, 44, 52, 54, 56, 58,
}
SERIOUS_IDS = {
    2, 5, 8, 11, 13, 15, 19, 21, 23, 26, 29, 32,
    35, 37, 40, 43, 45, 53, 55, 57, 59,
}
CUSTOM_SEVERITY_MAP: Dict[int, int] = {60: 2}


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


def map_to_severity(y61: int) -> int:
    if y61 in CUSTOM_SEVERITY_MAP:
        return int(CUSTOM_SEVERITY_MAP[y61])
    if y61 in HEALTHY_IDS:
        return 0
    if y61 in GENERAL_IDS:
        return 1
    if y61 in SERIOUS_IDS:
        return 3
    return 2


class MultiTaskDataset(Dataset):
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
        ysev = map_to_severity(int(y61))
        return img, int(y61), int(ysev)


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


class MultiTaskHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, num_severity: int):
        super().__init__()
        self.cls = nn.Linear(in_features, num_classes)
        self.sev = nn.Linear(in_features, num_severity)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cls(x), self.sev(x)


class MultiTaskNet(nn.Module):
    def __init__(self, model_name: str, num_classes: int, num_severity: int, pretrained: bool = True):
        super().__init__()
        self.pool = None
        self.flatten = nn.Flatten()
        in_features = None
        self.backbone = None
        # prefer timm: num_classes=0 to output features
        try:
            import timm  # type: ignore
            self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='avg')
            in_features = getattr(self.backbone, 'num_features', None)
        except Exception:
            self.backbone = None
        if self.backbone is None:
            # fallback: torchvision (prefer EfficientNetV2-S)
            try:
                from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
                weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
                m = efficientnet_v2_s(weights=weights)
                # remove classifier and use GAP features
                if hasattr(m, 'classifier'):
                    clf = m.classifier
                    if isinstance(clf, nn.Sequential) and len(clf)>0 and isinstance(clf[-1], nn.Linear):
                        in_features = clf[-1].in_features
                        m.classifier[-1] = nn.Identity()
                    elif isinstance(clf, nn.Linear):
                        in_features = clf.in_features
                        m.classifier = nn.Identity()
                self.backbone = m
            except Exception:
                # last fallback: resnet50
                from torchvision.models import resnet50, ResNet50_Weights
                m = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
                in_features = m.fc.in_features
                m.fc = nn.Identity()
                self.backbone = m
        if in_features is None:
            # try to find the last Linear's in_features as a fallback
            last_lin = None
            for mod in self.backbone.modules():
                if isinstance(mod, nn.Linear):
                    last_lin = mod
            in_features = getattr(last_lin, 'in_features', 1280)
        self.head = MultiTaskHead(in_features, num_classes, num_severity)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)
        if feats.ndim > 2:
            feats = torch.adaptive_avg_pool2d(feats, (1,1))
            feats = feats.view(feats.size(0), -1)
        logits_cls, logits_sev = self.head(feats)
        return logits_cls, logits_sev


def create_amp():
    try:
        from torch.amp import autocast as autocast_new  # type: ignore
        from torch.amp import GradScaler as GradScalerNew  # type: ignore
        return (lambda enabled: autocast_new('cuda', enabled=enabled)), (lambda enabled: GradScalerNew('cuda', enabled=enabled))
    except Exception:
        from torch.cuda.amp import autocast as autocast_old  # type: ignore
        from torch.cuda.amp import GradScaler as GradScalerOld  # type: ignore
    return (lambda enabled: autocast_old(enabled=enabled)), (lambda enabled: GradScalerOld(enabled=enabled))


def train_one_epoch(model, loader, criterions, optimizer, device, scaler):
    model.train()
    ce_cls, ce_sev = criterions
    loss_m = acc_cls_m = acc_sev_m = 0.0
    n = 0
    autocast_ctx, _ = create_amp()
    use_amp = scaler is not None
    for xb, y61, ysev in loader:
        xb = xb.to(device, non_blocking=True)
        y61 = y61.to(device, non_blocking=True)
        ysev = ysev.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx(enabled=use_amp):
            logits_cls, logits_sev = model(xb)
            loss = ce_cls(logits_cls, y61) + LAMBDA_SEV * ce_sev(logits_sev, ysev)
        if use_amp:
            scaler.scale(loss).backward()
            if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
        with torch.no_grad():
            bsz = xb.size(0)
            loss_m += float(loss.detach().item()) * bsz
            acc_cls_m += top1_accuracy(logits_cls, y61) * bsz
            acc_sev_m += top1_accuracy(logits_sev, ysev) * bsz
            n += bsz
    return loss_m/max(1,n), acc_cls_m/max(1,n), acc_sev_m/max(1,n)


@torch.no_grad()
def evaluate(model, loader, criterions, device, num_classes: int, num_severity: int):
    model.eval()
    ce_cls, ce_sev = criterions
    loss_m = acc_cls_m = acc_sev_m = 0.0
    n = 0
    all_logits_cls = []
    all_targets_cls = []
    all_logits_sev = []
    all_targets_sev = []
    for xb, y61, ysev in loader:
        xb = xb.to(device, non_blocking=True)
        y61 = y61.to(device, non_blocking=True)
        ysev = ysev.to(device, non_blocking=True)
        logits_cls, logits_sev = model(xb)
        loss = ce_cls(logits_cls, y61) + LAMBDA_SEV * ce_sev(logits_sev, ysev)
        bsz = xb.size(0)
        loss_m += float(loss.detach().item()) * bsz
        acc_cls_m += top1_accuracy(logits_cls, y61) * bsz
        acc_sev_m += top1_accuracy(logits_sev, ysev) * bsz
        n += bsz
        all_logits_cls.append(logits_cls.cpu())
        all_targets_cls.append(y61.cpu())
        all_logits_sev.append(logits_sev.cpu())
        all_targets_sev.append(ysev.cpu())
    logits_c = torch.cat(all_logits_cls, 0)
    targets_c = torch.cat(all_targets_cls, 0)
    logits_s = torch.cat(all_logits_sev, 0)
    targets_s = torch.cat(all_targets_sev, 0)
    f1_c = f1_macro(logits_c, targets_c, num_classes)
    f1_s = f1_macro(logits_s, targets_s, num_severity)
    return loss_m/max(1,n), acc_cls_m/max(1,n), acc_sev_m/max(1,n), f1_c, f1_s


def build_dataloaders():
    set_seed(SEED)
    t_train, t_val = build_transforms(IMG_SIZE, RESIZE_SHORT)
    tr = read_list_file(TRAIN_LIST)
    va = read_list_file(VAL_LIST)
    ds_tr = MultiTaskDataset(DATA_ROOT, tr, transform=t_train)
    ds_va = MultiTaskDataset(DATA_ROOT, va, transform=t_val)
    workers = NUM_WORKERS if NUM_WORKERS and NUM_WORKERS > 0 else max(1, os.cpu_count() or 1)
    dl_tr = DataLoader(
        ds_tr,
        batch_size=BATCH_SIZE,
        shuffle=True,
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


# ===== Grad-CAM visualization (based on disease head) =====
def _find_last_conv(m: nn.Module) -> nn.Module:
    last = None
    for mod in m.modules():
        if isinstance(mod, nn.Conv2d):
            last = mod
    if last is None:
        raise RuntimeError('No Conv2d layer found for Grad-CAM')
    return last


def _grad_cam_disease(model: nn.Module, x: torch.Tensor, target_layer: nn.Module) -> torch.Tensor:
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
        logits_cls, _ = model(x)
        cls = logits_cls.argmax(1)
        score = logits_cls.gather(1, cls.view(-1, 1)).sum()
        model.zero_grad(set_to_none=True)
        score.backward()
        w = grads.mean(dim=(2, 3), keepdim=True)
        cam = (w * feats).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam -= cam.amin(dim=(2, 3), keepdim=True)
        cam /= (cam.amax(dim=(2, 3), keepdim=True) + 1e-6)
        return cam
    finally:
        h1.remove(); h2.remove()


def _save_cam_examples(model, dl_val, device, out_dir: str, n_samples: int = 12) -> None:
    ensure_dir(out_dir)
    layer = _find_last_conv(model)
    saved = 0
    for xb, _, _ in dl_val:
        xb = xb.to(device)
        cam = _grad_cam_disease(model, xb, layer)  # [B,1,Hf,Wf]
        # upsample to input size
        try:
            cam = F.interpolate(cam, size=xb.shape[2:], mode='bilinear', align_corners=False)
        except Exception:
            pass
        for i in range(min(xb.size(0), n_samples - saved)):
            import torchvision.utils as vutils
            show = xb[i].detach().cpu()
            show = (show * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)).clamp(0, 1)
            heat = cam[i].expand_as(show).cpu()
            mix = (show * 0.6 + heat * 0.4).clamp(0, 1)
            vutils.save_image(mix, os.path.join(out_dir, f'cam_{saved:03d}.png'))
            saved += 1
            if saved >= n_samples:
                return


def main():
    ensure_dir(OUT_DIR)
    ds_tr, ds_va, dl_tr, dl_va = build_dataloaders()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MultiTaskNet(MODEL_NAME, NUM_CLASSES, NUM_SEVERITY, PRETRAINED)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion_cls = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    criterion_sev = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    criterions = (criterion_cls, criterion_sev)
    _, create_scaler = create_amp()
    scaler = create_scaler(AMP)

    from torch.optim.lr_scheduler import LambdaLR
    import math
    def lr_lambda(curr_epoch: int):
        if curr_epoch < 3:
            return max(1e-8, (curr_epoch+1)/3)
        t = (curr_epoch-3)/max(1,(EPOCHS-3))
        cos = 0.5*(1+math.cos(math.pi*t))
        return float((1e-6/LR) + (1 - (1e-6/LR))*cos)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    logger = CsvLogger(os.path.join(OUT_DIR, 'train_log.csv'), fieldnames=[
        'epoch','lr','train_loss','train_acc_cls','train_acc_sev','val_loss','val_acc_cls','val_acc_sev','val_f1_cls','val_f1_sev','best_metric'
    ])
    best_metric = -1.0

    for epoch in range(EPOCHS):
        tr_loss, tr_acc_c, tr_acc_s = train_one_epoch(model, dl_tr, criterions, optimizer, device, scaler)
        va_loss, va_acc_c, va_acc_s, f1_c, f1_s = evaluate(model, dl_va, criterions, device, NUM_CLASSES, NUM_SEVERITY)
        scheduler.step()
        lr_curr = optimizer.param_groups[0]['lr']
        # use average of two F1s as best metric
        metric = 0.5*(f1_c + f1_s)
        if metric > best_metric:
            best_metric = metric
            save_checkpoint(os.path.join(OUT_DIR, 'best.pth'), {
                'epoch': epoch+1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': {
                    'num_classes': NUM_CLASSES, 'num_severity': NUM_SEVERITY, 'model': MODEL_NAME
                }
            })
        logger.log({
            'epoch': epoch+1,
            'lr': lr_curr,
            'train_loss': tr_loss,
            'train_acc_cls': tr_acc_c,
            'train_acc_sev': tr_acc_s,
            'val_loss': va_loss,
            'val_acc_cls': va_acc_c,
            'val_acc_sev': va_acc_s,
            'val_f1_cls': f1_c,
            'val_f1_sev': f1_s,
            'best_metric': best_metric,
        })
        print(f"Epoch {epoch+1:03d}/{EPOCHS} | lr={lr_curr:.6f} | train loss={tr_loss:.4f} acc_cls={tr_acc_c:.4f} acc_sev={tr_acc_s:.4f} | val loss={va_loss:.4f} acc_cls={va_acc_c:.4f} acc_sev={va_acc_s:.4f} f1_cls={f1_c:.4f} f1_sev={f1_s:.4f} | best={best_metric:.4f}")

    logger.close()
    # optionally save Grad-CAM examples
    if SAVE_CAM:
        _save_cam_examples(model, dl_va, device, os.path.join(OUT_DIR, 'cams'), n_samples=NUM_CAM_SAMPLES)
    print('[i] Training finished. Best joint metric (avg F1) =', best_metric)


if __name__ == '__main__':
    set_seed(SEED)
    main()
