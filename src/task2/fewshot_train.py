#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task2: Few-shot crop disease recognition (61 classes, 10 shots per class)

AGENTS alignment:
- No CLI args; globals configured at top; run the file to train.
- Transfer learning + two-stage fine-tuning (linear probe → unfreeze last blocks).
- Model size <20M: default DeiT‑Tiny (~5.7M params); fallback to EfficientNet‑B0 if timm is unavailable.

Training highlights:
- Build a 10‑shot training subset (max 10 per class; random but reproducible).
- Strong augmentation (RandAugment + flip + color jitter + light rotation).
- Linear probe: train head only; then unfreeze last blocks and fine‑tune with a smaller LR.
- Optional attention distillation: teacher = Task1 EfficientNetV2‑S.
- Metrics: Top‑1 Acc and macro‑F1 across 61 classes.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
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

from src.utils.metrics import top1_accuracy, f1_macro
from src.utils.logger import CsvLogger
from src.utils.saver import ensure_dir, save_checkpoint
from src.dataloaders.build import compute_class_weights as compute_class_weights_task1


# ===== Global configurable parameters =====
DATA_ROOT = 'data/ProblemB-Data'
TRAIN_LIST = 'data/ProblemB-Data/AgriculturalDisease_trainingset/train_list.txt'
VAL_LIST = 'data/ProblemB-Data/AgriculturalDisease_validationset/ttest_list.txt'
OUT_DIR = 'outputs/task2_deit_tiny_10shot'

NUM_CLASSES = 61
SHOTS_PER_CLASS = 10
SEED = 2025

MODEL_NAME = 'deit_tiny_patch16_224'  # default DeiT-Tiny (~5.7M), within <20M
PRETRAINED = True
DEIT_DROP_PATH = 0.1  # DeiT uses drop_path (not drop_rate)

# Training hyperparameters
EPOCHS_TOTAL = 30
EPOCHS_HEAD = 5           # linear probe stage
LR_HEAD = 1e-2            # higher LR for probe
LR_FT = 5e-5              # smaller LR for fine-tune (DeiT)
WEIGHT_DECAY = 5e-4       # stronger weight decay to regularize
LABEL_SMOOTHING = 0.05
AMP = True
GRAD_CLIP_NORM = 1.0

# DataLoader
IMG_SIZE = 224
RESIZE_SHORT = 256
BATCH_SIZE = 32            # batch 32 is stabler
NUM_WORKERS = -1           # -1 means use all CPU cores
PIN_MEMORY = True

# Optional: attention distillation (see Task2_DeiT notes)
# Use Task1 EfficientNetV2‑S weights as teacher
DISTILLATION = True
TEACHER_FROM_TASK1 = True
TEACHER_TASK1_WEIGHTS = 'outputs/task1_effv2s/best.pth'
TEACHER_MODEL_NAME = 'efficientnetv2_s'  # used only when TEACHER_FROM_TASK1=False
TEACHER_PRETRAINED = True

# Distillation weights (loss = alpha*CE + (1-alpha)*KD)
# In few-shot + strong aug, KD can dominate; raise CE and temperature for stability.
DISTILL_ALPHA = 0.7          # higher CE weight (was 0.5)
DISTILL_TEMPERATURE = 4.0    # higher T for softer targets

# Distillation flow control
KD_IN_HEAD_STAGE = False      # enable KD in probe? keep off for stability
KD_WARMUP_EPOCHS = 2          # fine-tune warmup epochs without KD (CE only)

# Teacher Logit-Adjusted calibration (mitigate long-tail bias):
# If Task1 training enabled LA (configs/task1.yaml default true), add the same bias to teacher logits during KD.
TEACHER_ADD_LOGIT_ADJUST = True
TEACHER_LA_TAU = 1.0
TEACHER_PRIOR_LIST = 'datasets/train_list_rebalanced.txt'  # prefer rebalanced; fallback datasets/train_list.txt


# Optional (align with Task1): apply Logit-Adjusted bias to student logits as well
STUDENT_ADD_LOGIT_ADJUST = True
STUDENT_LA_TAU = 1.0
STUDENT_PRIOR_LIST = 'datasets/train_list_rebalanced.txt'
STUDENT_EVAL_LA = True  # also apply bias at evaluation

# Optional (align with Task1): class-weighted CE (few-shot is often balanced; keep the switch for API parity)
CLASS_WEIGHTING = 'none'  # 'none' | 'sqrt' | 'log' | 'inverse'
CLASS_WEIGHT_LOG_ALPHA = 1.02


# ===== Utilities & dataset =====
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


def build_fewshot_entries(all_entries: Sequence[Tuple[str, int]], shots: int, seed: int) -> List[Tuple[str, int]]:
    """Randomly sample a few‑shot subset per class from the full list."""
    by_cls: Dict[int, List[str]] = {}
    for rel, y in all_entries:
        by_cls.setdefault(int(y), []).append(rel)
    rng = random.Random(seed)
    few: List[Tuple[str, int]] = []
    for c, rels in by_cls.items():
        rng.shuffle(rels)
        take = min(shots, len(rels))
        for r in rels[:take]:
            few.append((r, c))
    return few


class EntriesDataset(Dataset):
    def __init__(self, root: str, entries: Sequence[Tuple[str, int]], transform=None, *, min_dim: int = 16):
        self.root = root
        self.entries = list(entries)
        self.t = transform
        self.min_dim = min_dim

    def __len__(self) -> int:
        return len(self.entries)

    def _load_image(self, abs_path: str) -> Image.Image:
        img = Image.open(abs_path)
        img.load()
        img = ImageOps.exif_transpose(img)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img

    def __getitem__(self, i: int):
        rel, y = self.entries[i]
        abs_path = os.path.join(self.root, rel)
        img = self._load_image(abs_path)
        w, h = img.size
        if w < self.min_dim or h < self.min_dim:
            raise ValueError(f'small: {w}x{h}')
        if self.t is not None:
            img = self.t(img)
        return img, int(y)


def build_transforms(img_size: int = 224, resize_short: int = 256):
    # training aug: RandAugment + jitter + light rotation + RRC + norm
    t_train = T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(3/4, 4/3)),
        T.RandAugment(num_ops=2, magnitude=9),
        T.RandomHorizontalFlip(0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.RandomRotation(degrees=15),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    t_val = T.Compose([
        T.Resize(resize_short),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return t_train, t_val


def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    model = None
    try:
        import timm  # type: ignore
        if model_name.startswith('deit'):
            # DeiT uses drop_path_rate
            model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, drop_path_rate=DEIT_DROP_PATH)
        else:
            model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    except Exception:
        # fallback: torchvision efficientnet_b0
        try:
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            model = efficientnet_b0(weights=weights)
            # replace classifier head
            if hasattr(model, 'classifier'):
                clf = model.classifier
                if isinstance(clf, nn.Sequential) and len(clf)>0 and isinstance(clf[-1], nn.Linear):
                    in_f = clf[-1].in_features
                    clf[-1] = nn.Linear(in_f, num_classes)
                elif isinstance(clf, nn.Linear):
                    in_f = clf.in_features
                    model.classifier = nn.Linear(in_f, num_classes)
            elif hasattr(model, 'fc'):
                in_f = model.fc.in_features
                model.fc = nn.Linear(in_f, num_classes)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError('Failed to build model; please install timm or torchvision') from e
    # param count check
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[i] Model parameters: {n_params/1e6:.2f}M')
    if n_params > 20_000_000:
        print('[w] Parameters exceed 20M; please switch to a smaller model.')
    return model


def build_teacher(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build teacher model (forward only for distillation; no grads)."""
    try:
        import timm  # type: ignore
        m = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        return m
    except Exception:
        try:
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            w = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            m = efficientnet_b0(weights=w)
            if hasattr(m, 'classifier'):
                clf = m.classifier
                if isinstance(clf, nn.Sequential) and len(clf)>0 and isinstance(clf[-1], nn.Linear):
                    in_f = clf[-1].in_features
                    clf[-1] = nn.Linear(in_f, num_classes)
                elif isinstance(clf, nn.Linear):
                    in_f = clf.in_features
                    m.classifier = nn.Linear(in_f, num_classes)
            elif hasattr(m, 'fc'):
                in_f = m.fc.in_features
                m.fc = nn.Linear(in_f, num_classes)
            return m
        except Exception as e:  # noqa: BLE001
            raise RuntimeError('Failed to build teacher model; please install timm or torchvision') from e


def build_teacher_from_task1(num_classes: int, weights_path: str) -> nn.Module:
    """Use Task1 EfficientNetV2‑S as teacher and load its best weights."""
    from src.models.build_model import build_model as build_task1_model
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f'Cannot find Task1 teacher weights: {weights_path}')
    m = build_task1_model(model_name='efficientnetv2_s', num_classes=num_classes, pretrained=False, drop_rate=0.0)
    state = torch.load(weights_path, map_location='cpu')
    sd = state.get('model', state)
    m.load_state_dict(sd, strict=False)
    return m


def compute_logit_adjust_bias_from_list(list_path: str, num_classes: int, tau: float = 1.0):
    """Compute LA bias from a list; returns Tensor [num_classes] or None on failure."""
    try:
        entries = read_list_file(list_path)
        if not entries:
            return None
        from collections import Counter
        cnt = Counter([int(lab) for _, lab in entries])
        import torch as _torch
        prior = _torch.full((num_classes,), 1e-12, dtype=_torch.float)
        for k, v in cnt.items():
            if 0 <= int(k) < num_classes:
                prior[int(k)] = float(v)
        prior = prior / prior.sum()
        return float(tau) * _torch.log(prior + 1e-12)
    except Exception:
        return None


def distillation_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 3.0) -> torch.Tensor:
    """KL distillation term: KL(softmax_t(teacher) || log_softmax_t(student)) * T^2."""
    import torch.nn.functional as F
    T = float(max(1e-6, temperature))
    log_p = F.log_softmax(student_logits / T, dim=1)
    q = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(log_p, q, reduction='batchmean') * (T * T)


def find_classifier_params(model: nn.Module) -> List[nn.Parameter]:
    # prefer common attributes
    if hasattr(model, 'head') and isinstance(getattr(model, 'head'), nn.Linear):
        return list(getattr(model, 'head').parameters())
    if hasattr(model, 'classifier'):
        clf = getattr(model, 'classifier')
        if isinstance(clf, nn.Linear):
            return list(clf.parameters())
        if isinstance(clf, nn.Sequential):
            # find last Linear
            lin = None
            for m in clf:
                if isinstance(m, nn.Linear):
                    lin = m
            return list(lin.parameters()) if lin else list(clf.parameters())
    if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
        return list(model.fc.parameters())
    # fallback: find last Linear
    last = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last = m
    return list(last.parameters()) if last is not None else []


def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def unfreeze_norm_layers(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (nn.LayerNorm,)):
            for p in m.parameters():
                p.requires_grad_(True)

def unfreeze_last_blocks(model: nn.Module) -> None:
    # Compatible with ViT/DeiT and EfficientNet: try blocks attribute first
    # DeiT-Tiny has 12 blocks; unfreeze last 2
    if hasattr(model, 'blocks') and isinstance(getattr(model, 'blocks'), (list, nn.ModuleList)):
        try:
            blocks = getattr(model, 'blocks')
            for p in blocks[-2:].parameters():
                p.requires_grad_(True)
            # common LayerNorm / head
            if hasattr(model, 'head'):
                for p in getattr(model, 'head').parameters():
                    p.requires_grad_(True)
            if hasattr(model, 'norm'):
                for p in getattr(model, 'norm').parameters():
                    p.requires_grad_(True)
            return
        except Exception:
            pass
    # EfficientNet fallback: unfreeze tail conv and bn
    if hasattr(model, 'blocks'):
        try:
            for p in model.blocks[-2:].parameters():
                p.requires_grad_(True)
            if hasattr(model, 'conv_head'):
                for p in model.conv_head.parameters():
                    p.requires_grad_(True)
            if hasattr(model, 'bn2'):
                for p in model.bn2.parameters():
                    p.requires_grad_(True)
            return
        except Exception:
            pass
    # Generic fallback: unfreeze last two children
    children = list(model.children())
    if len(children) >= 2:
        for child in children[-3:-1]:
            for p in child.parameters():
                p.requires_grad_(True)


def build_optim(params, lr: float, wd: float):
    return optim.AdamW(params, lr=lr, weight_decay=wd)


def cosine_scheduler(optimizer, total_epochs: int, warmup_epochs: int, base_lr: float, min_lr: float):
    from torch.optim.lr_scheduler import LambdaLR
    import math

    def lr_lambda(curr_epoch: int):
        if curr_epoch < warmup_epochs:
            return max(1e-8, (curr_epoch + 1) / max(1, warmup_epochs))
        t = (curr_epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
        cos = 0.5 * (1 + math.cos(math.pi * t))
        scale = (min_lr / base_lr) + (1 - (min_lr / base_lr)) * cos
        return float(scale)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, *, grad_clip: float | None = None,
                   teacher: nn.Module | None = None, distill_alpha: float = 0.5, distill_T: float = 3.0,
                   teacher_bias: torch.Tensor | None = None, kd_enabled: bool = True,
                   student_bias: torch.Tensor | None = None):
    model.train()
    loss_meter = 0.0
    acc_meter = 0.0
    n = 0
    use_amp = scaler is not None
    # AMP context
    try:
        from torch.amp import autocast as autocast_new  # type: ignore
        amp_ctx = lambda enabled: autocast_new('cuda', enabled=enabled)
    except Exception:
        from torch.cuda.amp import autocast as autocast_old  # type: ignore
        amp_ctx = lambda enabled: autocast_old(enabled=enabled)

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with amp_ctx(enabled=use_amp):
            logits = model(xb)
            if student_bias is not None:
                logits = logits + student_bias.to(logits.device)
            if teacher is not None and kd_enabled:
                with torch.no_grad():
                    teacher.eval()
                    t_logits = teacher(xb)
                    if teacher_bias is not None:
                        # apply LA bias to teacher logits (broadcast add)
                        t_logits = t_logits + teacher_bias.to(t_logits.device)
                ce = criterion(logits, yb)
                kd = distillation_kl(logits, t_logits, temperature=distill_T)
                alpha = float(min(max(distill_alpha, 0.0), 1.0))
                loss = alpha * ce + (1.0 - alpha) * kd
            else:
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
            loss_meter += float(loss.detach().item()) * bsz
            acc_meter += top1_accuracy(logits, yb) * bsz
            n += bsz
    return loss_meter / max(1, n), acc_meter / max(1, n)


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes: int, *, student_bias: torch.Tensor | None = None):
    model.eval()
    loss_meter = 0.0
    acc_meter = 0.0
    n = 0
    all_logits = []
    all_targets = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        if student_bias is not None:
            logits = logits + student_bias.to(logits.device)
        loss = criterion(logits, yb)
        bsz = xb.size(0)
        loss_meter += float(loss.detach().item()) * bsz
        acc_meter += top1_accuracy(logits, yb) * bsz
        n += bsz
        all_logits.append(logits.cpu())
        all_targets.append(yb.cpu())
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    f1m = f1_macro(logits, targets, num_classes)
    return loss_meter / max(1, n), acc_meter / max(1, n), f1m


def main():
    set_seed(SEED)
    ensure_dir(OUT_DIR)

    # read full list and build 10‑shot train subset
    all_train = read_list_file(TRAIN_LIST)
    few_train = build_fewshot_entries(all_train, SHOTS_PER_CLASS, SEED)
    val_entries = read_list_file(VAL_LIST)

    print(f'[i] few‑shot train samples: {len(few_train)} (≤{SHOTS_PER_CLASS} per class)')
    print(f'[i] original val samples: {len(val_entries)}')

    t_train, t_val = build_transforms(IMG_SIZE, RESIZE_SHORT)

    ds_train = EntriesDataset(DATA_ROOT, few_train, transform=t_train)
    ds_val = EntriesDataset(DATA_ROOT, val_entries, transform=t_val)

    workers_eff = NUM_WORKERS if NUM_WORKERS and NUM_WORKERS > 0 else max(1, os.cpu_count() or 1)
    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=workers_eff, pin_memory=PIN_MEMORY)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=workers_eff, pin_memory=PIN_MEMORY)

    # model & optimizer
    model = build_model(MODEL_NAME, NUM_CLASSES, PRETRAINED)
    # optional teacher
    teacher = None
    if DISTILLATION:
        try:
            if TEACHER_FROM_TASK1:
                teacher = build_teacher_from_task1(NUM_CLASSES, TEACHER_TASK1_WEIGHTS)
            else:
                teacher = build_teacher(TEACHER_MODEL_NAME, NUM_CLASSES, pretrained=TEACHER_PRETRAINED)
        except Exception as e:  # noqa: BLE001
            print('[w] Failed to build teacher model; disabling distillation:', e)
            teacher = None

    # teacher LA bias (optional)
    teacher_bias = None
    if DISTILLATION and TEACHER_ADD_LOGIT_ADJUST and teacher is not None:
        la_list = TEACHER_PRIOR_LIST if os.path.isfile(TEACHER_PRIOR_LIST) else 'datasets/train_list.txt'
        bias = compute_logit_adjust_bias_from_list(la_list, NUM_CLASSES, tau=TEACHER_LA_TAU)
        if bias is not None:
            teacher_bias = bias
            print(f'[i] Distillation uses teacher LA bias (tau={TEACHER_LA_TAU}); prior from: {la_list}')
        else:
            print('[w] Failed to compute teacher LA bias; continue without bias.')
    # student LA bias (align with Task1; on by default)
    student_bias = None
    if STUDENT_ADD_LOGIT_ADJUST:
        la_list_s = STUDENT_PRIOR_LIST if os.path.isfile(STUDENT_PRIOR_LIST) else 'datasets/train_list.txt'
        sb = compute_logit_adjust_bias_from_list(la_list_s, NUM_CLASSES, tau=STUDENT_LA_TAU)
        if sb is not None:
            student_bias = sb
            print(f'[i] Student uses LA bias (tau={STUDENT_LA_TAU}); prior from: {la_list_s}')
        else:
            print('[w] Failed to compute student LA bias; continue without bias.')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    if teacher is not None:
        teacher.to(device)
        teacher.eval()

    # linear probe: freeze backbone, train head
    freeze_all(model)
    head_params = find_classifier_params(model)
    for p in head_params:
        p.requires_grad_(True)
    # also unfreeze LayerNorm during probe
    unfreeze_norm_layers(model)
    # optimize only parameters with requires_grad=True
    probe_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optim(probe_params, LR_HEAD, WEIGHT_DECAY)
    # Align with Task1: optional class-weighted CE (few-shot often balanced; keep API)
    ce_weight = None
    if CLASS_WEIGHTING.lower() in {'sqrt','log','inverse'}:
        try:
            labels_fs = [lab for _, lab in few_train]
            scheme = 'sqrt' if CLASS_WEIGHTING=='sqrt' else ('log' if CLASS_WEIGHTING=='log' else 'inverse')
            w_dict = compute_class_weights_task1(labels_fs, scheme=scheme, log_alpha=CLASS_WEIGHT_LOG_ALPHA)
            w = torch.ones(NUM_CLASSES, dtype=torch.float)
            for c, val in w_dict.items():
                if 0 <= int(c) < NUM_CLASSES:
                    w[int(c)] = float(val)
            ce_weight = w.to(device)
            print(f'[i] Few-shot class weighting ({scheme}) enabled; mean≈{float(w.mean()):.3f}')
        except Exception as e:
            print('[w] Failed to compute few-shot class weights; falling back to none.', e)
            ce_weight = None
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING, weight=ce_weight)
    try:
        from torch.amp import GradScaler as GradScalerNew  # type: ignore
        scaler = GradScalerNew('cuda', enabled=AMP)
    except Exception:
        from torch.cuda.amp import GradScaler as GradScalerOld  # type: ignore
        scaler = GradScalerOld(enabled=AMP)

    logger = CsvLogger(os.path.join(OUT_DIR, 'train_log.csv'), fieldnames=['epoch','phase','lr','train_loss','train_acc','val_loss','val_acc','val_f1_macro','best_metric'])
    best_f1 = -1.0

    # Stage 1: linear probe
    # add a light cosine scheduler (warmup=1)
    sched_head = cosine_scheduler(optimizer, total_epochs=EPOCHS_HEAD, warmup_epochs=1, base_lr=LR_HEAD, min_lr=max(1e-6, LR_HEAD*1e-2)) if EPOCHS_HEAD>0 else None
    for epoch in range(EPOCHS_HEAD):
        lr_curr = optimizer.param_groups[0]['lr']
        # keep KD off in probe for stability
        tr_loss, tr_acc = train_one_epoch(
            model, dl_train, criterion, optimizer, device, scaler,
            grad_clip=GRAD_CLIP_NORM,
            teacher=(teacher if (DISTILLATION and KD_IN_HEAD_STAGE and teacher is not None) else None),
            distill_alpha=DISTILL_ALPHA,
            distill_T=DISTILL_TEMPERATURE,
            teacher_bias=teacher_bias,
            kd_enabled=bool(DISTILLATION and KD_IN_HEAD_STAGE),
            student_bias=(student_bias if STUDENT_ADD_LOGIT_ADJUST else None),
        )
        va_loss, va_acc, va_f1 = evaluate(model, dl_val, criterion, device, NUM_CLASSES, student_bias=(student_bias if STUDENT_EVAL_LA else None))
        if sched_head is not None:
            sched_head.step()
        if va_f1 > best_f1:
            best_f1 = va_f1
            save_checkpoint(os.path.join(OUT_DIR, 'best.pth'), {'epoch': epoch+1, 'model': model.state_dict(), 'config': {
                'shots': SHOTS_PER_CLASS, 'model': MODEL_NAME, 'seed': SEED
            }})
        logger.log({'epoch': epoch+1, 'phase': 'head', 'lr': lr_curr, 'train_loss': tr_loss, 'train_acc': tr_acc,
                    'val_loss': va_loss, 'val_acc': va_acc, 'val_f1_macro': va_f1, 'best_metric': best_f1})
        print(f'[head] Epoch {epoch+1:02d}/{EPOCHS_HEAD} | lr={lr_curr:.6f} | train loss={tr_loss:.4f} acc={tr_acc:.4f} | val loss={va_loss:.4f} acc={va_acc:.4f} f1={va_f1:.4f} | best={best_f1:.4f}')

    # Stage 2: unfreeze tail blocks and fine‑tune (low LR)
    unfreeze_last_blocks(model)
    # optimize current requires_grad=True parameters
    ft_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optim(ft_params, LR_FT, WEIGHT_DECAY)
    sched = cosine_scheduler(optimizer, total_epochs=EPOCHS_TOTAL - EPOCHS_HEAD, warmup_epochs=5, base_lr=LR_FT, min_lr=1e-6)

    for i in range(EPOCHS_TOTAL - EPOCHS_HEAD):
        epoch = EPOCHS_HEAD + i + 1
        lr_curr = optimizer.param_groups[0]['lr']
        # during fine‑tune, skip KD for KD_WARMUP_EPOCHS and use CE only; then enable KD
        use_kd_now = bool(DISTILLATION and (i >= max(0, int(KD_WARMUP_EPOCHS)))) and (teacher is not None)
        tr_loss, tr_acc = train_one_epoch(
            model, dl_train, criterion, optimizer, device, scaler,
            grad_clip=GRAD_CLIP_NORM,
            teacher=(teacher if use_kd_now else None),
            distill_alpha=DISTILL_ALPHA,
            distill_T=DISTILL_TEMPERATURE,
            teacher_bias=teacher_bias,
            kd_enabled=use_kd_now,
            student_bias=(student_bias if STUDENT_ADD_LOGIT_ADJUST else None),
        )
        va_loss, va_acc, va_f1 = evaluate(model, dl_val, criterion, device, NUM_CLASSES, student_bias=(student_bias if STUDENT_EVAL_LA else None))
        sched.step()
        if va_f1 > best_f1:
            best_f1 = va_f1
            save_checkpoint(os.path.join(OUT_DIR, 'best.pth'), {'epoch': epoch, 'model': model.state_dict(), 'config': {
                'shots': SHOTS_PER_CLASS, 'model': MODEL_NAME, 'seed': SEED
            }})
        logger.log({'epoch': epoch, 'phase': 'ft', 'lr': lr_curr, 'train_loss': tr_loss, 'train_acc': tr_acc,
                    'val_loss': va_loss, 'val_acc': va_acc, 'val_f1_macro': va_f1, 'best_metric': best_f1})
        print(f'[ ft ] Epoch {epoch:02d}/{EPOCHS_TOTAL} | lr={lr_curr:.6f} | train loss={tr_loss:.4f} acc={tr_acc:.4f} | val loss={va_loss:.4f} acc={va_acc:.4f} f1={va_f1:.4f} | best={best_f1:.4f}')

    logger.close()
    print('[i] Training finished; best macro F1 =', best_f1)


if __name__ == '__main__':
    main()
