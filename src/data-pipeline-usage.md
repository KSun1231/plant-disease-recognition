# Training Data Pipeline Guide

This guide explains how to use the PyTorch training data pipeline scaffold under `src/`, including dataset loading, building transforms, assembling DataLoaders, and a simple preview script.

## Layout
- Datasets and lists:
  - Root: `data/ProblemB-Data`
  - Lists: `datasets/train_list.txt`, `datasets/val_list.txt`
- Code:
  - Dataset: `src/datamodules/list_dataset.py`
  - Transforms: `src/transforms/build.py`
  - Dataloaders: `src/dataloaders/build.py`
  - Preview script: `src/tools/preview_dataloader.py`

## Dependencies
- Python 3.8+
- PyTorch (CPU or GPU)
- torchvision (for common image augmentations and tensor conversion)

## Quickstart
1) Preview a batch:
```bash
python3 src/tools/preview_dataloader.py \
  --root data/ProblemB-Data \
  --train-list datasets/train_list.txt \
  --val-list datasets/val_list.txt \
  --batch-size 16 --workers 2 --img-size 224
```

2) Integrate in a training script:
```python
from src.dataloaders.build import DataConfig, build_dataloaders

cfg = DataConfig(
    root='data/ProblemB-Data',
    train_list='datasets/train_list.txt',
    val_list='datasets/val_list.txt',
    img_size=224,
    batch_size=64,
    num_workers=4,
    class_balancing=True,  # enable class-balanced sampling if needed
)
(ds_train, ds_val), (dl_train, dl_val), info = build_dataloaders(cfg)
print(info)
for xb, yb in dl_train:
    # training loop
    pass
```

## Design Highlights
- Data consistency: apply EXIF transpose and convert to RGB during loading; samples with min side < 16 are dropped (or raise if strict mode).
- Train-time augmentation: RandomResizedCrop, horizontal flip, color jitter, light rotation, ImageNet normalization, and optional RandomErasing.
- Validation: resize short side to 256 + CenterCrop to 224, with ImageNet normalization.
- Sampling balance: optional `WeightedRandomSampler` with `sqrt`/`log` weighting and mean-normalization for stability.
- Reproducibility: set global and DataLoader worker seeds.

## FAQ
- torchvision not installed: please install the matching torchvision version first.
- List path prefix: paths in lists are relative to `data/ProblemB-Data`, e.g. `AgriculturalDisease_trainingset/images/1_0.jpg 1`.
- Train strictness: to raise on abnormal samples, pass `strict=True` when constructing the training dataset.

## Future Extensions
- Add RandAugment/AutoAugment scheduling.
- Add batch-level MixUp/CutMix and the corresponding collate_fn.
- Include image IDs/paths in dataset returns for easier debugging and explainable visualization.
