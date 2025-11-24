# About the `datasets` folder

This folder stores training/validation lists and their derivatives:

- `train_list.txt` / `val_list.txt`: active train/val lists (`path label`, relative to `data/ProblemB-Data`).
- `train_list.cleaned.txt` / `val_list.cleaned.txt`: lists produced by the latest cleaning run (backup/reference before overwriting).
- `train_list.removed.txt` / `val_list.removed.txt`: removed samples and reasons (missing/corrupt/small/duplicate, etc.).
- `cross_split_duplicates.txt`: duplicate pairs across train/val (based on dHash/MD5), for leakage checks.
- `missing_train.txt` / `missing_val.txt`: entries that could not be matched to actual files when rebuilding lists from JSON.

Conventions:
- Line format: `relative_path (relative to data/ProblemB-Data) + space + integer label`.
- Path separator is `/`, line ending is `LF`.
- Use `train_list.txt` and `val_list.txt` as the entrypoint lists for training/validation (they are overwritten by the latest cleaning run).
