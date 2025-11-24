#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
import os
from typing import List


DEFAULT_NAMES: List[str] = [
    "Apple Healthy",
    "Apple Scab (General)",
    "Apple Scab (Serious)",
    "Apple Frogeye Spot",
    "Cedar Apple Rust (General)",
    "Cedar Apple Rust (Serious)",
    "Cherry Healthy",
    "Cherry Powdery Mildew (General)",
    "Cherry Powdery Mildew (Serious)",
    "Corn Healthy",
    "Cercospora Zeaemaydis (General)",
    "Cercospora Zeaemaydis (Serious)",
    "Grape Healthy",
    "Grape Black Rot (General)",
    "Grape Black Rot (Serious)",
]


def gaussian_counts(n: int, peak: int = 1000, sigma_frac: float = 0.25) -> List[int]:
    mu = (n - 1) / 2.0
    sigma = max(1.0, sigma_frac * n)
    out: List[int] = []
    for i in range(n):
        x = (i - mu) / sigma
        v = peak * math.exp(-0.5 * x * x)
        out.append(max(1, int(round(v))))
    return out


def draw_with_matplotlib(names: List[str], counts: List[int], out_path: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt  # type: ignore

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.8), 5))
    xs = list(range(len(names)))
    ax.bar(xs, counts, color="#5DADE2", alpha=0.7, label="Count")
    # Overlay curve shape (normalized line) for a Gaussian look
    m = max(counts)
    ax.plot(xs, [c for c in counts], color="#1F618D", linewidth=2, label="Gaussian shape")
    ax.set_title("Synthetic Gaussian Distribution of Disease Counts")
    ax.set_xlabel("Disease (Name)")
    ax.set_ylabel("Count")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def draw_with_pil(names: List[str], counts: List[int], out_path: str):
    from PIL import Image, ImageDraw, ImageFont

    n = len(names)
    W = max(800, 80 * n)
    H = 500
    padding = 60
    img = Image.new('RGB', (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # fonts
    try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
        font_tick = ImageFont.truetype("DejaVuSans.ttf", 12)
    except Exception:
        font_title = ImageFont.load_default()
        font_tick = ImageFont.load_default()

    # axes area
    left = padding
    right = W - padding
    top = padding
    bottom = H - padding - 40
    # axes lines
    draw.line((left, bottom, right, bottom), fill=(0, 0, 0), width=2)
    draw.line((left, bottom, left, top), fill=(0, 0, 0), width=2)

    # y ticks
    ymax = max(counts)
    y_steps = 5
    for k in range(y_steps + 1):
        yv = int(round(ymax * k / y_steps))
        y = bottom - (bottom - top) * k / y_steps
        draw.line((left - 5, int(y), right, int(y)), fill=(220, 220, 220), width=1)
        draw.text((left - 55, int(y) - 8), str(yv), fill=(0, 0, 0), font=font_tick)

    # bars and polyline
    bw = (right - left) / max(1, n)
    pts = []
    for i, c in enumerate(counts):
        cx = left + (i + 0.5) * bw
        h = (c / ymax) * (bottom - top)
        draw.rectangle((cx - bw * 0.35, bottom - h, cx + bw * 0.35, bottom), fill=(93, 173, 226))
        pts.append((cx, bottom - h))
    draw.line(pts, fill=(31, 97, 141), width=2)

    # x labels (tilted)
    for i, name in enumerate(names):
        cx = left + (i + 0.5) * bw
        txt = name
        # tilt text (simple: render and rotate)
        tw_img = Image.new('RGBA', (300, 60), (255, 255, 255, 0))
        td = ImageDraw.Draw(tw_img)
        td.text((0, 0), txt, font=font_tick, fill=(0, 0, 0))
        tw_img = tw_img.rotate(45, expand=1)
        img.paste(tw_img, (int(cx - 20), bottom + 5), tw_img)

    # title and axis labels
    draw.text((left, 15), "Synthetic Gaussian Distribution of Disease Counts", fill=(0, 0, 0), font=font_title)
    draw.text((right - 80, H - 25), "Disease (Name)", fill=(0, 0, 0), font=font_tick)
    draw.text((5, top), "Count", fill=(0, 0, 0), font=font_tick)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='outputs/normal_distribution_diseases.png', help='output image path')
    ap.add_argument('--names-file', default=None, help='optional text file with one disease name per line')
    ap.add_argument('--peak', type=int, default=1000, help='peak count of the Gaussian')
    ap.add_argument('--sigma-frac', type=float, default=0.25, help='sigma as fraction of category count')
    args = ap.parse_args()

    names: List[str]
    if args.names_file and os.path.exists(args.names_file):
        with open(args.names_file, 'r', encoding='utf-8', errors='ignore') as f:
            names = [line.strip() for line in f if line.strip()]
    else:
        names = DEFAULT_NAMES

    counts = gaussian_counts(len(names), peak=args.peak, sigma_frac=args.sigma_frac)

    # Prefer matplotlib; fallback to PIL if unavailable
    try:
        draw_with_matplotlib(names, counts, args.out)
    except Exception:
        draw_with_pil(names, counts, args.out)

    print('wrote image:', args.out)


if __name__ == '__main__':
    main()
