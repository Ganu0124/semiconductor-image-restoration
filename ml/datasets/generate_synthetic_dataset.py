"""
generate_synthetic_dataset.py

*** THIS DATASET IS A SYNTHETIC PLACEHOLDER ***
No real semiconductor inspection dataset was supplied in this environment.
This script procedurally generates images that mimic coarse structural
characteristics of semiconductor/wafer imagery (grid lines, circular dies,
via/pad-like dots, sharp edges) purely so the rest of the pipeline
(dataset analyzer, loader, training, evaluation, dashboard) can be exercised
end-to-end with real code paths and real metric computation.

Replace `data/synthetic` with your real dataset (or point
paths.dataset_root in configs/config.yaml / DATASET_ROOT env var at it) and
re-run ml/datasets/dataset_analyzer.py — nothing downstream needs to change,
as long as the folder layout below is respected:

<dataset_root>/
    train/clean/*.png      train/degraded/*.png
    val/clean/*.png        val/degraded/*.png
    test/clean/*.png       test/degraded/*.png

Every "clean" image has a same-named "degraded" counterpart -> paired dataset.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.preprocessing.degradation import apply_degradation  # noqa: E402
from ml.utils.config import load_config, resolve_path  # noqa: E402


def _draw_die_pattern(size: int, rng: random.Random) -> Image.Image:
    """Procedurally draw a wafer/die-like grayscale pattern: grid lines,
    rectangular circuit blocks, circular vias, and a few 'defect' blemishes."""
    img = Image.new("L", (size, size), color=rng.randint(30, 60))
    draw = ImageDraw.Draw(img)

    # Die grid (dicing lines)
    step = rng.choice([16, 24, 32])
    for x in range(0, size, step):
        draw.line([(x, 0), (x, size)], fill=rng.randint(90, 130), width=1)
    for y in range(0, size, step):
        draw.line([(0, y), (size, y)], fill=rng.randint(90, 130), width=1)

    # Circuit-like rectangular blocks
    for _ in range(rng.randint(8, 20)):
        w, h = rng.randint(4, 14), rng.randint(4, 14)
        x0, y0 = rng.randint(0, size - w), rng.randint(0, size - h)
        shade = rng.randint(140, 220)
        draw.rectangle([x0, y0, x0 + w, y0 + h], fill=shade, outline=shade - 30)

    # Via / pad dots
    for _ in range(rng.randint(15, 40)):
        r = rng.randint(1, 3)
        cx, cy = rng.randint(r, size - r), rng.randint(r, size - r)
        shade = rng.randint(200, 255)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=shade)

    # Rare "defect" — a small scratch or particle, kept in both clean & degraded
    if rng.random() < 0.3:
        x0, y0 = rng.randint(0, size - 10), rng.randint(0, size - 10)
        x1, y1 = x0 + rng.randint(3, 10), y0 + rng.randint(3, 10)
        draw.line([(x0, y0), (x1, y1)], fill=rng.randint(0, 20), width=1)

    return img


def generate_split(split: str, n_images: int, size: int, out_root: Path, seed: int) -> None:
    rng = random.Random(seed)
    cfg = load_config()
    clean_dir = out_root / split / "clean"
    degraded_dir = out_root / split / "degraded"
    clean_dir.mkdir(parents=True, exist_ok=True)
    degraded_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_images):
        clean = _draw_die_pattern(size, rng)
        clean_arr = np.array(clean, dtype=np.uint8)

        degraded_arr = apply_degradation(
            clean_arr,
            noise_types=cfg["degradation"]["noise_types"],
            gaussian_sigma_range=cfg["degradation"]["gaussian_sigma_range"],
            poisson_scale=cfg["degradation"]["poisson_scale"],
            sensor_read_noise_std=cfg["degradation"]["sensor_read_noise_std"],
            downsample_scale=rng.choice(cfg["degradation"]["downsample_scales"] + [1, 1]),
            rng=rng,
        )

        fname = f"{split}_{i:04d}.png"
        Image.fromarray(clean_arr).save(clean_dir / fname)
        Image.fromarray(degraded_arr).save(degraded_dir / fname)

    print(f"[synthetic-dataset] {split}: wrote {n_images} pairs -> {clean_dir} / {degraded_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/synthetic", help="Output dataset root")
    parser.add_argument("--size", type=int, default=128, help="Image side length (square)")
    parser.add_argument("--train", type=int, default=60)
    parser.add_argument("--val", type=int, default=15)
    parser.add_argument("--test", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_root = resolve_path(args.out)
    generate_split("train", args.train, args.size, out_root, args.seed)
    generate_split("val", args.val, args.size, out_root, args.seed + 1)
    generate_split("test", args.test, args.size, out_root, args.seed + 2)
    print("[synthetic-dataset] DONE. This is placeholder data — see module docstring.")


if __name__ == "__main__":
    main()
