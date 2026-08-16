"""PyTorch Dataset for paired (degraded, clean) semiconductor inspection images.
Reads real files from disk under <dataset_root>/<split>/{clean,degraded}/.
Patch size, and whether patches are used at all, are configurable — no fixed
image dimension is assumed.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class PairedRestorationDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        split: str,
        extensions: list[str],
        patch_size: Optional[int] = None,
        augment: bool = True,
    ):
        self.split_dir = Path(dataset_root) / split
        self.clean_dir = self.split_dir / "clean"
        self.degraded_dir = self.split_dir / "degraded"
        self.patch_size = patch_size
        self.augment = augment and split == "train"

        if not (self.clean_dir.exists() and self.degraded_dir.exists()):
            raise FileNotFoundError(
                f"Expected paired layout at {self.clean_dir} and {self.degraded_dir}. "
                f"Run ml/datasets/dataset_analyzer.py to inspect your dataset layout."
            )

        clean_names = {p.name for p in self.clean_dir.iterdir() if p.suffix.lower() in extensions}
        degraded_names = {p.name for p in self.degraded_dir.iterdir() if p.suffix.lower() in extensions}
        self.filenames = sorted(clean_names & degraded_names)

        if not self.filenames:
            raise RuntimeError(f"No matching clean/degraded pairs found in {self.split_dir}")

    def __len__(self) -> int:
        return len(self.filenames)

    def _load(self, path: Path) -> np.ndarray:
        with Image.open(path) as im:
            arr = np.array(im.convert("L"), dtype=np.float32) / 255.0
        return arr

    def __getitem__(self, idx: int):
        name = self.filenames[idx]
        clean = self._load(self.clean_dir / name)
        degraded = self._load(self.degraded_dir / name)

        h, w = clean.shape
        if self.patch_size and self.patch_size < min(h, w):
            ps = self.patch_size
            top = random.randint(0, h - ps)
            left = random.randint(0, w - ps)
            clean = clean[top: top + ps, left: left + ps]
            degraded = degraded[top: top + ps, left: left + ps]

        if self.augment:
            if random.random() < 0.5:
                clean, degraded = np.fliplr(clean).copy(), np.fliplr(degraded).copy()
            if random.random() < 0.5:
                clean, degraded = np.flipud(clean).copy(), np.flipud(degraded).copy()
            k = random.randint(0, 3)
            clean, degraded = np.rot90(clean, k).copy(), np.rot90(degraded, k).copy()

        clean_t = torch.from_numpy(clean).unsqueeze(0).float()
        degraded_t = torch.from_numpy(degraded).unsqueeze(0).float()
        return {"degraded": degraded_t, "clean": clean_t, "filename": name}
