"""NpyPairedRestorationDataset — dataset loader for .npy semiconductor images.

Handles the dataset produced by ``scripts/prepare_real_dataset.py``:

    <dataset_root>/<split>/clean/     <- GT arrays, shape (H, W) float32, range [0,1]
    <dataset_root>/<split>/degraded/  <- NoisyLR arrays, shape (H/s, W/s) float32

Key behaviours
--------------
* Files are loaded with ``numpy.load()`` — no PIL required.
* GT is 256×256, NoisyLR is 128×128 (scale_factor=2). The degraded array is
  upsampled to GT resolution via bilinear interpolation *inside* ``__getitem__``
  so that both tensors entering the model are always the same spatial size.
* Values are clipped to [0, 1] at load time (NoisyLR has small out-of-range
  values due to noise).
* Random paired patch extraction and horizontal/vertical flip augmentation
  are applied consistently to both the clean and (upsampled) degraded arrays.
* The test split may contain only a ``degraded/`` subfolder (no GT). The
  dataset raises ``FileNotFoundError`` early in that case so the user gets a
  clear message instead of a cryptic KeyError at training time.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class NpyPairedRestorationDataset(Dataset):
    """Paired (degraded, clean) dataset backed by .npy arrays.

    Parameters
    ----------
    dataset_root:
        Path to the dataset root (``data/real``).
    split:
        One of ``"train"``, ``"val"``, ``"test"``.
    patch_size:
        If given, random square patches of this size are extracted from the
        *GT* (clean) array.  The degraded array is cropped at the corresponding
        location scaled by ``1 / scale_factor``.  Set to ``None`` to use the
        full image.
    scale_factor:
        Spatial downscale factor between GT and NoisyLR (default 2).
    augment:
        If ``True`` (and split == ``"train"``), applies random flips and 90°
        rotations.
    """

    def __init__(
        self,
        dataset_root: Path,
        split: str,
        patch_size: Optional[int] = None,
        scale_factor: int = 2,
        augment: bool = True,
    ):
        self.split_dir    = Path(dataset_root) / split
        self.clean_dir    = self.split_dir / "clean"
        self.degraded_dir = self.split_dir / "degraded"
        self.patch_size   = patch_size
        self.scale_factor = scale_factor
        self.augment      = augment and (split == "train")

        # Test split may not have clean dir — that is expected.
        if not self.degraded_dir.exists():
            raise FileNotFoundError(
                f"Degraded directory not found: {self.degraded_dir}\n"
                f"Run scripts/prepare_real_dataset.py first."
            )

        # Pair by matching filename stems
        degraded_files = {
            p.name: p for p in self.degraded_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".npy"
        }

        if self.clean_dir.exists():
            clean_files = {
                p.name: p for p in self.clean_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".npy"
            }
            common = sorted(clean_files.keys() & degraded_files.keys())
            if not common:
                raise RuntimeError(
                    f"No matching .npy pairs found in {self.split_dir}. "
                    f"GT count={len(clean_files)}, Noisy count={len(degraded_files)}"
                )
            self.filenames     = common
            self._clean_map    = clean_files
            self._degraded_map = degraded_files
            self._has_gt       = True
        else:
            # Inference-only split (test without GT)
            self.filenames     = sorted(degraded_files.keys())
            self._clean_map    = {}
            self._degraded_map = degraded_files
            self._has_gt       = False

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.filenames)

    # ------------------------------------------------------------------
    def _load(self, path: Path) -> np.ndarray:
        arr = np.load(path).astype(np.float32)
        # Ensure 2-D (H, W) — drop any singleton channel dim if present
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        # Clip to [0, 1]; NoisyLR can have small out-of-range values
        return np.clip(arr, 0.0, 1.0)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        name    = self.filenames[idx]
        deg_arr = self._load(self._degraded_map[name])   # (H/s, W/s)

        # Upsample degraded → GT resolution using bilinear interpolation
        # so both tensors have the same spatial size entering the model.
        if self.scale_factor != 1:
            lr_t = torch.from_numpy(deg_arr).unsqueeze(0).unsqueeze(0)  # (1,1,H/s,W/s)
            lr_up = torch.nn.functional.interpolate(
                lr_t,
                scale_factor=float(self.scale_factor),
                mode="bilinear",
                align_corners=False,
            )
            deg_up = lr_up.squeeze(0).squeeze(0).numpy()  # (H, W)
        else:
            deg_up = deg_arr

        if self._has_gt:
            clean_arr = self._load(self._clean_map[name])  # (H, W)

            # ---- Patch extraction ----
            h, w = clean_arr.shape
            if self.patch_size and self.patch_size < min(h, w):
                ps  = self.patch_size
                top  = random.randint(0, h - ps)
                left = random.randint(0, w - ps)
                clean_arr = clean_arr[top: top + ps, left: left + ps]
                deg_up    = deg_up[top: top + ps, left: left + ps]

            # ---- Augmentation ----
            if self.augment:
                if random.random() < 0.5:
                    clean_arr = np.fliplr(clean_arr).copy()
                    deg_up    = np.fliplr(deg_up).copy()
                if random.random() < 0.5:
                    clean_arr = np.flipud(clean_arr).copy()
                    deg_up    = np.flipud(deg_up).copy()
                k = random.randint(0, 3)
                clean_arr = np.rot90(clean_arr, k).copy()
                deg_up    = np.rot90(deg_up,    k).copy()

            clean_t = torch.from_numpy(clean_arr).unsqueeze(0).float()
        else:
            clean_t = torch.zeros(1, *deg_up.shape, dtype=torch.float32)

        degraded_t = torch.from_numpy(deg_up).unsqueeze(0).float()
        return {"degraded": degraded_t, "clean": clean_t, "filename": name, "has_gt": self._has_gt}
