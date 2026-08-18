from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ml.evaluation.metrics import compute_all_metrics  # noqa: E402
from ml.inference.inference import restore_image  # noqa: E402
from ml.utils.config import get_device, load_config, resolve_path  # noqa: E402

cfg = load_config()
STORAGE_DIR = resolve_path("backend/storage")
INPUT_DIR = STORAGE_DIR / "inputs"
RESTORED_DIR = STORAGE_DIR / "restored"
for d in (INPUT_DIR, RESTORED_DIR):
    d.mkdir(parents=True, exist_ok=True)


def save_upload(file_bytes: bytes, original_filename: str) -> Path:
    ext = Path(original_filename).suffix or ".png"
    uid = uuid.uuid4().hex[:12]
    path = INPUT_DIR / f"{uid}{ext}"
    with open(path, "wb") as f:
        f.write(file_bytes)
    return path


def _load_as_uint8(path: Path) -> np.ndarray:
    """Load grayscale image as uint8 (H, W). Handles both PIL images and .npy arrays."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path).astype(np.float32)
        arr = np.clip(arr, 0.0, 1.0)
        return (arr * 255.0).astype(np.uint8)
    return np.array(Image.open(path).convert("L"))


def _load_as_float32(path: Path) -> np.ndarray:
    """Load grayscale image as float32 [0, 1]. Handles both PIL images and .npy arrays."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path).astype(np.float32)
        return np.clip(arr, 0.0, 1.0)
    return np.array(Image.open(path).convert("L")).astype(np.float32) / 255.0


def run_restoration(image_path: Path, model_name: str, ground_truth_path: Optional[Path] = None) -> dict:
    """Runs the REAL inference pipeline (ml/inference/inference.py) and, if a
    ground-truth image is supplied, computes REAL metrics (ml/evaluation/metrics.py).
    Nothing here fabricates numbers: metrics are None unless a ground truth exists.
    Supports both PIL-readable images (.png/.jpg/.tif) and .npy arrays.
    """
    img = _load_as_uint8(image_path)
    result = restore_image(img, model_name, cfg)

    uid = uuid.uuid4().hex[:12]
    restored_path = RESTORED_DIR / f"{uid}.png"
    Image.fromarray(result["restored"]).save(restored_path)

    metrics = {"psnr": None, "ssim": None, "lpips": None, "mae": None, "mse": None}
    if ground_truth_path is not None and Path(ground_truth_path).exists():
        gt = _load_as_float32(ground_truth_path)
        # resize restored/gt if mismatched (shouldn't happen for paired dataset)
        if gt.shape == result["restored_float"].shape:
            metrics = compute_all_metrics(gt, result["restored_float"], lpips_net=cfg["evaluation"]["lpips_net"])

    return {
        "restored_path": restored_path,
        "model": model_name,
        "is_trained_checkpoint": result["is_trained_checkpoint"],
        "device": result["device"],
        "inference_time_seconds": result["inference_time_seconds"],
        **metrics,
    }
