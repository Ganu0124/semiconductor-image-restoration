"""Inference pipeline: loads a trained checkpoint for the requested model and
restores a given image. Used directly by the CLI and imported by the FastAPI
backend (backend/app/services/restoration_service.py) so the API performs the
SAME real inference code path, not a mock.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.models.registry import build_model, checkpoint_path  # noqa: E402
from ml.utils.config import get_device, load_config  # noqa: E402

_model_cache: dict[str, torch.nn.Module] = {}


def load_model(model_name: str, cfg: dict, device: str) -> tuple[torch.nn.Module, bool]:
    """Returns (model, is_trained_checkpoint). If no checkpoint exists yet,
    returns a freshly-initialized (untrained) model and False, so the API can
    surface that honestly instead of pretending the output is from a trained
    model."""
    cache_key = f"{model_name}:{device}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    model = build_model(model_name, cfg).to(device)
    ckpt_path = checkpoint_path(model_name, cfg)
    is_trained = False
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        is_trained = True
    model.eval()
    _model_cache[cache_key] = (model, is_trained)
    return model, is_trained


def restore_image(
    image: np.ndarray,
    model_name: str,
    cfg: Optional[dict] = None,
    device: Optional[str] = None,
) -> dict:
    """image: uint8 or float grayscale array (H, W).
    Returns dict with restored uint8 array, inference_time_seconds, is_trained_checkpoint.
    """
    cfg = cfg or load_config()
    device = device or get_device(cfg)
    model, is_trained = load_model(model_name, cfg, device)

    arr = image.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0

    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        t0 = time.perf_counter()
        out = model(tensor)
        elapsed = time.perf_counter() - t0

    restored = out.squeeze(0).squeeze(0).cpu().numpy()
    restored_uint8 = np.clip(restored * 255.0, 0, 255).astype(np.uint8)

    return {
        "restored": restored_uint8,
        "restored_float": restored,
        "inference_time_seconds": elapsed,
        "is_trained_checkpoint": is_trained,
        "device": device,
        "model": model_name,
    }


def main():
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to degraded image")
    parser.add_argument("--output", required=True, help="Path to write restored image")
    parser.add_argument("--model", default=cfg["models"]["default"], choices=cfg["models"]["available"])
    args = parser.parse_args()

    img = np.array(Image.open(args.input).convert("L"))
    result = restore_image(img, args.model, cfg)
    Image.fromarray(result["restored"]).save(args.output)
    print(f"[inference] model={args.model} trained_checkpoint={result['is_trained_checkpoint']} "
          f"time={result['inference_time_seconds']*1000:.1f}ms -> {args.output}")


if __name__ == "__main__":
    main()
