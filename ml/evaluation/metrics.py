"""Real, computed evaluation metrics. No fabricated values anywhere —
if an input pair can't be scored (e.g. LPIPS weights unavailable offline),
the caller receives None for that metric and the API/dashboard must display
"No results available" rather than inventing a number.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

_lpips_model = None
_lpips_import_error: Optional[str] = None


def _get_lpips_model(net: str = "alex"):
    global _lpips_model, _lpips_import_error
    if _lpips_model is not None:
        return _lpips_model
    if _lpips_import_error is not None:
        return None
    try:
        import lpips as lpips_lib
        _lpips_model = lpips_lib.LPIPS(net=net)
        _lpips_model.eval()
        return _lpips_model
    except Exception as e:  # noqa: BLE001 — offline weight download, missing net, etc.
        _lpips_import_error = str(e)
        return None


def psnr(clean: np.ndarray, restored: np.ndarray) -> float:
    """clean, restored: float arrays in [0, 1], same shape.
    For pixel-identical inputs the true PSNR is infinite; JSON has no
    representation for inf, so it is capped at 100 dB (effectively perfect,
    far above any real restoration result) rather than serialized as null."""
    value = float(peak_signal_noise_ratio(clean, restored, data_range=1.0))
    if not np.isfinite(value):
        return 100.0
    return value


def ssim(clean: np.ndarray, restored: np.ndarray) -> float:
    return float(structural_similarity(clean, restored, data_range=1.0))


def mae(clean: np.ndarray, restored: np.ndarray) -> float:
    return float(np.mean(np.abs(clean - restored)))


def mse(clean: np.ndarray, restored: np.ndarray) -> float:
    return float(np.mean((clean - restored) ** 2))


def lpips_score(clean: np.ndarray, restored: np.ndarray, net: str = "alex") -> Optional[float]:
    """Returns None (not a fake 0) if the LPIPS backbone can't be loaded
    (e.g. no internet access to fetch pretrained weights in this sandbox)."""
    model = _get_lpips_model(net)
    if model is None:
        return None
    import torch

    def to_tensor(arr):
        t = torch.from_numpy(arr).float()
        if t.ndim == 2:
            t = t.unsqueeze(0).repeat(3, 1, 1)  # grayscale -> 3ch
        t = t.unsqueeze(0)  # batch
        return t * 2 - 1  # LPIPS expects [-1, 1]

    with torch.no_grad():
        d = model(to_tensor(clean), to_tensor(restored))
    return float(d.item())


def compute_all_metrics(clean: np.ndarray, restored: np.ndarray, lpips_net: str = "alex") -> dict:
    """clean, restored: float32 arrays in [0, 1]. Returns a dict with real
    computed values; lpips may be None if unavailable (see lpips_score)."""
    return {
        "psnr": psnr(clean, restored),
        "ssim": ssim(clean, restored),
        "lpips": lpips_score(clean, restored, net=lpips_net),
        "mae": mae(clean, restored),
        "mse": mse(clean, restored),
    }


class Timer:
    """Context manager for measuring real inference time (seconds)."""
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._start
