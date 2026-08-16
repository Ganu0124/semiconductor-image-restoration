"""Configurable image degradation: noise (Gaussian / Poisson / sensor-like)
and reduced spatial resolution (downsample + upsample back, simulating a
low-resolution acquisition). All parameters are configurable — no fixed
image dimension is assumed anywhere in this module.
"""
from __future__ import annotations

import random
from typing import Sequence

import numpy as np
from PIL import Image


def add_gaussian_noise(img: np.ndarray, sigma: float, rng: random.Random) -> np.ndarray:
    noise = np.random.RandomState(rng.randint(0, 2**31 - 1)).normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255)


def add_poisson_noise(img: np.ndarray, scale: float, rng: random.Random) -> np.ndarray:
    """Simulates shot noise: signal-dependent noise typical of photon/electron counting
    sensors (e.g. SEM detectors)."""
    rs = np.random.RandomState(rng.randint(0, 2**31 - 1))
    vals = np.clip(img.astype(np.float32), 1, 255) / 255.0 * scale
    noisy = rs.poisson(vals).astype(np.float32) / scale * 255.0
    return np.clip(noisy, 0, 255)


def add_sensor_noise(img: np.ndarray, read_noise_std: float, rng: random.Random) -> np.ndarray:
    """Approximates sensor read noise + fixed pattern noise: Gaussian read noise plus
    a small per-row bias term, mimicking line-scan detector artifacts."""
    rs = np.random.RandomState(rng.randint(0, 2**31 - 1))
    read = rs.normal(0, read_noise_std, img.shape)
    row_bias = rs.normal(0, read_noise_std / 2, (img.shape[0], 1))
    return np.clip(img.astype(np.float32) + read + row_bias, 0, 255)


def downsample_upsample(img: np.ndarray, scale: int) -> np.ndarray:
    """Simulates reduced spatial resolution acquisition: downsample by `scale`
    then upsample back to original size with bilinear interpolation, which is
    the standard way to create a degraded/clean SR-restoration pair at fixed size."""
    if scale <= 1:
        return img
    h, w = img.shape[:2]
    pil = Image.fromarray(img.astype(np.uint8))
    small = pil.resize((max(1, w // scale), max(1, h // scale)), Image.BICUBIC)
    back = small.resize((w, h), Image.BILINEAR)
    return np.array(back, dtype=np.float32)


def apply_degradation(
    clean: np.ndarray,
    noise_types: Sequence[str],
    gaussian_sigma_range: Sequence[float],
    poisson_scale: float,
    sensor_read_noise_std: float,
    downsample_scale: int,
    rng: random.Random,
) -> np.ndarray:
    """Applies a randomly-chosen configured degradation pipeline to a clean image
    and returns a uint8 degraded image of the SAME shape as the input (paired
    restoration target)."""
    out = clean.astype(np.float32).copy()

    out = downsample_upsample(out, downsample_scale)

    noise_type = rng.choice(list(noise_types)) if noise_types else "gaussian"
    if noise_type == "gaussian":
        sigma = rng.uniform(*gaussian_sigma_range)
        out = add_gaussian_noise(out, sigma, rng)
    elif noise_type == "poisson":
        out = add_poisson_noise(out, poisson_scale, rng)
    elif noise_type == "sensor":
        out = add_sensor_noise(out, sensor_read_noise_std, rng)

    return np.clip(out, 0, 255).astype(np.uint8)
