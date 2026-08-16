import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.evaluation.metrics import psnr, ssim, mae, mse, compute_all_metrics
from ml.models.unet import UNet, count_parameters
from ml.models.swinir_lite import SwinIRLite
from ml.preprocessing.degradation import apply_degradation
from ml.datasets.dataset_analyzer import analyze_dataset
from ml.utils.config import load_config, resolve_path
import random


def test_psnr_identical_is_high():
    img = np.random.rand(32, 32).astype(np.float32)
    assert psnr(img, img) > 60  # identical images -> very high PSNR


def test_ssim_identical_is_one():
    img = np.random.rand(32, 32).astype(np.float32)
    assert ssim(img, img) == pytest.approx(1.0, abs=1e-5)


def test_mae_mse_zero_for_identical():
    img = np.random.rand(16, 16).astype(np.float32)
    assert mae(img, img) == 0
    assert mse(img, img) == 0


def test_metrics_degrade_with_noise():
    img = np.random.rand(32, 32).astype(np.float32)
    noisy = np.clip(img + np.random.normal(0, 0.2, img.shape), 0, 1).astype(np.float32)
    assert psnr(img, noisy) < psnr(img, img)
    assert ssim(img, noisy) < 1.0


def test_compute_all_metrics_keys():
    img = np.random.rand(16, 16).astype(np.float32)
    result = compute_all_metrics(img, img)
    assert set(result.keys()) == {"psnr", "ssim", "lpips", "mae", "mse"}


def test_unet_forward_pass_shape():
    model = UNet(base_channels=8, depth=2)
    x = torch.rand(2, 1, 32, 32)
    out = model(x)
    assert out.shape == x.shape
    assert out.min() >= 0 and out.max() <= 1


def test_swinir_forward_pass_shape():
    model = SwinIRLite(embed_dim=16, window_size=4, depths=[1, 1])
    x = torch.rand(1, 1, 24, 24)
    out = model(x)
    assert out.shape == x.shape


def test_unet_param_count_positive():
    model = UNet()
    assert count_parameters(model) > 0


def test_degradation_changes_image():
    rng = random.Random(0)
    clean = (np.random.rand(32, 32) * 255).astype(np.uint8)
    degraded = apply_degradation(
        clean, noise_types=["gaussian"], gaussian_sigma_range=[10, 10],
        poisson_scale=30.0, sensor_read_noise_std=3.0, downsample_scale=1, rng=rng,
    )
    assert degraded.shape == clean.shape
    assert not np.array_equal(clean, degraded)


def test_dataset_analyzer_missing_dir_reports_not_exists(tmp_path):
    report = analyze_dataset(tmp_path / "nonexistent", ["train", "val", "test"], [".png"])
    assert report["root_exists"] is False
    assert report["total_images"] == 0


def test_dataset_analyzer_on_synthetic_dataset():
    cfg = load_config()
    dataset_root = resolve_path(cfg["paths"]["dataset_root"])
    if not dataset_root.exists():
        pytest.skip("Synthetic dataset not generated yet; run ml/datasets/generate_synthetic_dataset.py")
    report = analyze_dataset(dataset_root, cfg["dataset"]["splits"], cfg["dataset"]["image_extensions"])
    assert report["total_images"] > 0
    assert report["is_paired_dataset"] is True
