import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def _png_bytes(size=32, val=128):
    arr = np.full((size, size), val, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "device" in body


def test_models_list():
    r = client.get("/api/models")
    assert r.status_code == 200
    names = [m["name"] for m in r.json()]
    assert "unet" in names
    assert "swinir" in names


def test_dataset_stats():
    r = client.get("/api/dataset/stats")
    assert r.status_code == 200
    body = r.json()
    assert "splits" in body


def test_restore_upload():
    r = client.post(
        "/api/restore",
        files={"file": ("test.png", _png_bytes(), "image/png")},
        data={"model": "unet"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "unet"
    assert "restored_image_url" in body
    assert body["inference_time_seconds"] >= 0


def test_restore_unknown_model_rejected():
    r = client.post(
        "/api/restore",
        files={"file": ("test.png", _png_bytes(), "image/png")},
        data={"model": "not-a-real-model"},
    )
    assert r.status_code == 400


def test_evaluate_shape_mismatch_rejected():
    r = client.post(
        "/api/evaluate",
        files={
            "restored": ("a.png", _png_bytes(size=32), "image/png"),
            "ground_truth": ("b.png", _png_bytes(size=48), "image/png"),
        },
    )
    assert r.status_code == 400


def test_evaluate_identical_images():
    buf1 = _png_bytes(size=32, val=100)
    buf2 = _png_bytes(size=32, val=100)
    r = client.post(
        "/api/evaluate",
        files={
            "restored": ("a.png", buf1, "image/png"),
            "ground_truth": ("b.png", buf2, "image/png"),
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["psnr"] > 60
    assert body["ssim"] == pytest.approx(1.0, abs=1e-4)


def test_results_endpoint_after_restore():
    r = client.get("/api/results")
    assert r.status_code == 200
    assert "results" in r.json()


def test_analytics_summary():
    r = client.get("/api/analytics/summary")
    assert r.status_code == 200
    assert "has_results" in r.json()


def test_models_compare():
    r = client.get("/api/models/compare")
    assert r.status_code == 200
    assert len(r.json()["models"]) >= 2
