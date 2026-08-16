from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from fastapi import APIRouter, File, HTTPException, UploadFile
import numpy as np
from PIL import Image
import io

from ml.evaluation.metrics import compute_all_metrics
from ml.utils.config import load_config
from backend.app.schemas.schemas import MetricsResult

router = APIRouter()
cfg = load_config()


@router.post("/evaluate", response_model=MetricsResult)
async def evaluate(restored: UploadFile = File(...), ground_truth: UploadFile = File(...)):
    """Computes REAL PSNR/SSIM/LPIPS/MAE/MSE between two uploaded images
    (e.g. a restored output and its ground truth). Used by the Evaluation page
    for ad-hoc comparisons outside the main restore flow."""
    restored_bytes = await restored.read()
    gt_bytes = await ground_truth.read()

    restored_arr = np.array(Image.open(io.BytesIO(restored_bytes)).convert("L")).astype(np.float32) / 255.0
    gt_arr = np.array(Image.open(io.BytesIO(gt_bytes)).convert("L")).astype(np.float32) / 255.0

    if restored_arr.shape != gt_arr.shape:
        raise HTTPException(
            status_code=400,
            detail=f"Shape mismatch: restored={restored_arr.shape} ground_truth={gt_arr.shape}. "
                   f"Images must be the same size to compute pixel-wise metrics.",
        )

    metrics = compute_all_metrics(gt_arr, restored_arr, lpips_net=cfg["evaluation"]["lpips_net"])
    return MetricsResult(**metrics)
