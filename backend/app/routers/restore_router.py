from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from PIL import Image
from sqlalchemy.orm import Session

from ml.utils.config import load_config, resolve_path
from backend.app.db.models import ImageRecord, ResultRecord
from backend.app.db.session import get_db
from backend.app.schemas.schemas import RestoreResponse
from backend.app.services.restoration_service import run_restoration, save_upload

router = APIRouter()
cfg = load_config()


@router.post("/restore", response_model=RestoreResponse)
async def restore(
    file: Optional[UploadFile] = File(default=None),
    model: str = Form(default=cfg["models"]["default"]),
    ground_truth_split: Optional[str] = Form(default=None),
    ground_truth_filename: Optional[str] = Form(default=None),
    input_split: Optional[str] = Form(default=None),
    input_filename: Optional[str] = Form(default=None),
    db: Session = Depends(get_db),
):
    # ---------------------------------------------------------
    # Validate model
    # ---------------------------------------------------------
    if model not in cfg["models"]["available"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model}'. Available: {cfg['models']['available']}"
        )

    dataset_root = resolve_path(cfg["paths"]["dataset_root"])

    # ---------------------------------------------------------
    # 1. Determine input image
    # ---------------------------------------------------------
    input_path = None
    input_url = None
    filename = None

    # CASE A: Image uploaded from user's computer
    if file is not None:
        contents = await file.read()

        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        filename = file.filename or "uploaded_image.png"
        input_path = save_upload(contents, filename)

        input_url = f"/api/storage/inputs/{Path(input_path).name}"

    # CASE B: Image selected from test/train dataset
    elif input_split and input_filename:
        # Only allow known dataset splits
        allowed_splits = {"train", "val", "validation", "test"}

        if input_split not in allowed_splits:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid input split '{input_split}'"
            )

        filename = Path(input_filename).name

        input_path = (
            dataset_root
            / input_split
            / "degraded"
            / filename
        )

        if not input_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Dataset input image not found: {input_split}/degraded/{filename}"
            )

        input_url = (
            f"/api/dataset/image/"
            f"{input_split}/degraded/{filename}"
        )

    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either an uploaded file or input_split/input_filename"
        )

    # ---------------------------------------------------------
    # Read image dimensions
    # ---------------------------------------------------------
    try:
        with Image.open(input_path) as im:
            w, h = im.size
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read input image: {e}"
        )

    # ---------------------------------------------------------
    # Save image record
    # ---------------------------------------------------------
    img_record = ImageRecord(
        filename=filename,
        path=str(input_path),
        width=w,
        height=h,
        channels="L",
        split=input_split if input_split else "upload",
    )

    db.add(img_record)
    db.commit()
    db.refresh(img_record)

    # ---------------------------------------------------------
    # Ground truth
    # ---------------------------------------------------------
    gt_path = None
    gt_url = None

    if ground_truth_split and ground_truth_filename:
        gt_filename = Path(ground_truth_filename).name

        candidate = (
            dataset_root
            / ground_truth_split
            / "clean"
            / gt_filename
        )

        if candidate.exists():
            gt_path = candidate

            gt_url = (
                f"/api/dataset/image/"
                f"{ground_truth_split}/clean/{gt_filename}"
            )

    # ---------------------------------------------------------
    # Run AI restoration
    # ---------------------------------------------------------
    result = run_restoration(
        input_path,
        model,
        gt_path
    )

    # ---------------------------------------------------------
    # Save result
    # ---------------------------------------------------------
    result_record = ResultRecord(
        image_id=img_record.id,
        model_name=model,
        input_path=str(input_path),
        restored_path=str(result["restored_path"]),
        ground_truth_path=str(gt_path) if gt_path else None,
        psnr=result["psnr"],
        ssim=result["ssim"],
        lpips=result["lpips"],
        mae=result["mae"],
        mse=result["mse"],
        inference_time_seconds=result["inference_time_seconds"],
        is_trained_checkpoint=result["is_trained_checkpoint"],
        device=result["device"],
    )

    db.add(result_record)
    db.commit()
    db.refresh(result_record)

    # ---------------------------------------------------------
    # Result note
    # ---------------------------------------------------------
    note = None

    if not result["is_trained_checkpoint"]:
        note = (
            f"No trained checkpoint found for '{model}'. "
            f"Output is from an UNTRAINED model (random weights) — "
            f"run training first: "
            f"python ml/training/train.py --model {model}"
        )

    elif gt_path is None:
        note = (
            "No ground-truth image supplied, so "
            "PSNR/SSIM/LPIPS/MAE/MSE are not available for this result."
        )

    # ---------------------------------------------------------
    # Return response
    # ---------------------------------------------------------
    return RestoreResponse(
        result_id=result_record.id,
        model=model,
        is_trained_checkpoint=result["is_trained_checkpoint"],
        device=result["device"],
        input_image_url=input_url,
        restored_image_url=(
            f"/api/storage/restored/"
            f"{Path(result['restored_path']).name}"
        ),
        ground_truth_image_url=gt_url,
        psnr=result["psnr"],
        ssim=result["ssim"],
        lpips=result["lpips"],
        mae=result["mae"],
        mse=result["mse"],
        inference_time_seconds=result["inference_time_seconds"],
        note=note,
    )


@router.get("/results")
def list_results(
    db: Session = Depends(get_db),
    limit: int = 50
):
    rows = (
        db.query(ResultRecord)
        .order_by(ResultRecord.created_at.desc())
        .limit(limit)
        .all()
    )

    if not rows:
        return {
            "count": 0,
            "results": [],
            "note": "No results available — run an experiment."
        }

    return {
        "count": len(rows),
        "results": [
            {
                "id": r.id,
                "model_name": r.model_name,
                "psnr": r.psnr,
                "ssim": r.ssim,
                "lpips": r.lpips,
                "mae": r.mae,
                "mse": r.mse,
                "inference_time_seconds": r.inference_time_seconds,
                "is_trained_checkpoint": r.is_trained_checkpoint,
                "created_at": r.created_at.isoformat(),
                "restored_image_url": (
                    f"/api/storage/restored/"
                    f"{Path(r.restored_path).name}"
                    if r.restored_path
                    else None
                ),
            }
            for r in rows
        ],
    }