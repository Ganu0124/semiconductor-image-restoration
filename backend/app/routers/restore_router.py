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
    file: UploadFile = File(...),
    model: str = Form(default=cfg["models"]["default"]),
    ground_truth_split: Optional[str] = Form(default=None),
    ground_truth_filename: Optional[str] = Form(default=None),
    db: Session = Depends(get_db),
):
    if model not in cfg["models"]["available"]:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model}'. Available: {cfg['models']['available']}")

    contents = await file.read()
    input_path = save_upload(contents, file.filename)

    with Image.open(input_path) as im:
        w, h = im.size

    img_record = ImageRecord(filename=file.filename, path=str(input_path), width=w, height=h,
                              channels="L", split="upload")
    db.add(img_record)
    db.commit()
    db.refresh(img_record)

    gt_path = None
    gt_url = None
    if ground_truth_split and ground_truth_filename:
        candidate = resolve_path(cfg["paths"]["dataset_root"]) / ground_truth_split / "clean" / ground_truth_filename
        if candidate.exists():
            gt_path = candidate
            gt_url = f"/api/dataset/image/{ground_truth_split}/clean/{ground_truth_filename}"

    result = run_restoration(input_path, model, gt_path)

    result_record = ResultRecord(
        image_id=img_record.id,
        model_name=model,
        input_path=str(input_path),
        restored_path=str(result["restored_path"]),
        ground_truth_path=str(gt_path) if gt_path else None,
        psnr=result["psnr"], ssim=result["ssim"], lpips=result["lpips"],
        mae=result["mae"], mse=result["mse"],
        inference_time_seconds=result["inference_time_seconds"],
        is_trained_checkpoint=result["is_trained_checkpoint"],
        device=result["device"],
    )
    db.add(result_record)
    db.commit()
    db.refresh(result_record)

    note = None
    if not result["is_trained_checkpoint"]:
        note = (f"No trained checkpoint found for '{model}'. Output is from an UNTRAINED model "
                f"(random weights) — run training first: python ml/training/train.py --model {model}")
    elif gt_path is None:
        note = "No ground-truth image supplied, so PSNR/SSIM/LPIPS/MAE/MSE are not available for this result."

    return RestoreResponse(
        result_id=result_record.id,
        model=model,
        is_trained_checkpoint=result["is_trained_checkpoint"],
        device=result["device"],
        input_image_url=f"/api/storage/inputs/{Path(input_path).name}",
        restored_image_url=f"/api/storage/restored/{Path(result['restored_path']).name}",
        ground_truth_image_url=gt_url,
        psnr=result["psnr"], ssim=result["ssim"], lpips=result["lpips"],
        mae=result["mae"], mse=result["mse"],
        inference_time_seconds=result["inference_time_seconds"],
        note=note,
    )


@router.get("/results")
def list_results(db: Session = Depends(get_db), limit: int = 50):
    rows = db.query(ResultRecord).order_by(ResultRecord.created_at.desc()).limit(limit).all()
    if not rows:
        return {"count": 0, "results": [], "note": "No results available — run an experiment."}
    return {
        "count": len(rows),
        "results": [
            {
                "id": r.id, "model_name": r.model_name, "psnr": r.psnr, "ssim": r.ssim,
                "lpips": r.lpips, "mae": r.mae, "mse": r.mse,
                "inference_time_seconds": r.inference_time_seconds,
                "is_trained_checkpoint": r.is_trained_checkpoint,
                "created_at": r.created_at.isoformat(),
                "restored_image_url": f"/api/storage/restored/{Path(r.restored_path).name}" if r.restored_path else None,
            }
            for r in rows
        ],
    }
