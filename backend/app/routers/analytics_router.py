from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.app.db.models import ExperimentRecord, ResultRecord
from backend.app.db.session import get_db
from backend.app.schemas.schemas import DashboardSummary

router = APIRouter()


@router.get("/analytics/summary", response_model=DashboardSummary)
def summary(db: Session = Depends(get_db)):
    rows = db.query(ResultRecord).all()
    if not rows:
        return DashboardSummary(
            total_images_processed=0, avg_psnr=None, avg_ssim=None, avg_lpips=None,
            avg_inference_time_seconds=None, best_performing_model=None, has_results=False,
        )

    def avg(vals):
        vals = [v for v in vals if v is not None]
        return float(sum(vals) / len(vals)) if vals else None

    avg_psnr = avg([r.psnr for r in rows])
    avg_ssim = avg([r.ssim for r in rows])
    avg_lpips = avg([r.lpips for r in rows])
    avg_time = avg([r.inference_time_seconds for r in rows])

    # Best model = highest average PSNR among models that have >=1 scored result
    by_model: dict[str, list[float]] = {}
    for r in rows:
        if r.psnr is not None:
            by_model.setdefault(r.model_name, []).append(r.psnr)
    best_model = None
    if by_model:
        best_model = max(by_model.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))[0]

    return DashboardSummary(
        total_images_processed=len(rows),
        avg_psnr=avg_psnr, avg_ssim=avg_ssim, avg_lpips=avg_lpips,
        avg_inference_time_seconds=avg_time,
        best_performing_model=best_model,
        has_results=True,
    )


@router.get("/analytics/trends")
def trends(db: Session = Depends(get_db), model_name: str | None = None):
    q = db.query(ResultRecord).order_by(ResultRecord.created_at.asc())
    if model_name:
        q = q.filter(ResultRecord.model_name == model_name)
    rows = q.all()
    return {
        "count": len(rows),
        "points": [
            {
                "created_at": r.created_at.isoformat(),
                "model_name": r.model_name,
                "psnr": r.psnr, "ssim": r.ssim, "lpips": r.lpips,
                "inference_time_seconds": r.inference_time_seconds,
            }
            for r in rows
        ],
    }


@router.get("/analytics/training-history/{model_name}")
def training_history(model_name: str):
    """Reads the most recent training experiment JSON log for this model
    (written by ml/training/train.py) to show real per-epoch train/val loss."""
    from ml.utils.config import load_config, resolve_path
    cfg = load_config()
    exp_dir = resolve_path(cfg["paths"]["experiments_dir"])
    logs = sorted(exp_dir.glob(f"{model_name}_*.json"), reverse=True)
    if not logs:
        return {"model_name": model_name, "history": [], "note": "No training runs found for this model yet."}
    import json
    with open(logs[0]) as f:
        data = json.load(f)
    return {"model_name": model_name, "experiment_id": data["experiment_id"], "history": data["history"]}


@router.get("/models/compare")
def compare_models(db: Session = Depends(get_db)):
    """Real model-comparison table: aggregates actual results per model.
    Returns empty per-model rows (not zeros) when a model has no results yet."""
    from ml.models.registry import build_model, checkpoint_path
    from ml.utils.config import load_config
    from ml.models.unet import count_parameters

    cfg = load_config()
    out = []
    for name in cfg["models"]["available"]:
        rows = db.query(ResultRecord).filter(ResultRecord.model_name == name).all()
        ckpt = checkpoint_path(name, cfg)
        model = build_model(name, cfg)

        def avg(attr):
            vals = [getattr(r, attr) for r in rows if getattr(r, attr) is not None]
            return float(sum(vals) / len(vals)) if vals else None

        out.append({
            "model": name,
            "is_trained": ckpt.exists(),
            "num_results": len(rows),
            "avg_psnr": avg("psnr"),
            "avg_ssim": avg("ssim"),
            "avg_lpips": avg("lpips"),
            "avg_inference_time_seconds": avg("inference_time_seconds"),
            "parameters": count_parameters(model),
            "model_size_mb": (ckpt.stat().st_size / (1024 * 1024)) if ckpt.exists() else None,
        })
    return {"models": out}
