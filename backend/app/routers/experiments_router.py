from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from ml.training.train import train as run_training
from ml.utils.config import load_config, resolve_path
from backend.app.db.models import ExperimentRecord
from backend.app.db.session import get_db
from backend.app.schemas.schemas import ExperimentCreate, ExperimentOut

router = APIRouter()
cfg = load_config()


def _train_and_record(model_name: str, dev_mode: bool, epochs: int | None, experiment_uid: str):
    """Runs real training synchronously in a background task, then updates
    the DB row with the actual results from the training log."""
    from backend.app.db.session import SessionLocal

    log_path = run_training(model_name, dev_mode, epochs, resume=None)
    with open(log_path) as f:
        log = json.load(f)

    db = SessionLocal()
    try:
        row = db.query(ExperimentRecord).filter_by(experiment_uid=experiment_uid).first()
        if row:
            row.best_val_psnr = log["best_val_psnr"]
            row.elapsed_seconds = log["elapsed_seconds"]
            row.epochs = log["epochs_run"]
            row.gpu = log["device"]
            db.commit()
    finally:
        db.close()


@router.post("/experiments", response_model=ExperimentOut)
def create_experiment(payload: ExperimentCreate, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    if payload.model_name not in cfg["models"]["available"]:
        raise HTTPException(status_code=400, detail=f"Unknown model. Available: {cfg['models']['available']}")

    experiment_uid = f"{payload.model_name}_{uuid.uuid4().hex[:8]}"
    patch_size = cfg["training"]["dev_patch_size"] if payload.dev_mode else cfg["training"]["patch_size"]
    batch_size = cfg["training"]["dev_batch_size"] if payload.dev_mode else cfg["training"]["batch_size"]

    row = ExperimentRecord(
        experiment_uid=experiment_uid,
        model_name=payload.model_name,
        dataset_version="synthetic-placeholder-v1" if cfg["dataset"]["is_synthetic_placeholder"] else "user-dataset-v1",
        patch_size=patch_size,
        noise_level="configured (see configs/config.yaml -> degradation)",
        scale_factor=max(cfg["degradation"]["downsample_scales"]),
        learning_rate=cfg["training"]["learning_rate"],
        batch_size=batch_size,
        epochs=payload.epochs,
        gpu="pending",
        dev_mode=payload.dev_mode,
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    # Training runs as a background task; DEV_MODE keeps this to a few seconds.
    background_tasks.add_task(_train_and_record, payload.model_name, payload.dev_mode, payload.epochs, experiment_uid)

    return row


@router.get("/experiments", response_model=list[ExperimentOut])
def list_experiments(db: Session = Depends(get_db), limit: int = 50):
    return db.query(ExperimentRecord).order_by(ExperimentRecord.created_at.desc()).limit(limit).all()
