from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from fastapi import APIRouter

from ml.models.registry import build_model, checkpoint_path
from ml.models.unet import count_parameters as count_params_unet
from ml.utils.config import load_config
from backend.app.schemas.schemas import ModelInfo

router = APIRouter()
cfg = load_config()


@router.get("/models", response_model=list[ModelInfo])
def list_models():
    infos = []
    for name in cfg["models"]["available"]:
        ckpt_path = checkpoint_path(name, cfg)
        is_trained = ckpt_path.exists()
        model = build_model(name, cfg)
        n_params = count_params_unet(model)  # generic parameter counter
        size_mb = None
        if is_trained:
            size_mb = ckpt_path.stat().st_size / (1024 * 1024)
        infos.append(ModelInfo(name=name, is_trained=is_trained, parameters=n_params, model_size_mb=size_mb))
    return infos
