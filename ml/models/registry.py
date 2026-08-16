"""Central model registry so the API / training / inference scripts can select
a model by name (as required by the dashboard's model dropdown) without
duplicating construction logic."""
from __future__ import annotations

import torch.nn as nn

from ml.models.unet import UNet
from ml.models.swinir_lite import SwinIRLite


def build_model(name: str, cfg: dict) -> nn.Module:
    name = name.lower()
    if name == "unet":
        mcfg = cfg["models"]["unet"]
        return UNet(base_channels=mcfg["base_channels"], depth=mcfg["depth"])
    if name == "swinir":
        mcfg = cfg["models"]["swinir"]
        return SwinIRLite(
            embed_dim=mcfg["embed_dim"],
            window_size=mcfg["window_size"],
            depths=mcfg["depths"],
        )
    raise ValueError(f"Unknown model '{name}'. Available: {cfg['models']['available']}")


def checkpoint_path(name: str, cfg: dict, filename: str = "best_model.pth"):
    from ml.utils.config import resolve_path
    models_dir = resolve_path(cfg["paths"]["models_dir"])
    return models_dir / name / filename
