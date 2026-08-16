"""Central config loader. Everything in ml/ and backend/ reads settings from
configs/config.yaml through this module instead of hardcoding values.
Environment variables (see .env.example) can override select keys, notably
DEV_MODE, DATASET_ROOT, and DEVICE.
"""
from __future__ import annotations

import os
import functools
from pathlib import Path
from typing import Any

import yaml

# Project root = two levels up from this file (ml/utils/config.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


@functools.lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    # Environment overrides
    dev_mode_env = os.environ.get("DEV_MODE")
    if dev_mode_env is not None:
        cfg["training"]["dev_mode"] = dev_mode_env.strip().lower() in ("1", "true", "yes")

    dataset_root_env = os.environ.get("DATASET_ROOT")
    if dataset_root_env:
        cfg["paths"]["dataset_root"] = dataset_root_env

    device_env = os.environ.get("DEVICE")
    if device_env:
        cfg["training"]["device"] = device_env

    return cfg


def resolve_path(relative: str) -> Path:
    """Resolve a path from config relative to the project root."""
    p = Path(relative)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def get_device(cfg: dict[str, Any] | None = None) -> str:
    import torch

    cfg = cfg or load_config()
    device_setting = cfg["training"].get("device", "auto")
    if device_setting != "auto":
        return device_setting
    return "cuda" if torch.cuda.is_available() else "cpu"
