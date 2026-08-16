from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from fastapi import APIRouter

from ml.datasets.dataset_analyzer import analyze_dataset
from ml.utils.config import get_device, load_config, resolve_path
from backend.app.schemas.schemas import HealthResponse

router = APIRouter()
cfg = load_config()


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        dataset_is_synthetic_placeholder=cfg["dataset"]["is_synthetic_placeholder"],
        device=get_device(cfg),
    )


@router.get("/dataset/stats")
def dataset_stats():
    dataset_root = resolve_path(cfg["paths"]["dataset_root"])
    report = analyze_dataset(dataset_root, cfg["dataset"]["splits"], cfg["dataset"]["image_extensions"])
    report["is_synthetic_placeholder"] = cfg["dataset"]["is_synthetic_placeholder"]
    return report


@router.get("/dataset/gallery/{split}")
def dataset_gallery(split: str, limit: int = 24):
    """Lists real filenames + relative paths for the dataset explorer gallery."""
    dataset_root = resolve_path(cfg["paths"]["dataset_root"])
    clean_dir = dataset_root / split / "clean"
    degraded_dir = dataset_root / split / "degraded"
    if not clean_dir.exists():
        return {"split": split, "images": [], "note": f"No split '{split}' found."}

    items = []
    for p in sorted(clean_dir.iterdir())[:limit]:
        if not p.is_file():
            continue
        counterpart = degraded_dir / p.name
        items.append({
            "filename": p.name,
            "clean_path": f"/api/dataset/image/{split}/clean/{p.name}",
            "degraded_path": f"/api/dataset/image/{split}/degraded/{p.name}" if counterpart.exists() else None,
        })
    return {"split": split, "count": len(items), "images": items}
