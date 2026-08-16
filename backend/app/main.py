from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ml.utils.config import load_config, resolve_path
from backend.app.db.session import init_db
from backend.app.routers import (analytics_router, evaluate_router, experiments_router,
                                  models_router, reports_router, restore_router, system)

cfg = load_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    storage_dir = resolve_path("backend/storage")
    (storage_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (storage_dir / "restored").mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(
    title=cfg["project"]["name"],
    description=cfg["project"]["subtitle"],
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg["api"]["cors_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(system.router, prefix="/api", tags=["system"])
app.include_router(models_router.router, prefix="/api", tags=["models"])
app.include_router(restore_router.router, prefix="/api", tags=["restoration"])
app.include_router(evaluate_router.router, prefix="/api", tags=["evaluation"])
app.include_router(experiments_router.router, prefix="/api", tags=["experiments"])
app.include_router(analytics_router.router, prefix="/api", tags=["analytics"])
app.include_router(reports_router.router, prefix="/api", tags=["reports"])

# Static mounts so the frontend can render actual dataset / result images
dataset_root = resolve_path(cfg["paths"]["dataset_root"])
dataset_root.mkdir(parents=True, exist_ok=True)
app.mount("/api/dataset/image", StaticFiles(directory=str(dataset_root)), name="dataset-images")

storage_dir = resolve_path("backend/storage")
storage_dir.mkdir(parents=True, exist_ok=True)
app.mount("/api/storage", StaticFiles(directory=str(storage_dir)), name="storage")


@app.get("/")
def root():
    return {"project": cfg["project"]["name"], "subtitle": cfg["project"]["subtitle"], "docs": "/docs"}
