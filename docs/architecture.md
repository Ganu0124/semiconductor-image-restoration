# Architecture

## System flow

```
Semiconductor Inspection Image
            |
       Input / Upload  (React dashboard -> POST /api/restore)
            |
    Image Preprocessing (grayscale normalize, ml/inference/inference.py)
            |
     AI Image Restoration (U-Net or SwinIR-Lite, ml/models/*)
            |
      Restored Image (saved to backend/storage/restored/)
            |
 PSNR / SSIM / LPIPS Evaluation (ml/evaluation/metrics.py, only if ground truth given)
            |
     Performance Analysis (SQLite results table -> /api/analytics/*)
            |
     Inspection Report (PDF via reportlab, /api/reports/{id})
```

## Components

- **ml/** — framework-agnostic core: dataset generation/analysis, PyTorch models,
  training loop, inference, metrics. Runnable standalone from the CLI; also imported
  directly by the backend so the API performs the exact same code path as the CLI
  (no duplicated/mocked logic).
- **backend/** — FastAPI app (`backend/app/main.py`) with routers per concern
  (system, models, restore, evaluate, experiments, analytics, reports), SQLAlchemy
  models backed by SQLite (`backend/semi_vision.db`), and a thin service layer
  (`backend/app/services/restoration_service.py`) that bridges HTTP requests to
  `ml/inference` and `ml/evaluation`.
- **frontend/** — React + Vite dashboard. Talks to the backend only via `/api/*`
  (proxied to `localhost:8000` in dev, same-origin in production). No client-side
  mock data; every number comes from a live API response.
- **configs/config.yaml** — single source of truth for paths, dataset assumptions,
  degradation parameters, training hyperparameters, and model architecture sizes.
  Nothing is hardcoded at call sites; `ml/utils/config.py` loads and (optionally)
  environment-overrides this file.

## Data flow for a single restoration request

1. Browser uploads a file to `POST /api/restore` (multipart form: `file`, `model`,
   optional `ground_truth_split` + `ground_truth_filename`).
2. `restore_router.py` saves the upload, records an `ImageRecord` row, calls
   `restoration_service.run_restoration()`.
3. `run_restoration()` calls `ml/inference/inference.py::restore_image()`, which
   loads (and caches) the requested model + checkpoint, runs a real forward pass,
   and times it.
4. If a ground-truth path was supplied and matches in shape, `ml/evaluation/metrics.py`
   computes real PSNR/SSIM/LPIPS/MAE/MSE; otherwise all five are `None`.
5. A `ResultRecord` row is written to SQLite; the API responds with URLs to the
   input/restored/ground-truth images (served via FastAPI `StaticFiles` mounts) and
   the metrics (or `None` — the frontend renders "No results available" for `None`).
6. The Reports page can turn any `ResultRecord` into a PDF via `POST /api/reports/{id}`.

## Why U-Net + a Swin-style model, both selectable

The dashboard's model dropdown (`GET /api/models`) is populated from
`configs/config.yaml -> models.available`, and `ml/models/registry.py::build_model()`
constructs whichever the caller asks for. Adding a third model means adding one
`nn.Module` + one registry branch + one config block — nothing else in the backend
or frontend needs to change.
