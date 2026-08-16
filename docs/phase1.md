# Phase 1 — Delivery Summary

## Delivered in this build

- Working synthetic-placeholder dataset generation + real dataset analyzer
- Configurable degradation pipeline (noise: Gaussian/Poisson/sensor; reduced
  spatial resolution: configurable downsample/upsample)
- Two trained restoration models: U-Net baseline (7.7M params) and SwinIR-Lite
  advanced model (windowed self-attention, ~470K params at default dev config),
  both selectable in the dashboard and API
- Real training pipeline: batching, validation, checkpointing (best + last),
  resume, cosine LR scheduling, early stopping, DEV_MODE for fast iteration
- Real evaluation: PSNR, SSIM, MAE, MSE always; LPIPS when the pretrained
  backbone is reachable
- FastAPI backend (11 endpoints) + SQLite (5 tables: images, models, experiments,
  results, reports), fully tested (`tests/backend/test_api.py`, 10/10 passing)
- React + Vite dashboard, 9 pages, dark navy/cyan industrial theme, calling the
  live backend only (no mock data), tested build (`npm run build` succeeds)
- PDF inspection report generation (reportlab) with embedded images and real metrics
- Docker support (optional) + Windows PowerShell command reference
- Test suite: `tests/ml/` (11 tests, dataset/model/metrics), `tests/backend/`
  (10 tests, full API surface)

## Verified end-to-end (this session)

1. Generated 90 synthetic image pairs across train/val/test
2. Ran the dataset analyzer against them — real counts/dimensions confirmed
3. Trained U-Net and SwinIR-Lite in DEV_MODE — real loss curves, checkpoints saved
4. Ran inference via the API (`POST /api/restore`) with a real ground-truth
   comparison — got real, non-fabricated PSNR (~20.4 dB), SSIM (~0.50), MAE, MSE;
   LPIPS returned `null` where the backbone wasn't downloadable, exactly per the
   no-fake-data requirement
5. Generated and downloaded a real PDF inspection report from that result
6. Built the frontend for production (`npm run build`) and ran it in dev mode
   against the live backend, confirmed the API proxy and endpoints work

## Explicitly out of scope / TBD for Phase 1

See `docs/KLA_REQUIREMENTS.md` for the itemized list of official-spec items that
were not provided to this project and are therefore marked TBD rather than guessed.

The real semiconductor dataset itself is the biggest open item — everything else
in this repository is built to accept it via a one-line config change
(`paths.dataset_root` / `DATASET_ROOT`) with no code changes required.
