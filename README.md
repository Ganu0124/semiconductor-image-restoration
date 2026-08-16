# SEMI-VISION AI
### AI-Powered Semiconductor Image Restoration & Inspection

A complete, working, end-to-end system: dataset analysis → configurable degradation
→ deep-learning restoration (U-Net baseline + SwinIR-style advanced model) → real
PSNR/SSIM/LPIPS/MAE/MSE evaluation → FastAPI backend with SQLite → React dashboard →
PDF inspection reports.

> ⚠️ **No real semiconductor dataset was supplied when this project was built.**
> Every ML/backend/dashboard component runs against a **synthetic placeholder
> dataset** (`data/synthetic/`, procedurally generated die/via/grid patterns —
> see `ml/datasets/generate_synthetic_dataset.py`) purely so the full pipeline is
> exercised with real code and real, computed metrics — nothing is faked. Swap in
> your real dataset by pointing `paths.dataset_root` in `configs/config.yaml` (or
> the `DATASET_ROOT` env var) at it; nothing else needs to change. See
> [`docs/dataset.md`](docs/dataset.md).

---

## What's real vs. placeholder

| Component | Status |
|---|---|
| Dataset analyzer | ✅ Real — inspects actual files on disk, never invents stats |
| Degradation (noise / downsampling) | ✅ Real, configurable |
| U-Net baseline | ✅ Real, trains and checkpoints |
| SwinIR-style advanced model (SwinIR-Lite) | ✅ Real, windowed self-attention, trains and checkpoints |
| PSNR / SSIM / MAE / MSE | ✅ Real computed values (scikit-image / numpy) |
| LPIPS | ✅ Real when the AlexNet backbone can be downloaded; otherwise reported as `null` ("unavailable"), never faked |
| FastAPI backend + SQLite | ✅ Real, tested end-to-end |
| React dashboard | ✅ Real, calls the live backend, no mock data |
| PDF inspection reports | ✅ Real, generated with reportlab from actual result rows |
| Dataset content | ⚠️ **Synthetic placeholder** — replace with your real dataset |
| Model performance numbers you'll see initially | ⚠️ Trained only in `DEV_MODE` on the placeholder data (a few epochs, tiny subset) — retrain properly on your real dataset for meaningful numbers |

---

## Quick start (Windows / PowerShell)

```powershell
# 1. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the synthetic placeholder dataset (skip once you have a real one)
python ml/datasets/generate_synthetic_dataset.py

# 4. Inspect the dataset
python ml/datasets/dataset_analyzer.py

# 5. Train the baseline model (DEV_MODE is on by default — a few seconds on CPU)
python ml/training/train.py --model unet
python ml/training/train.py --model swinir

# 6. Run inference from the CLI (optional sanity check)
python ml/inference/inference.py --input data/synthetic/test/degraded/test_0000.png --output out.png --model unet

# 7. Start the backend
cd backend
uvicorn app.main:app --reload --port 8000
# in a new terminal:
cd ..

# 8. Start the frontend
cd frontend
npm install
npm run dev
# open http://localhost:5173
```

Full Windows command reference: [`docs/deployment_windows.md`](docs/deployment_windows.md).

## Quick start (Docker)

```bash
docker compose up --build
```
Docker is optional — the project runs the same way with plain Python + Node. See [`docs/architecture.md`](docs/architecture.md).

---

## Project structure

```
semiconductor-image-restoration/
├── frontend/            React + Vite dashboard (SEMI-VISION AI)
├── backend/              FastAPI + SQLite REST API
├── ml/
│   ├── datasets/         synthetic dataset generator, dataset analyzer, PyTorch loader
│   ├── models/            U-Net, SwinIR-Lite, model registry
│   ├── training/          training loop, checkpointing, early stopping
│   ├── inference/         inference pipeline used by CLI and API
│   ├── evaluation/        PSNR / SSIM / LPIPS / MAE / MSE
│   └── preprocessing/     configurable noise / downsampling degradation
├── data/                  synthetic placeholder dataset (replace with real data)
├── models/                saved checkpoints (models/unet/best_model.pth, ...)
├── experiments/           JSON training logs (one per run)
├── reports/               generated PDF inspection reports
├── configs/config.yaml    single source of truth for all paths/hyperparameters
├── docs/                  architecture, dataset, training, evaluation, API, KLA docs
├── tests/                 ml / backend / frontend tests
└── docker-compose.yml
```

## DEV_MODE

`DEV_MODE=true` (the default, set in `configs/config.yaml` under `training.dev_mode`,
overridable via the `DEV_MODE` env var) trains on a small subset with small patches,
small batches, and few epochs — this is what lets the whole system be exercised
in seconds without a GPU. Set it to `false` (or pass `--dev-mode false` to
`ml/training/train.py`) once you're training on your real dataset for real.

## Known limitations / TBDs

- **LPIPS** requires downloading pretrained AlexNet weights; if your environment
  has no internet access this returns `null` rather than a fake number.
- **SwinIR-Lite** is a real, working, windowed-attention model in the SwinIR/Restormer
  family, sized down to train on CPU in DEV_MODE. For a paper-scale SwinIR/Restormer,
  increase `models.swinir.embed_dim` / `depths` in `configs/config.yaml` and train on
  a CUDA GPU — the code path doesn't change.
- **Defect labels**: the dataset analyzer detects clean/degraded pairs but this
  project does not assume defect segmentation labels exist. If your real dataset
  has them, wire them into `ml/datasets/paired_dataset.py`; see `docs/evaluation.md`.
- **KLA-specific requirements**: see [`docs/KLA_REQUIREMENTS.md`](docs/KLA_REQUIREMENTS.md) —
  items without an official published spec are marked `TBD — Official specification required`.
