<div align="center">

  # 🔬 SEMI-VISION AI
  ### AI-Powered Semiconductor Image Restoration & Inspection Suite

  [![GitHub Stars](https://img.shields.io/github/stars/Ganu0124/semiconductor-image-restoration?style=for-the-badge&logo=github&color=blue)](https://github.com/Ganu0124/semiconductor-image-restoration)
  [![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
  [![React](https://img.shields.io/badge/React-18%2B-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

  <p align="center">
    <b>An industrial-grade, end-to-end computer vision and deep learning system built for automated semiconductor die, via, and grid inspection, degradation modeling, image restoration, and rigorous metric evaluation.</b>
  </p>

  <p align="center">
    <b>Author:</b> Ganesh M &bull; <b>Contact:</b> <a href="mailto:gganu8615@gmail.com">gganu8615@gmail.com</a>
  </p>

</div>

---

## ⚡ System Architecture & Pipeline Workflow

```text
[ Raw / Synthetic Dataset ] 
         │
         ▼
[ Configurable Degradation Engine ] ──(Noise, Blur, Downsampling)
         │
         ▼
[ Deep Learning Restoration ] ───────► [ U-Net Baseline ] & [ SwinIR-Lite (Transformer) ]
         │
         ▼
[ Rigorous Metrics Evaluation ] ─────► PSNR | SSIM | LPIPS | MAE | MSE
         │
         ▼
[ FastAPI Backend + SQLite ] ────────► REST Endpoints & Async Processing
         │
         ▼
[ React + Vite Dashboard ] ──────────► Interactive UI & Real-Time Inspection
         │
         ▼
[ ReportLab PDF Generator ] ─────────► Automated Exportable PDF Inspection Reports

## ⚠️ Important Note on Dataset State

> **No real semiconductor dataset was supplied when this project was built.**
> Every ML/backend/dashboard component runs against a **synthetic placeholder dataset** (`data/synthetic/`, procedurally generated die/via/grid patterns — see `ml/datasets/generate_synthetic_dataset.py`) purely so the full pipeline is exercised with real code and real, computed metrics — nothing is faked. Swap in your real dataset by pointing `paths.dataset_root` in `configs/config.yaml` (or the `DATASET_ROOT` env var) at it; nothing else needs to change. See `docs/dataset.md`.

---

## 🔍 What's Real vs. Placeholder

| Component | Status | Implementation Details |
| --- | --- | --- |
| **Dataset analyzer** | ✅ **Real** | Inspects actual files on disk, never invents stats |
| **Degradation (noise / downsampling)** | ✅ **Real** | Configurable noise insertion, blurring, and degradation models |
| **U-Net baseline** | ✅ **Real** | Fully implemented encoder-decoder, trains and checkpoints |
| **SwinIR-style advanced model (SwinIR-Lite)** | ✅ **Real** | Windowed self-attention transformer, trains and checkpoints |
| **PSNR / SSIM / MAE / MSE** | ✅ **Real** | Computed values via `scikit-image` & `numpy` |
| **LPIPS** | ✅ **Real / Fallback** | Real when the AlexNet backbone can be downloaded; otherwise reported as `null` ("unavailable"), never faked |
| **FastAPI backend + SQLite** | ✅ **Real** | Tested end-to-end REST API architecture |
| **React dashboard** | ✅ **Real** | Calls the live backend directly, no mock data |
| **PDF inspection reports** | ✅ **Real** | Generated with `reportlab` from actual result rows |
| **Dataset content** | ⚠️ **Synthetic placeholder** | Replace with your real semiconductor dataset |
| **Model performance numbers** | ⚠️ **DEV_MODE only** | Trained only in `DEV_MODE` on placeholder data (few epochs, tiny subset) — retrain properly on your real dataset |

---

## 🚀 Quick Start (Windows / PowerShell)

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

Full Windows command reference: `docs/deployment_windows.md`.

---

## 🐳 Quick Start (Docker)

```bash
docker compose up --build

```

Docker is optional — the project runs the same way with plain Python + Node. See `docs/architecture.md`.

---

## 📁 Project Structure

```text
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

---

## ⚙️ DEV_MODE

`DEV_MODE=true` (the default, set in `configs/config.yaml` under `training.dev_mode`,
overridable via the `DEV_MODE` env var) trains on a small subset with small patches,
small batches, and few epochs — this is what lets the whole system be exercised
in seconds without a GPU. Set it to `false` (or pass `--dev-mode false` to
`ml/training/train.py`) once you're training on your real dataset for real.

---

## 📌 Known limitations / TBDs

* **LPIPS** requires downloading pretrained AlexNet weights; if your environment
has no internet access this returns `null` rather than a fake number.
* **SwinIR-Lite** is a real, working, windowed-attention model in the SwinIR/Restormer
family, sized down to train on CPU in DEV_MODE. For a paper-scale SwinIR/Restormer,
increase `models.swinir.embed_dim` / `depths` in `configs/config.yaml` and train on
a CUDA GPU — the code path doesn't change.
* **Defect labels**: the dataset analyzer detects clean/degraded pairs but this
project does not assume defect segmentation labels exist. If your real dataset
has them, wire them into `ml/datasets/paired_dataset.py`; see `docs/evaluation.md`.
* **KLA-specific requirements**: see `docs/KLA_REQUIREMENTS.md` —
items without an official published spec are marked `TBD — Official specification required`.

---

## ✍️ Author

**Ganesh M**

* GitHub: [@Ganu0124](https://github.com/Ganu0124)
* Email: [gganu8615@gmail.com](https://www.google.com/search?q=mailto%3Agganu8615%40gmail.com)

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for more information.

```

```
