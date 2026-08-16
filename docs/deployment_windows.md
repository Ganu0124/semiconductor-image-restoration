# Windows Setup (PowerShell)

```powershell
# From the project root: semiconductor-image-restoration\

# 1) Create environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2) Install dependencies
pip install -r requirements.txt

# 3) Generate the synthetic placeholder dataset
#    (skip this once you've pointed DATASET_ROOT at a real dataset)
python ml\datasets\generate_synthetic_dataset.py

# 4) Run dataset analysis
python ml\datasets\dataset_analyzer.py

# 5) Train a model
#    DEV_MODE is on by default (configs\config.yaml). Override per-run:
$env:DEV_MODE = "true"
python ml\training\train.py --model unet
python ml\training\train.py --model swinir

#    Full training once you have a real dataset:
$env:DEV_MODE = "false"
python ml\training\train.py --model unet --epochs 60

# 6) Run inference from the CLI
python ml\inference\inference.py --input data\synthetic\test\degraded\test_0000.png --output out.png --model unet

# 7) Start the backend (new terminal, venv activated)
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 8) Start the frontend (another new terminal)
cd frontend
npm install
npm run dev
# Dashboard: http://localhost:5173
# API docs:  http://localhost:8000/docs

# 9) Run tests
pytest tests\ml -q
pytest tests\backend -q
```

## Environment variables (optional overrides)

```powershell
$env:DEV_MODE = "false"          # true|false — overrides configs\config.yaml training.dev_mode
$env:DATASET_ROOT = "C:\path\to\your\real\dataset"
$env:DEVICE = "cuda"             # auto|cpu|cuda
```

## Common issues

- **`uvicorn` not found**: ensure the venv is activated (`.\venv\Scripts\Activate.ps1`)
  and `pip install -r requirements.txt` completed without errors.
- **`npm` not found**: install Node.js LTS from nodejs.org, restart PowerShell.
- **Port already in use**: change `--port` for uvicorn, or `frontend\vite.config.js`'s
  `server.port` for the dashboard.
- **LPIPS download fails (no internet)**: expected in an offline environment — the
  API returns `lpips: null` rather than a fake value; PSNR/SSIM/MAE/MSE are unaffected.
