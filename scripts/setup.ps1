# scripts/setup.ps1 — one-shot environment setup
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python ml\datasets\generate_synthetic_dataset.py
python ml\datasets\dataset_analyzer.py
Write-Host "Setup complete. Next: scripts\train.ps1, then scripts\run_backend.ps1 and scripts\run_frontend.ps1"
