# scripts/run_backend.ps1
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
