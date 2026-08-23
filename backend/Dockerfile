FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ml/ ml/
COPY backend/ backend/
COPY configs/ configs/
COPY models/ models/
COPY scripts/ scripts/

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["sh", "-c", "python scripts/download_real_dataset.py && uvicorn backend.app.main:app --host 0.0.0.0 --port 8000"]