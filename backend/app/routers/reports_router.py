from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from sqlalchemy.orm import Session

from ml.utils.config import load_config, resolve_path
from backend.app.db.models import ReportRecord, ResultRecord
from backend.app.db.session import get_db

router = APIRouter()
cfg = load_config()


def _fmt(v, suffix=""):
    return f"{v:.4f}{suffix}" if isinstance(v, (int, float)) else "N/A"


@router.post("/reports/{result_id}")
def generate_report(result_id: int, db: Session = Depends(get_db)):
    result = db.query(ResultRecord).filter_by(id=result_id).first()
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")

    reports_dir = resolve_path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / f"inspection_report_{result_id}.pdf"

    c = canvas.Canvas(str(out_path), pagesize=A4)
    width, height = A4
    y = height - 25 * mm

    c.setFont("Helvetica-Bold", 18)
    c.drawString(20 * mm, y, cfg["project"]["name"])
    y -= 8 * mm
    c.setFont("Helvetica", 10)
    c.drawString(20 * mm, y, cfg["project"]["subtitle"])
    y -= 12 * mm

    c.setFont("Helvetica-Bold", 13)
    c.drawString(20 * mm, y, "Inspection Report")
    y -= 8 * mm

    c.setFont("Helvetica", 10)
    fields = [
        ("Result ID", result.id),
        ("Model", result.model_name),
        ("Trained checkpoint used", "Yes" if result.is_trained_checkpoint else "No (untrained weights)"),
        ("Device", result.device),
        ("Dataset version", "synthetic-placeholder-v1" if cfg["dataset"]["is_synthetic_placeholder"] else "user dataset"),
        ("Timestamp", result.created_at.isoformat()),
        ("PSNR (higher is better)", _fmt(result.psnr, " dB")),
        ("SSIM (higher is better)", _fmt(result.ssim)),
        ("LPIPS (lower is better)", _fmt(result.lpips)),
        ("MAE", _fmt(result.mae)),
        ("MSE", _fmt(result.mse)),
        ("Inference time", _fmt(result.inference_time_seconds, " s")),
    ]
    for label, value in fields:
        c.drawString(20 * mm, y, f"{label}:")
        c.drawString(90 * mm, y, str(value))
        y -= 7 * mm

    # Embed images if available
    try:
        img_y = y - 10 * mm
        img_size = 55 * mm
        x = 20 * mm
        for label, path in [
            ("Input", result.input_path),
            ("Restored", result.restored_path),
            ("Ground Truth", result.ground_truth_path),
        ]:
            if path and Path(path).exists():
                c.setFont("Helvetica-Bold", 9)
                c.drawString(x, img_y + img_size + 3, label)
                c.drawImage(path, x, img_y, width=img_size, height=img_size, preserveAspectRatio=True)
                x += img_size + 8 * mm
    except Exception:
        pass  # image embed is best-effort; report still generated with metrics

    c.setFont("Helvetica-Oblique", 8)
    c.drawString(20 * mm, 15 * mm,
                 f"Generated {datetime.now(timezone.utc).isoformat()} — SEMI-VISION AI. "
                 f"{'Dataset is a synthetic placeholder; replace with real data before production use.' if cfg['dataset']['is_synthetic_placeholder'] else ''}")
    c.showPage()
    c.save()

    report_record = ReportRecord(result_id=result_id, path=str(out_path))
    db.add(report_record)
    db.commit()

    return {"report_id": report_record.id, "download_url": f"/api/reports/download/{report_record.id}"}


@router.get("/reports/download/{report_id}")
def download_report(report_id: int, db: Session = Depends(get_db)):
    row = db.query(ReportRecord).filter_by(id=report_id).first()
    if not row or not Path(row.path).exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(row.path, media_type="application/pdf", filename=Path(row.path).name)
