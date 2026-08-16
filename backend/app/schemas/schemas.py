from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    status: str
    dataset_is_synthetic_placeholder: bool
    device: str


class ModelInfo(BaseModel):
    name: str
    is_trained: bool
    parameters: Optional[int] = None
    model_size_mb: Optional[float] = None


class RestoreResponse(BaseModel):
    result_id: int
    model: str
    is_trained_checkpoint: bool
    device: str
    input_image_url: str
    restored_image_url: str
    ground_truth_image_url: Optional[str] = None
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    lpips: Optional[float] = None
    mae: Optional[float] = None
    mse: Optional[float] = None
    inference_time_seconds: float
    note: Optional[str] = None


class MetricsResult(BaseModel):
    psnr: Optional[float]
    ssim: Optional[float]
    lpips: Optional[float]
    mae: Optional[float]
    mse: Optional[float]


class ExperimentCreate(BaseModel):
    model_name: str
    dev_mode: bool = True
    epochs: Optional[int] = None


class ExperimentOut(BaseModel):
    id: int
    experiment_uid: str
    model_name: str
    dataset_version: str
    dev_mode: bool
    epochs: Optional[int]
    best_val_psnr: Optional[float]
    elapsed_seconds: Optional[float]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ResultOut(BaseModel):
    id: int
    model_name: str
    psnr: Optional[float]
    ssim: Optional[float]
    lpips: Optional[float]
    mae: Optional[float]
    mse: Optional[float]
    inference_time_seconds: Optional[float]
    is_trained_checkpoint: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DashboardSummary(BaseModel):
    total_images_processed: int
    avg_psnr: Optional[float]
    avg_ssim: Optional[float]
    avg_lpips: Optional[float]
    avg_inference_time_seconds: Optional[float]
    best_performing_model: Optional[str]
    has_results: bool
