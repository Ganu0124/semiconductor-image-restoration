from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (Column, DateTime, Float, ForeignKey, Integer, String, Text, Boolean)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def utcnow():
    return datetime.now(timezone.utc)


class ImageRecord(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    path = Column(String, nullable=False)
    width = Column(Integer)
    height = Column(Integer)
    channels = Column(String)
    split = Column(String, default="upload")  # train/val/test/upload
    created_at = Column(DateTime, default=utcnow)


class ModelRecord(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, unique=True)
    version = Column(String, default="v1")
    parameters = Column(Integer)
    model_size_mb = Column(Float)
    is_trained = Column(Boolean, default=False)
    created_at = Column(DateTime, default=utcnow)


class ExperimentRecord(Base):
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, index=True)
    experiment_uid = Column(String, unique=True, index=True)
    model_id = Column(Integer, ForeignKey("models.id"))
    model_name = Column(String)
    dataset_version = Column(String, default="synthetic-placeholder-v1")
    patch_size = Column(Integer)
    noise_level = Column(String)
    scale_factor = Column(Integer)
    learning_rate = Column(Float)
    batch_size = Column(Integer)
    epochs = Column(Integer)
    gpu = Column(String)
    dev_mode = Column(Boolean, default=True)
    best_val_psnr = Column(Float, nullable=True)
    elapsed_seconds = Column(Float, nullable=True)
    created_at = Column(DateTime, default=utcnow)

    results = relationship("ResultRecord", back_populates="experiment")


class ResultRecord(Base):
    __tablename__ = "results"

    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=True)
    image_id = Column(Integer, ForeignKey("images.id"), nullable=True)
    model_name = Column(String)
    input_path = Column(String)
    restored_path = Column(String)
    ground_truth_path = Column(String, nullable=True)
    psnr = Column(Float, nullable=True)
    ssim = Column(Float, nullable=True)
    lpips = Column(Float, nullable=True)
    mae = Column(Float, nullable=True)
    mse = Column(Float, nullable=True)
    inference_time_seconds = Column(Float, nullable=True)
    is_trained_checkpoint = Column(Boolean, default=False)
    device = Column(String)
    created_at = Column(DateTime, default=utcnow)

    experiment = relationship("ExperimentRecord", back_populates="results")


class ReportRecord(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    result_id = Column(Integer, ForeignKey("results.id"))
    path = Column(String)
    created_at = Column(DateTime, default=utcnow)
