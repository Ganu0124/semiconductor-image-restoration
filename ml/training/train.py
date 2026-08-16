"""Training pipeline for restoration models (U-Net / SwinIR-Lite).

Supports:
  - dataset loading & batching (ml/datasets/paired_dataset.py)
  - train/val loops with L1 loss
  - cosine LR scheduling
  - checkpointing (best + last) under models/<name>/
  - resume from checkpoint
  - early stopping
  - DEV_MODE (tiny subset / patch / batch / epochs) for fast local iteration
  - simple JSON experiment log under experiments/

Usage (PowerShell / bash):
    python ml/training/train.py --model unet
    python ml/training/train.py --model unet --dev-mode false --epochs 60
    python ml/training/train.py --model swinir --resume models/swinir/last_model.pth
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.datasets.paired_dataset import PairedRestorationDataset  # noqa: E402
from ml.evaluation.metrics import psnr, ssim  # noqa: E402
from ml.models.registry import build_model, checkpoint_path  # noqa: E402
from ml.utils.config import get_device, load_config, resolve_path  # noqa: E402


def _bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes")


def build_dataloaders(cfg: dict, dev_mode: bool):
    dataset_root = resolve_path(cfg["paths"]["dataset_root"])
    extensions = cfg["dataset"]["image_extensions"]
    patch_size = cfg["training"]["dev_patch_size"] if dev_mode else cfg["training"]["patch_size"]
    batch_size = cfg["training"]["dev_batch_size"] if dev_mode else cfg["training"]["batch_size"]

    train_ds = PairedRestorationDataset(dataset_root, "train", extensions, patch_size=patch_size)
    val_ds = PairedRestorationDataset(dataset_root, "val", extensions, patch_size=patch_size, augment=False)

    if dev_mode:
        # Small dataset subset per requirement #7
        subset_n = min(len(train_ds), 16)
        train_ds.filenames = train_ds.filenames[:subset_n]
        val_subset_n = min(len(val_ds), 8)
        val_ds.filenames = val_ds.filenames[:val_subset_n]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=0 if dev_mode else cfg["training"]["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0 if dev_mode else cfg["training"]["num_workers"])
    return train_loader, val_loader


def validate(model, loader, device) -> dict:
    model.eval()
    psnrs, ssims, losses = [], [], []
    loss_fn = torch.nn.L1Loss()
    with torch.no_grad():
        for batch in loader:
            degraded = batch["degraded"].to(device)
            clean = batch["clean"].to(device)
            restored = model(degraded)
            losses.append(loss_fn(restored, clean).item())
            for i in range(restored.shape[0]):
                r = restored[i, 0].cpu().numpy()
                c = clean[i, 0].cpu().numpy()
                psnrs.append(psnr(c, r))
                ssims.append(ssim(c, r))
    return {
        "val_loss": float(sum(losses) / max(len(losses), 1)),
        "val_psnr": float(sum(psnrs) / max(len(psnrs), 1)),
        "val_ssim": float(sum(ssims) / max(len(ssims), 1)),
    }


def train(model_name: str, dev_mode: bool, epochs: int | None, resume: str | None):
    cfg = load_config()
    device = get_device(cfg)
    print(f"[train] model={model_name} dev_mode={dev_mode} device={device}")

    train_loader, val_loader = build_dataloaders(cfg, dev_mode)
    print(f"[train] train samples={len(train_loader.dataset)} val samples={len(val_loader.dataset)}")

    model = build_model(model_name, cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])

    n_epochs = epochs or (cfg["training"]["dev_epochs"] if dev_mode else cfg["training"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(n_epochs, 1))
    loss_fn = torch.nn.L1Loss()

    start_epoch = 0
    best_val_psnr = -float("inf")
    patience_counter = 0

    ckpt_dir = checkpoint_path(model_name, cfg).parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best_model.pth"
    last_path = ckpt_dir / "last_model.pth"

    if resume:
        resume_path = resolve_path(resume)
        if resume_path.exists():
            ckpt = torch.load(resume_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_psnr = ckpt.get("best_val_psnr", best_val_psnr)
            print(f"[train] resumed from {resume_path} at epoch {start_epoch}")
        else:
            print(f"[train] WARNING resume path {resume_path} not found, starting fresh")

    history = []
    t0 = time.time()

    for epoch in range(start_epoch, n_epochs):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            degraded = batch["degraded"].to(device)
            clean = batch["clean"].to(device)

            optimizer.zero_grad()
            restored = model(degraded)
            loss = loss_fn(restored, clean)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()
        val_metrics = validate(model, val_loader, device)
        train_loss = sum(epoch_losses) / max(len(epoch_losses), 1)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(f"[train] epoch {epoch+1}/{n_epochs} "
              f"train_loss={train_loss:.4f} val_loss={val_metrics['val_loss']:.4f} "
              f"val_psnr={val_metrics['val_psnr']:.2f} val_ssim={val_metrics['val_ssim']:.4f}")

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_psnr": best_val_psnr,
            "model_name": model_name,
            "config_snapshot": cfg["models"][model_name],
        }
        torch.save(ckpt, last_path)

        if val_metrics["val_psnr"] > best_val_psnr:
            best_val_psnr = val_metrics["val_psnr"]
            ckpt["best_val_psnr"] = best_val_psnr
            torch.save(ckpt, best_path)
            patience_counter = 0
            print(f"[train]   -> new best model saved (val_psnr={best_val_psnr:.2f})")
        else:
            patience_counter += 1

        if not dev_mode and patience_counter >= cfg["training"]["early_stopping_patience"]:
            print(f"[train] early stopping at epoch {epoch+1} (no improvement for "
                  f"{cfg['training']['early_stopping_patience']} epochs)")
            break

    elapsed = time.time() - t0

    experiments_dir = resolve_path(cfg["paths"]["experiments_dir"])
    experiments_dir.mkdir(parents=True, exist_ok=True)
    exp_id = f"{model_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    log_path = experiments_dir / f"{exp_id}.json"
    with open(log_path, "w") as f:
        json.dump({
            "experiment_id": exp_id,
            "model": model_name,
            "dev_mode": dev_mode,
            "epochs_run": len(history),
            "elapsed_seconds": elapsed,
            "best_val_psnr": best_val_psnr,
            "history": history,
            "device": device,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, f, indent=2)
    print(f"[train] DONE in {elapsed:.1f}s. Log: {log_path}. Best checkpoint: {best_path}")
    return log_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cfg = load_config()
    parser.add_argument("--model", default=cfg["models"]["default"], choices=cfg["models"]["available"])
    parser.add_argument("--dev-mode", default=str(cfg["training"]["dev_mode"]), type=_bool)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    train(args.model, args.dev_mode, args.epochs, args.resume)


if __name__ == "__main__":
    main()
