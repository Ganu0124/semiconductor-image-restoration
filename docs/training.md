# Training

## CLI

```bash
# Default: U-Net, DEV_MODE (from configs/config.yaml)
python ml/training/train.py

# Explicit model + mode
python ml/training/train.py --model unet --dev-mode true
python ml/training/train.py --model swinir --dev-mode false --epochs 60

# Resume from a checkpoint
python ml/training/train.py --model unet --resume models/unet/last_model.pth
```

Or trigger a run from the dashboard's **Experiments** page (`POST /api/experiments`),
which runs the same `ml/training/train.py::train()` function as a FastAPI background
task and writes the same JSON log + checkpoint.

## What the pipeline does

1. Loads `train`/`val` splits via `ml/datasets/paired_dataset.py` (real files from
   `paths.dataset_root`), with random-crop patches + flip/rotate augmentation on train.
2. In `DEV_MODE`, subsets to 16 train / 8 val samples, a small patch size
   (`training.dev_patch_size`), small batch (`training.dev_batch_size`), and few
   epochs (`training.dev_epochs`) — this is what makes local iteration fast.
3. Trains with L1 loss (chosen over L2/MSE because it's more robust to outlier
   pixels — relevant for preserving sharp defect edges rather than smoothing them
   out) and a cosine-annealed learning rate.
4. Validates every epoch: real PSNR/SSIM on the val set, val loss.
5. Saves `last_model.pth` every epoch and `best_model.pth` whenever val PSNR improves
   (both under `models/<model_name>/`).
6. Early-stops after `training.early_stopping_patience` epochs without
   improvement (skipped in DEV_MODE, since dev runs are intentionally short).
7. Writes a JSON log to `experiments/<model>_<timestamp>.json` with per-epoch
   train/val loss, val PSNR/SSIM, elapsed time, and device used — this is what
   powers the Analytics page's training-loss chart and the Experiments table.

## Full-scale training on a real dataset

1. Point `configs/config.yaml -> paths.dataset_root` (or `DATASET_ROOT`) at your
   real dataset in the layout described in `docs/dataset.md`.
2. Set `DEV_MODE=false` (env var) or pass `--dev-mode false`.
3. Tune `training.patch_size`, `training.batch_size`, `training.epochs`,
   `training.learning_rate` in `configs/config.yaml` for your dataset size/hardware.
4. For SwinIR at a competitive scale, increase `models.swinir.embed_dim` and
   `models.swinir.depths` — the current defaults are intentionally small so the
   model trains on CPU inside DEV_MODE. Increasing them requires a CUDA GPU for
   reasonable training time; set `training.device: cuda` or leave `auto`.
5. Run on a machine with a GPU: `training.device: auto` will pick `cuda` automatically
   if `torch.cuda.is_available()`.

## Checkpoint format

```python
{
    "epoch": int,
    "model_state": state_dict,
    "optimizer_state": state_dict,
    "best_val_psnr": float,
    "model_name": str,
    "config_snapshot": dict,   # models.<name> config at time of training
}
```
