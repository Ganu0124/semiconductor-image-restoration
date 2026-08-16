# Evaluation

All metrics are computed in `ml/evaluation/metrics.py` — nothing displayed by the
API or dashboard is fabricated. If a metric cannot be computed (no ground truth
supplied, or the LPIPS backbone can't be downloaded), the API returns `null` for
that field and the dashboard renders "No results available."

## Metrics

| Metric | Direction | Library | Notes |
|---|---|---|---|
| PSNR | higher = better | scikit-image | capped at 100 dB for pixel-identical inputs (true value is infinite; JSON can't represent `inf`) |
| SSIM | higher = better | scikit-image | `data_range=1.0`, grayscale |
| LPIPS | lower = better | `lpips` (AlexNet backbone) | requires downloading pretrained weights once; returns `null`, not 0, if unavailable offline |
| MAE | lower = better | numpy | mean absolute pixel error |
| MSE | lower = better | numpy | mean squared pixel error |

Inference time is measured with `time.perf_counter()` around the actual forward
pass (`ml/evaluation/metrics.py::Timer`, also inlined in `ml/inference/inference.py`),
not estimated.

## Where evaluation happens

- **During restoration** (`POST /api/restore`): if `ground_truth_split` +
  `ground_truth_filename` are supplied (the dashboard's "pick from test split"
  flow), metrics are computed against that ground truth and stored with the result.
- **Ad-hoc** (`POST /api/evaluate`, the Evaluation page): upload any two
  same-size images (e.g. a restored output and its known-good reference) and get
  metrics back directly, independent of any stored experiment.

## Defect preservation (semiconductor-specific)

Per the project's core requirement, this system optimizes for **structural
fidelity over visual beautification**:

- The U-Net and SwinIR-Lite models are both trained with **residual learning**
  (`output = input + predicted_residual`, clamped to valid range) rather than
  generating an image from scratch — this biases the model toward correcting
  noise/blur while leaving real structure (including defects) untouched, instead
  of hallucinating a "cleaner-looking" image.
- L1 loss (not L2) is used during training because it penalizes large per-pixel
  deviations less aggressively than L2, which in practice avoids over-smoothing
  sharp edges (die boundaries, via edges, scratch defects).
- The dashboard's before/after slider and side-by-side triplet (degraded /
  restored / ground truth) let a human inspector directly verify that structure
  was preserved, rather than trusting a single scalar metric.

**Not yet implemented** (documented rather than silently skipped, per project
requirement #9): a dedicated defect-preservation score (e.g. IoU between defect
masks pre/post restoration), difference maps, and edge-comparison overlays. The
synthetic placeholder dataset has no defect segmentation labels to build this
against. To add it once real defect labels are available:
1. Extend `ml/datasets/paired_dataset.py` to also load a defect mask.
2. Add a mask-aware metric (e.g. `defect_iou`) to `ml/evaluation/metrics.py`.
3. Surface it in `backend/app/schemas/schemas.py::RestoreResponse` and the
   Image Restoration page's metrics grid.
