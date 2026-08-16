# KLA Requirements — Phase 1

This document maps the project brief's "KLA Requirements" list to what is actually
implemented, and marks anything without an official published specification as
**TBD — Official specification required**, per the instruction to never invent
requirements this project wasn't given.

| # | Requirement | Status |
|---|---|---|
| 1 | Problem statement and semiconductor-inspection context | ✅ See README + `docs/architecture.md`. General context: semiconductor inspection imagery (SEM/optical) is often degraded by sensor noise and limited resolution, which can obscure real defects and cause false positives/negatives in automated inspection. |
| 2 | Image degradation caused by noise | ✅ Implemented: Gaussian, Poisson (shot noise), sensor-like (read noise + row bias) — `ml/preprocessing/degradation.py` |
| 3 | Reduced spatial resolution | ✅ Implemented: configurable downsample/upsample — `ml/preprocessing/degradation.py::downsample_upsample` |
| 4 | Expected restoration output | ✅ Same-dimension restored image maximizing structural fidelity (PSNR/SSIM) to ground truth while preserving defects — see `docs/evaluation.md` |
| 5 | Solution workflow | ✅ See `docs/architecture.md` flow diagram |
| 6 | Dataset structure | ⚠️ **TBD — Official specification required.** This project defines and uses a `clean/`+`degraded/` paired layout (`docs/dataset.md`) for its synthetic placeholder; the real KLA dataset's actual structure was not provided |
| 7 | Image dimensions | ⚠️ **TBD — Official specification required.** No fixed dimension is assumed anywhere in the pipeline; the dataset analyzer reports whatever is actually present |
| 8 | Possible AI/ML/deep-learning approaches | ✅ U-Net (baseline, implemented+trained) and a SwinIR-style windowed self-attention model (advanced, implemented+trained) — see `docs/training.md` |
| 9 | SSIM | ✅ Implemented, real computation — `ml/evaluation/metrics.py` |
| 10 | PSNR | ✅ Implemented, real computation |
| 11 | LPIPS | ✅ Implemented; requires downloadable pretrained weights, else reported as unavailable (never faked) |
| 12 | Model performance | ✅ Real, measured only — Model Comparison page, `/api/models/compare` |
| 13 | Inference requirements | ⚠️ **TBD — Official specification required** (e.g. target latency/throughput on production hardware). Current measured baseline: see Experiments/Analytics pages for actual CPU inference times on this environment |
| 14 | KLA knowledge-session clarifications | ⚠️ **TBD — Official specification required.** No knowledge-session notes were provided to this project |
| 15 | Phase 1 submission requirements | ⚠️ **TBD — Official specification required.** See `docs/phase1.md` for what this repository currently delivers, pending official submission criteria |

## Note on the dataset

Per the "no fake data" and "do not invent dataset information" requirements, this
project explicitly **does not** claim the synthetic placeholder dataset represents
real KLA/semiconductor imagery anywhere in the code, UI, or docs — every surface
that shows dataset-derived numbers also shows a "synthetic placeholder" indicator
(dashboard banner, `is_synthetic_placeholder` API field, this doc).
