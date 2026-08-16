# Dashboard

**SEMI-VISION AI** — React + Vite, dark navy / cyan industrial theme
(`frontend/src/styles/theme.css`), served in dev on `http://localhost:5173`
(proxies `/api/*` to the backend on `:8000`; in production, serve the built
`frontend/dist/` from the same origin as the API or configure your reverse proxy).

## Pages

- **Dashboard** — real aggregate metrics (`/api/analytics/summary`), system status.
- **Image Restoration** — upload or pick a test-split image, choose model, run
  real restoration, view before/after slider + triplet (degraded/restored/ground
  truth), real metrics, download.
- **Evaluation** — ad-hoc PSNR/SSIM/LPIPS/MAE/MSE between two uploaded images,
  plus a table of recent stored results.
- **Model Comparison** — U-Net vs SwinIR-Lite table (PSNR/SSIM/LPIPS/inference
  time/parameters/size), best model highlighted only when real data supports it.
- **Dataset Explorer** — real dataset-analyzer stats + image gallery by split.
- **Experiments** — launch a real training run, see history with real hyperparameters
  and results.
- **Analytics** — charts (via `recharts`) of result trends and training loss curves,
  built from real stored data.
- **Reports** — generate/download a PDF inspection report from any stored result.
- **Settings** — read-only view of current backend configuration.

## No fake data policy (enforced in the UI layer)

Every page fetches from the live backend on mount; there is no mock/sample data
baked into the frontend. Components that display a metric always check for
`null`/empty and render **"No results available — run an experiment."** rather
than a placeholder number (see `MetricCard.jsx`, and inline checks in
`ImageRestoration.jsx`, `Evaluation.jsx`, `ModelComparison.jsx`).

## Design tokens

Colors, typography, and spacing are defined once as CSS variables in
`frontend/src/styles/theme.css` (`--bg-void`, `--accent-cyan`, `--font-display`,
etc.) — component files reference these variables rather than hardcoding hex
values, so the theme can be retinted from one file.
