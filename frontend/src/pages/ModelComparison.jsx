import { useEffect, useState } from 'react'
import { api } from '../api/client.js'

export default function ModelComparison() {
  const [data, setData] = useState(null)
  useEffect(() => { api.modelsCompare().then(setData).catch(() => {}) }, [])

  const models = data?.models ?? []
  const scored = models.filter((m) => m.avg_psnr !== null)
  const best = scored.length ? scored.reduce((a, b) => (b.avg_psnr > a.avg_psnr ? b : a)) : null

  return (
    <div>
      <div className="page-header">
        <div className="page-eyebrow">Comparison</div>
        <h1 className="page-title">Model Comparison</h1>
        <p className="page-subtitle">U-Net vs SwinIR-Lite, scored on actually-run restorations. A model with zero results shows no ranking data.</p>
      </div>

      <div className="card">
        <div className="card-title">Comparison Table</div>
        <table>
          <thead>
            <tr>
              <th>Model</th><th>Trained</th><th>Results</th><th>Avg PSNR</th><th>Avg SSIM</th>
              <th>Avg LPIPS</th><th>Avg Infer (s)</th><th>Parameters</th><th>Size (MB)</th>
            </tr>
          </thead>
          <tbody>
            {models.map((m) => (
              <tr key={m.model} className={best && m.model === best.model ? 'row-best' : ''}>
                <td style={{ textTransform: 'uppercase' }}>{m.model}{best && m.model === best.model ? ' ★' : ''}</td>
                <td>{m.is_trained ? 'yes' : 'no'}</td>
                <td>{m.num_results}</td>
                <td>{m.avg_psnr?.toFixed(2) ?? '—'}</td>
                <td>{m.avg_ssim?.toFixed(4) ?? '—'}</td>
                <td>{m.avg_lpips?.toFixed(4) ?? '—'}</td>
                <td>{m.avg_inference_time_seconds?.toFixed(3) ?? '—'}</td>
                <td>{m.parameters?.toLocaleString() ?? '—'}</td>
                <td>{m.model_size_mb?.toFixed(1) ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {!scored.length && (
          <div className="notice" style={{ marginTop: 16 }}>
            No model has scored results yet — run restorations with ground truth (Image Restoration page, pick from the test split)
            to populate this comparison, or launch training runs on the Experiments page.
          </div>
        )}
      </div>
    </div>
  )
}
