import { useEffect, useState } from 'react'
import { api } from '../api/client.js'
import MetricCard from '../components/MetricCard.jsx'
import { AlertTriangle, CheckCircle2 } from 'lucide-react'

export default function Dashboard() {
  const [summary, setSummary] = useState(null)
  const [health, setHealth] = useState(null)
  const [datasetStats, setDatasetStats] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([api.analyticsSummary(), api.health(), api.datasetStats()])
      .then(([s, h, d]) => { setSummary(s); setHealth(h); setDatasetStats(d) })
      .catch((e) => setError(e.message))
  }, [])

  return (
    <div>
      <div className="page-header">
        <div className="page-eyebrow">Overview</div>
        <h1 className="page-title">Dashboard</h1>
        <p className="page-subtitle">Real-time restoration performance across all runs on this instance.</p>
      </div>

      {error && <div className="notice notice-amber"><AlertTriangle size={14} style={{verticalAlign:'-2px'}} /> &nbsp;Could not reach backend: {error}. Is it running on :8000?</div>}

      {health?.dataset_is_synthetic_placeholder && (
        <div className="notice notice-amber">
          <AlertTriangle size={14} style={{verticalAlign:'-2px'}} />&nbsp;
          Running against the <strong>synthetic placeholder dataset</strong> (no real semiconductor dataset supplied).
          Point <span className="mono">paths.dataset_root</span> in <span className="mono">configs/config.yaml</span> at your real dataset to replace it.
        </div>
      )}

      <div className="grid-metrics">
        <MetricCard label="Images Processed" value={summary?.total_images_processed} precision={0} />
        <MetricCard label="Avg PSNR" value={summary?.avg_psnr} unit="dB" precision={2} />
        <MetricCard label="Avg SSIM" value={summary?.avg_ssim} precision={4} />
        <MetricCard label="Avg LPIPS" value={summary?.avg_lpips} precision={4} />
        <MetricCard label="Avg Inference Time" value={summary?.avg_inference_time_seconds} unit="s" precision={3} />
        <MetricCard label="Best Performing Model" value={summary?.best_performing_model} />
      </div>

      <div className="card">
        <div className="card-title">
          {health?.status === 'ok' ? <CheckCircle2 size={15} color="var(--accent-green)" /> : null}
          System Status
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 14, fontSize: 13 }}>
          <div>
            <div className="metric-label">Backend</div>
            <span className={`badge ${health?.status === 'ok' ? 'badge-ok' : 'badge-warn'}`}>{health?.status ?? 'unreachable'}</span>
          </div>
          <div>
            <div className="metric-label">Inference Device</div>
            <span className="badge badge-info mono">{health?.device ?? '—'}</span>
          </div>
          <div>
            <div className="metric-label">Dataset Images</div>
            <span className="mono">{datasetStats?.total_images ?? '—'}</span>
          </div>
          <div>
            <div className="metric-label">Dataset Source</div>
            <span className={`badge ${health?.dataset_is_synthetic_placeholder ? 'badge-warn' : 'badge-ok'}`}>
              {health?.dataset_is_synthetic_placeholder ? 'synthetic placeholder' : 'real dataset'}
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
