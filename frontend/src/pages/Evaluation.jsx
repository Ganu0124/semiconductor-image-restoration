import { useEffect, useState } from 'react'
import { api } from '../api/client.js'
import MetricCard from '../components/MetricCard.jsx'

export default function Evaluation() {
  const [restoredFile, setRestoredFile] = useState(null)
  const [gtFile, setGtFile] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)

  useEffect(() => { api.results(20).then(setResults).catch(() => {}) }, [])

  const runEval = async () => {
    if (!restoredFile || !gtFile) return
    setLoading(true); setError(null)
    try {
      const fd = new FormData()
      fd.append('restored', restoredFile)
      fd.append('ground_truth', gtFile)
      setMetrics(await api.evaluate(fd))
    } catch (e) { setError(e.message) } finally { setLoading(false) }
  }

  return (
    <div>
      <div className="page-header">
        <div className="page-eyebrow">Evaluation</div>
        <h1 className="page-title">Evaluation</h1>
        <p className="page-subtitle">Compute PSNR / SSIM / LPIPS / MAE / MSE between any two same-size images.</p>
      </div>

      <div className="card" style={{ marginBottom: 24 }}>
        <div className="card-title">Ad-hoc Metric Comparison</div>
        <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', alignItems: 'flex-end', marginBottom: 16 }}>
          <div>
            <label style={{ display: 'block', fontSize: 11.5, marginBottom: 6, color: 'var(--text-secondary)' }}>Restored image</label>
            <input type="file" accept="image/*" onChange={(e) => setRestoredFile(e.target.files[0])} />
          </div>
          <div>
            <label style={{ display: 'block', fontSize: 11.5, marginBottom: 6, color: 'var(--text-secondary)' }}>Ground truth image</label>
            <input type="file" accept="image/*" onChange={(e) => setGtFile(e.target.files[0])} />
          </div>
          <button className="btn btn-primary" disabled={!restoredFile || !gtFile || loading} onClick={runEval}>
            {loading ? 'Computing…' : 'Compute Metrics'}
          </button>
        </div>
        {error && <div className="notice notice-amber">{error}</div>}
        {metrics && (
          <div className="grid-metrics" style={{ marginBottom: 0 }}>
            <MetricCard label="PSNR" value={metrics.psnr} unit=" dB" precision={2} />
            <MetricCard label="SSIM" value={metrics.ssim} precision={4} />
            <MetricCard label="LPIPS" value={metrics.lpips} precision={4} emptyText="Unavailable offline" />
            <MetricCard label="MAE" value={metrics.mae} precision={4} />
            <MetricCard label="MSE" value={metrics.mse} precision={5} />
          </div>
        )}
      </div>

      <div className="card">
        <div className="card-title">Recent Results</div>
        {!results?.results?.length ? (
          <div className="metric-empty">No results available — run an experiment.</div>
        ) : (
          <table>
            <thead>
              <tr><th>ID</th><th>Model</th><th>PSNR</th><th>SSIM</th><th>LPIPS</th><th>MAE</th><th>MSE</th><th>Infer (s)</th><th>Checkpoint</th></tr>
            </thead>
            <tbody>
              {results.results.map((r) => (
                <tr key={r.id}>
                  <td>{r.id}</td>
                  <td>{r.model_name}</td>
                  <td>{r.psnr?.toFixed(2) ?? '—'}</td>
                  <td>{r.ssim?.toFixed(4) ?? '—'}</td>
                  <td>{r.lpips?.toFixed(4) ?? '—'}</td>
                  <td>{r.mae?.toFixed(4) ?? '—'}</td>
                  <td>{r.mse?.toFixed(5) ?? '—'}</td>
                  <td>{r.inference_time_seconds?.toFixed(3) ?? '—'}</td>
                  <td>{r.is_trained_checkpoint ? 'trained' : 'untrained'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
