import { useEffect, useState } from 'react'
import { api } from '../api/client.js'

export default function Experiments() {
  const [experiments, setExperiments] = useState([])
  const [model, setModel] = useState('unet')
  const [devMode, setDevMode] = useState(true)
  const [launching, setLaunching] = useState(false)
  const [error, setError] = useState(null)

  const refresh = () => api.experiments(50).then(setExperiments).catch(() => {})
  useEffect(() => { refresh(); const id = setInterval(refresh, 4000); return () => clearInterval(id) }, [])

  const launch = async () => {
    setLaunching(true); setError(null)
    try {
      await api.createExperiment({ model_name: model, dev_mode: devMode })
      refresh()
    } catch (e) { setError(e.message) } finally { setLaunching(false) }
  }

  return (
    <div>
      <div className="page-header">
        <div className="page-eyebrow">Experiments</div>
        <h1 className="page-title">Experiments</h1>
        <p className="page-subtitle">Every training run is logged with its real hyperparameters and results.</p>
      </div>

      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-title">Launch Training Run</div>
        <div style={{ display: 'flex', gap: 14, alignItems: 'flex-end', flexWrap: 'wrap' }}>
          <div>
            <label style={{ display: 'block', fontSize: 11.5, marginBottom: 6, color: 'var(--text-secondary)' }}>Model</label>
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              <option value="unet">UNET</option>
              <option value="swinir">SWINIR</option>
            </select>
          </div>
          <div>
            <label style={{ display: 'block', fontSize: 11.5, marginBottom: 6, color: 'var(--text-secondary)' }}>Mode</label>
            <select value={devMode ? 'dev' : 'full'} onChange={(e) => setDevMode(e.target.value === 'dev')}>
              <option value="dev">DEV_MODE (fast, small subset)</option>
              <option value="full">Full training</option>
            </select>
          </div>
          <button className="btn btn-primary" disabled={launching} onClick={launch}>
            {launching ? 'Launching…' : 'Run Training'}
          </button>
        </div>
        {error && <div className="notice notice-amber" style={{ marginTop: 12 }}>{error}</div>}
        <div className="notice" style={{ marginTop: 14, marginBottom: 0 }}>
          Training runs as a real background job on the backend process (DEV_MODE finishes in seconds on CPU; full training on the
          synthetic placeholder is small but full-scale training on a real dataset should be run from the CLI — see docs/training.md).
        </div>
      </div>

      <div className="card">
        <div className="card-title">Experiment History</div>
        {!experiments.length ? (
          <div className="metric-empty">No results available — run an experiment.</div>
        ) : (
          <table>
            <thead>
              <tr><th>ID</th><th>Model</th><th>Dev</th><th>Epochs</th><th>Best Val PSNR</th><th>Elapsed (s)</th><th>Created</th></tr>
            </thead>
            <tbody>
              {experiments.map((e) => (
                <tr key={e.id}>
                  <td className="mono">{e.experiment_uid}</td>
                  <td>{e.model_name}</td>
                  <td>{e.dev_mode ? 'yes' : 'no'}</td>
                  <td>{e.epochs ?? 'running…'}</td>
                  <td>{e.best_val_psnr?.toFixed(2) ?? 'running…'}</td>
                  <td>{e.elapsed_seconds?.toFixed(1) ?? '—'}</td>
                  <td>{new Date(e.created_at).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
