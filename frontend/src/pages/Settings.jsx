import { useEffect, useState } from 'react'
import { api } from '../api/client.js'

export default function Settings() {
  const [health, setHealth] = useState(null)
  const [stats, setStats] = useState(null)
  useEffect(() => { api.health().then(setHealth).catch(() => {}); api.datasetStats().then(setStats).catch(() => {}) }, [])

  return (
    <div>
      <div className="page-header">
        <div className="page-eyebrow">Configuration</div>
        <h1 className="page-title">Settings</h1>
        <p className="page-subtitle">All values below are read from <span className="mono">configs/config.yaml</span> — nothing is hardcoded in the frontend.</p>
      </div>

      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-title">Runtime</div>
        <table>
          <tbody>
            <tr><td style={{ color: 'var(--text-secondary)' }}>Backend status</td><td>{health?.status ?? '—'}</td></tr>
            <tr><td style={{ color: 'var(--text-secondary)' }}>Inference device</td><td>{health?.device ?? '—'}</td></tr>
            <tr><td style={{ color: 'var(--text-secondary)' }}>Dataset root</td><td>{stats?.dataset_root ?? '—'}</td></tr>
            <tr><td style={{ color: 'var(--text-secondary)' }}>Dataset is synthetic placeholder</td><td>{String(health?.dataset_is_synthetic_placeholder ?? '—')}</td></tr>
          </tbody>
        </table>
      </div>

      <div className="notice">
        To change dataset path, model hyperparameters, degradation settings, or DEV_MODE, edit{' '}
        <span className="mono">configs/config.yaml</span> or set the <span className="mono">DEV_MODE</span> /{' '}
        <span className="mono">DATASET_ROOT</span> / <span className="mono">DEVICE</span> environment variables and restart the backend.
        See <span className="mono">docs/api.md</span> and the README for details.
      </div>
    </div>
  )
}
