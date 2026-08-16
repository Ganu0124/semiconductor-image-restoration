import { useEffect, useState } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'
import { api } from '../api/client.js'

export default function Analytics() {
  const [modelFilter, setModelFilter] = useState('')
  const [trends, setTrends] = useState(null)
  const [history, setHistory] = useState(null)
  const [historyModel, setHistoryModel] = useState('unet')

  useEffect(() => { api.analyticsTrends(modelFilter || undefined).then(setTrends).catch(() => {}) }, [modelFilter])
  useEffect(() => { api.trainingHistory(historyModel).then(setHistory).catch(() => {}) }, [historyModel])

  const points = (trends?.points ?? []).map((p, i) => ({ idx: i + 1, ...p }))

  return (
    <div>
      <div className="page-header">
        <div className="page-eyebrow">Analytics</div>
        <h1 className="page-title">Analytics</h1>
        <p className="page-subtitle">Trends over actual restoration results and training logs.</p>
      </div>

      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-title">
          Result Trends
          <select value={modelFilter} onChange={(e) => setModelFilter(e.target.value)} style={{ marginLeft: 'auto' }}>
            <option value="">All models</option>
            <option value="unet">UNET</option>
            <option value="swinir">SWINIR</option>
          </select>
        </div>
        {!points.length ? <div className="metric-empty">No results available — run an experiment.</div> : (
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={points}>
              <CartesianGrid stroke="var(--border-subtle)" strokeDasharray="3 3" />
              <XAxis dataKey="idx" stroke="var(--text-tertiary)" fontSize={11} />
              <YAxis stroke="var(--text-tertiary)" fontSize={11} />
              <Tooltip contentStyle={{ background: 'var(--bg-panel-raised)', border: '1px solid var(--border-strong)' }} />
              <Legend />
              <Line type="monotone" dataKey="psnr" stroke="#4dd8e6" dot={false} name="PSNR (dB)" />
              <Line type="monotone" dataKey="ssim" stroke="#5b8def" dot={false} name="SSIM" yAxisId={0} />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      <div className="card">
        <div className="card-title">
          Training Loss History
          <select value={historyModel} onChange={(e) => setHistoryModel(e.target.value)} style={{ marginLeft: 'auto' }}>
            <option value="unet">UNET</option>
            <option value="swinir">SWINIR</option>
          </select>
        </div>
        {!history?.history?.length ? (
          <div className="metric-empty">{history?.note || 'No training runs found for this model yet.'}</div>
        ) : (
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={history.history}>
              <CartesianGrid stroke="var(--border-subtle)" strokeDasharray="3 3" />
              <XAxis dataKey="epoch" stroke="var(--text-tertiary)" fontSize={11} />
              <YAxis stroke="var(--text-tertiary)" fontSize={11} />
              <Tooltip contentStyle={{ background: 'var(--bg-panel-raised)', border: '1px solid var(--border-strong)' }} />
              <Legend />
              <Line type="monotone" dataKey="train_loss" stroke="#e6a94d" dot={false} name="Train Loss" />
              <Line type="monotone" dataKey="val_loss" stroke="#e65a5a" dot={false} name="Val Loss" />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  )
}
