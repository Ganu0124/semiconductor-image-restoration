import { useEffect, useState } from 'react'
import { api } from '../api/client.js'
import { FileDown } from 'lucide-react'

export default function Reports() {
  const [results, setResults] = useState(null)
  const [busyId, setBusyId] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => { api.results(50).then(setResults).catch(() => {}) }, [])

  const generate = async (id) => {
    setBusyId(id); setError(null)
    try {
      const res = await api.generateReport(id)
      window.open(res.download_url, '_blank')
    } catch (e) { setError(e.message) } finally { setBusyId(null) }
  }

  return (
    <div>
      <div className="page-header">
        <div className="page-eyebrow">Reports</div>
        <h1 className="page-title">Reports</h1>
        <p className="page-subtitle">Generate a PDF inspection report from any stored restoration result.</p>
      </div>

      {error && <div className="notice notice-amber">{error}</div>}

      <div className="card">
        <div className="card-title">Available Results</div>
        {!results?.results?.length ? (
          <div className="metric-empty">No results available — run a restoration first.</div>
        ) : (
          <table>
            <thead>
              <tr><th>ID</th><th>Model</th><th>PSNR</th><th>SSIM</th><th>Created</th><th></th></tr>
            </thead>
            <tbody>
              {results.results.map((r) => (
                <tr key={r.id}>
                  <td>{r.id}</td>
                  <td>{r.model_name}</td>
                  <td>{r.psnr?.toFixed(2) ?? '—'}</td>
                  <td>{r.ssim?.toFixed(4) ?? '—'}</td>
                  <td>{new Date(r.created_at).toLocaleString()}</td>
                  <td>
                    <button className="btn" disabled={busyId === r.id} onClick={() => generate(r.id)}>
                      <FileDown size={14} /> {busyId === r.id ? 'Generating…' : 'Generate PDF'}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
