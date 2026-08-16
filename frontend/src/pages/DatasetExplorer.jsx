import { useEffect, useState } from 'react'
import { api } from '../api/client.js'

export default function DatasetExplorer() {
  const [stats, setStats] = useState(null)
  const [split, setSplit] = useState('train')
  const [gallery, setGallery] = useState(null)
  const [selected, setSelected] = useState(null)

  useEffect(() => { api.datasetStats().then(setStats).catch(() => {}) }, [])
  useEffect(() => { api.datasetGallery(split, 24).then(setGallery).catch(() => {}) }, [split])

  const s = stats?.splits ?? {}

  return (
    <div>
      <div className="page-header">
        <div className="page-eyebrow">Dataset</div>
        <h1 className="page-title">Dataset Explorer</h1>
        <p className="page-subtitle">Real statistics from the dataset analyzer — no invented numbers.</p>
      </div>

      {stats?.is_synthetic_placeholder && (
        <div className="notice notice-amber">Currently pointed at the synthetic placeholder dataset. Swap in your real dataset via <span className="mono">DATASET_ROOT</span>.</div>
      )}

      <div className="grid-metrics">
        <MiniStat label="Total Images" value={stats?.total_images} />
        <MiniStat label="Train Images" value={s.train?.num_images} />
        <MiniStat label="Val Images" value={s.val?.num_images} />
        <MiniStat label="Test Images" value={s.test?.num_images} />
        <MiniStat label="Paired Dataset" value={stats ? (stats.is_paired_dataset ? 'yes' : 'no') : undefined} />
        <MiniStat label="Format" value={Object.keys(s.train?.formats ?? {})[0]} />
      </div>

      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-title">Split Details — {split}</div>
        {s[split] ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px,1fr))', gap: 12, fontSize: 13 }}>
            <Detail label="Layout" value={s[split].layout} />
            <Detail label="Channels" value={Object.keys(s[split].channels || {}).join(', ') || '—'} />
            <Detail label="Width range" value={s[split].min_width != null ? `${s[split].min_width}–${s[split].max_width}px` : '—'} />
            <Detail label="Height range" value={s[split].min_height != null ? `${s[split].min_height}–${s[split].max_height}px` : '—'} />
            <Detail label="Pairs" value={s[split].num_pairs} />
            <Detail label="Corrupt files" value={s[split].corrupt_files?.length ?? 0} />
          </div>
        ) : <div className="metric-empty">No data for this split.</div>}
      </div>

      <div className="card">
        <div className="card-title">Gallery</div>
        <div style={{ display: 'flex', gap: 8, marginBottom: 14 }}>
          {['train', 'val', 'test'].map((sp) => (
            <button key={sp} className="btn" style={{ background: split === sp ? 'var(--accent-cyan)' : undefined, color: split === sp ? '#04121a' : undefined }}
              onClick={() => setSplit(sp)}>{sp}</button>
          ))}
        </div>
        <div className="gallery-grid">
          {(gallery?.images ?? []).map((img) => (
            <div key={img.filename} className="gallery-item" onClick={() => setSelected(img)}>
              <img src={img.clean_path} alt={img.filename} />
              <div className="fname">{img.filename}</div>
            </div>
          ))}
        </div>
        {!gallery?.images?.length && <div className="metric-empty">No images in this split.</div>}
      </div>

      {selected && (
        <div className="card" style={{ marginTop: 20 }}>
          <div className="card-title">Selected: {selected.filename}</div>
          <div className="image-triplet" style={{ maxWidth: 460 }}>
            <figure><img src={selected.clean_path} /><figcaption>Clean</figcaption></figure>
            {selected.degraded_path && <figure><img src={selected.degraded_path} /><figcaption>Degraded</figcaption></figure>}
          </div>
        </div>
      )}
    </div>
  )
}

function MiniStat({ label, value }) {
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className="metric-value" style={{ fontSize: 20 }}>{value ?? '—'}</div>
    </div>
  )
}
function Detail({ label, value }) {
  return <div><div className="metric-label">{label}</div><div className="mono">{value ?? '—'}</div></div>
}
