import { useEffect, useRef, useState } from 'react'
import { api, resolveApiUrl } from '../api/client.js'
import CompareSlider from '../components/CompareSlider.jsx'
import { UploadCloud, Download, Loader2 } from 'lucide-react'

export default function ImageRestoration() {
  const [models, setModels] = useState([])
  const [selectedModel, setSelectedModel] = useState('unet')
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [galleryPick, setGalleryPick] = useState(null) // {split, filename}
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [trainImages, setTrainImages] = useState([])
  const fileInputRef = useRef(null)

  useEffect(() => {
    api.models().then(setModels).catch(() => { })
    api.datasetGallery('test', 8).then((r) => setTrainImages(r.images || [])).catch(() => { })
  }, [])

  const onFileSelected = (f) => {
    setFile(f)
    setGalleryPick(null)
    setPreviewUrl(URL.createObjectURL(f))
    setResult(null)
  }

  const pickFromDataset = async (img) => {
    try {
      const imageUrl = resolveApiUrl(img.degraded_path)

      const res = await fetch(imageUrl)

      if (!res.ok) {
        throw new Error(`Could not load dataset image: ${res.status}`)
      }

      const blob = await res.blob()
      const f = new File([blob], img.filename, {
        type: blob.type || 'image/png'
      })

      setFile(f)
      setGalleryPick(img)
      setPreviewUrl(imageUrl)
      setResult(null)
    } catch (e) {
      setError(e.message)
    }
  }

  const runRestoration = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    try {
      const fd = new FormData()
      fd.append('file', file)
      fd.append('model', selectedModel)
      if (galleryPick) {
        fd.append('ground_truth_split', 'test')
        fd.append('ground_truth_filename', galleryPick.filename)
      }
      const res = await api.restore(fd)
      setResult(res)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <div className="page-header">
        <div className="page-eyebrow">Restoration</div>
        <h1 className="page-title">Image Restoration</h1>
        <p className="page-subtitle">Upload a degraded inspection image, or pick one from the test split, then run real AI restoration.</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '360px 1fr', gap: 20, alignItems: 'start' }}>
        <div className="card">
          <div className="card-title">1 · Input</div>

          <div
            className="upload-zone"
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => { e.preventDefault(); if (e.dataTransfer.files[0]) onFileSelected(e.dataTransfer.files[0]) }}
          >
            <UploadCloud size={22} style={{ marginBottom: 8, color: 'var(--accent-cyan)' }} />
            <div style={{ fontSize: 13 }}>{file ? file.name : 'Click or drop an image'}</div>
          </div>
          <input ref={fileInputRef} type="file" accept="image/*" style={{ display: 'none' }}
            onChange={(e) => e.target.files[0] && onFileSelected(e.target.files[0])} />

          <div style={{ margin: '14px 0 6px 0', fontSize: 11.5, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            Or pick from test split (has ground truth)
          </div>
          <div className="gallery-grid" style={{ gridTemplateColumns: 'repeat(4, 1fr)' }}>
            {trainImages.map((img) => (
              <div key={img.filename} className="gallery-item" onClick={() => pickFromDataset(img)}
                style={{ borderColor: galleryPick?.filename === img.filename ? 'var(--accent-cyan)' : undefined }}>
                <img src={resolveApiUrl(img.degraded_path)} alt={img.filename} />
              </div>
            ))}
          </div>

          <div className="form-row" style={{ marginTop: 18 }}>
            <label>2 · Model</label>
            <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)} style={{ width: '100%' }}>
              {models.map((m) => (
                <option key={m.name} value={m.name}>
                  {m.name.toUpperCase()} {m.is_trained ? '' : '(untrained checkpoint)'}
                </option>
              ))}
              {models.length === 0 && <option>unet</option>}
            </select>
          </div>

          <button className="btn btn-primary" style={{ width: '100%', justifyContent: 'center' }}
            disabled={!file || loading} onClick={runRestoration}>
            {loading ? <Loader2 size={15} className="spin" /> : null}
            {loading ? 'Restoring…' : '3 · Restore Image'}
          </button>

          {error && <div className="notice notice-amber" style={{ marginTop: 12 }}>{error}</div>}
        </div>

        <div className="card">
          <div className="card-title">Restoration Result</div>
          {!result && !previewUrl && <div className="metric-empty">Upload or select an image to begin.</div>}

          {result && (
            <>
              {result.note && <div className="notice notice-amber">{result.note}</div>}
              <div style={{ display: 'flex', gap: 10, marginBottom: 16 }}>
                <span className="badge badge-info">{result.model.toUpperCase()}</span>
                <span className={`badge ${result.is_trained_checkpoint ? 'badge-ok' : 'badge-warn'}`}>
                  {result.is_trained_checkpoint ? 'trained checkpoint' : 'untrained weights'}
                </span>
                <span className="badge">device: {result.device}</span>
              </div>

              <CompareSlider
                beforeSrc={resolveApiUrl(result.input_image_url)}
                afterSrc={resolveApiUrl(result.restored_image_url)}
              />

              {result.ground_truth_image_url && (
                <div className="image-triplet" style={{ marginTop: 16 }}>
                  <figure><img src={resolveApiUrl(result.input_image_url)} /><figcaption>Degraded</figcaption></figure>
                  <figure><img src={resolveApiUrl(result.restored_image_url)} /><figcaption>Restored</figcaption></figure>
                  <figure><img src={resolveApiUrl(result.ground_truth_image_url)} /><figcaption>Ground Truth</figcaption></figure>
                </div>
              )}

              <div className="grid-metrics" style={{ marginTop: 18, marginBottom: 0 }}>
                <MetricInline label="PSNR" value={result.psnr} unit=" dB" precision={2} />
                <MetricInline label="SSIM" value={result.ssim} precision={4} />
                <MetricInline label="LPIPS" value={result.lpips} precision={4} />
                <MetricInline label="Inference Time" value={result.inference_time_seconds} unit=" s" precision={3} />
              </div>

              <a className="btn" style={{ marginTop: 16 }} href={resolveApiUrl(result.restored_image_url)} download>
                <Download size={14} /> Download Restored Image
              </a>
            </>
          )}

          {!result && previewUrl && (
            <img src={previewUrl} alt="preview" style={{ maxWidth: 320, borderRadius: 8, border: '1px solid var(--border-subtle)' }} />
          )}
        </div>
      </div>
    </div>
  )
}

function MetricInline({ label, value, unit = '', precision = 3 }) {
  const has = value !== null && value !== undefined
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      {has ? <div className="metric-value">{value.toFixed(precision)}{unit && <span className="unit">{unit}</span>}</div>
        : <div className="metric-empty">No results available — run an experiment.</div>}
    </div>
  )
}
