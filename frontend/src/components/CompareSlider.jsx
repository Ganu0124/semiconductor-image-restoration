import { useRef, useState, useCallback } from 'react'

export default function CompareSlider({ beforeSrc, afterSrc, beforeLabel = 'Degraded', afterLabel = 'Restored' }) {
  const [pct, setPct] = useState(50)
  const wrapRef = useRef(null)
  const dragging = useRef(false)

  const updateFromClientX = useCallback((clientX) => {
    const el = wrapRef.current
    if (!el) return
    const rect = el.getBoundingClientRect()
    const x = Math.min(Math.max(clientX - rect.left, 0), rect.width)
    setPct((x / rect.width) * 100)
  }, [])

  const onDown = (e) => { dragging.current = true; updateFromClientX(e.clientX ?? e.touches?.[0]?.clientX) }
  const onMove = (e) => { if (dragging.current) updateFromClientX(e.clientX ?? e.touches?.[0]?.clientX) }
  const onUp = () => { dragging.current = false }

  return (
    <div
      className="compare-wrap"
      ref={wrapRef}
      onMouseDown={onDown}
      onMouseMove={onMove}
      onMouseUp={onUp}
      onMouseLeave={onUp}
      onTouchStart={onDown}
      onTouchMove={onMove}
      onTouchEnd={onUp}
    >
      <img src={beforeSrc} alt={beforeLabel} draggable={false} />
      <div className="compare-after-clip" style={{ width: `${pct}%` }}>
        <img src={afterSrc} alt={afterLabel} style={{ width: wrapRef.current?.clientWidth || '100%' }} draggable={false} />
      </div>
      <div className="compare-handle" style={{ left: `${pct}%` }} />
      <div className="compare-tag" style={{ left: 8 }}>{beforeLabel}</div>
      <div className="compare-tag" style={{ right: 8 }}>{afterLabel}</div>
    </div>
  )
}
