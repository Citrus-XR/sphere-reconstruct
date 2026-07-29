import { useEffect, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api, frameImageUrl, type FisheyeRegion, type LensCircle } from '../api/client'
import { useSettings } from '../ui/settings'

// 魚眼の円形有効領域をフレーム画像上で調整する. 中心/半径をドラッグ, ズームで拡大確認.
// レンズ端の反射/汚れを外周から除外する. lens0=front, lens1=back を個別調整.

type Handle = 'center' | 'radius' | null
const DEFAULT_CIRCLE: LensCircle = { cx: 0.5, cy: 0.5, r: 0.459 }

export const FisheyeRegionEditor = ({
  projectId,
  sourceId,
  initialFrameIndex,
  frameIndices,
  onSaved,
}: {
  projectId: string
  sourceId: string
  initialFrameIndex: number
  frameIndices: number[]
  onSaved?: () => void
}) => {
  const { t } = useSettings()
  const qc = useQueryClient()
  const { data } = useQuery({
    queryKey: ['fisheye-region', projectId, sourceId],
    queryFn: () => api.getFisheyeRegion(projectId, sourceId),
    enabled: !!projectId, retry: false,
  })
  const [region, setRegion] = useState<FisheyeRegion>({ lens0: DEFAULT_CIRCLE, lens1: DEFAULT_CIRCLE })
  const [lens, setLens] = useState<0 | 1>(0)
  const [zoom, setZoom] = useState(1)
  const [savedMsg, setSavedMsg] = useState(false)
  const [previewPosition, setPreviewPosition] = useState(() => Math.max(
    0,
    frameIndices.indexOf(initialFrameIndex),
  ))

  useEffect(() => { if (data) setRegion(data) }, [data])
  useEffect(() => {
    setPreviewPosition(Math.max(0, frameIndices.indexOf(initialFrameIndex)))
  }, [frameIndices, initialFrameIndex])

  const save = useMutation({
    mutationFn: () => api.putFisheyeRegion(projectId, sourceId, region),
    onSuccess: r => {
      setRegion(r)
      setSavedMsg(true); setTimeout(() => setSavedMsg(false), 1500)
      qc.invalidateQueries({ queryKey: ['fisheye-region', projectId, sourceId] })
      if (r.invalidated?.length) {
        for (const key of ['stages', 'reconstruction', 'masks', 'export-info'])
          qc.invalidateQueries({ queryKey: [key, projectId] })
      }
      onSaved?.()
    },
  })

  const lensKey = lens === 0 ? 'lens0' : 'lens1'
  const circle = region[lensKey]
  const previewFrameIndex = frameIndices[previewPosition] ?? initialFrameIndex
  const setCircle = (c: LensCircle) => setRegion(prev => ({ ...prev, [lensKey]: c }))

  const svgRef = useRef<SVGSVGElement>(null)
  const [drag, setDrag] = useState<Handle>(null)

  const toNorm = (e: { clientX: number; clientY: number }) => {
    const rect = svgRef.current?.getBoundingClientRect()
    if (!rect) return null
    return { x: clamp01((e.clientX - rect.left) / rect.width), y: clamp01((e.clientY - rect.top) / rect.height) }
  }
  const onMove = (e: React.PointerEvent) => {
    if (!drag) return
    const n = toNorm(e); if (!n) return
    if (drag === 'center') setCircle({ ...circle, cx: n.x, cy: n.y })
    else setCircle({ ...circle, r: Math.max(0.05, Math.min(0.75, Math.hypot(n.x - circle.cx, n.y - circle.cy))) })
  }

  return (
    <div>
      <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 4 }}>{t('regionTitle')}</div>
      <div className="hint" style={{ marginBottom: 8 }}>{t('regionHelp')}</div>
      <div style={{ display: 'flex', gap: 8, marginBottom: 8, alignItems: 'center', flexWrap: 'wrap' }}>
        <select className="input" style={{ maxWidth: 130 }} value={lens} onChange={e => setLens(Number(e.target.value) as 0 | 1)}>
          <option value={0}>front (lens0)</option>
          <option value={1}>back (lens1)</option>
        </select>
        <button className="btn" disabled={save.isPending} onClick={() => save.mutate()}>{t('save')}</button>
        {savedMsg && <span className="mono" style={{ color: 'var(--accent)' }}>{t('saved')}</span>}
        {data && !data.saved && <span className="mono" style={{ color: '#d69a2a' }}>{t('regionUnsaved')}</span>}
        {data?.needs_review && <span className="mono" style={{ color: '#d69a2a' }}>{t('regionNeedsReview')}</span>}
      </div>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 6 }}>
        <button type="button" className="btn" aria-label={t('previousFrame')}
          disabled={previewPosition === 0}
          onClick={() => setPreviewPosition(position => Math.max(0, position - 1))}>‹</button>
        <input type="range" aria-label={t('regionPreviewFrame')} min={0}
          max={Math.max(0, frameIndices.length - 1)} step={1} value={previewPosition}
          onChange={event => setPreviewPosition(Number(event.target.value))} style={{ flex: 1 }} />
        <button type="button" className="btn" aria-label={t('nextFrame')}
          disabled={previewPosition >= frameIndices.length - 1}
          onClick={() => setPreviewPosition(position => Math.min(frameIndices.length - 1, position + 1))}>›</button>
        <span className="mono">{t('frameLabel')} {previewFrameIndex} · {previewPosition + 1}/{frameIndices.length}</span>
      </div>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 6 }}>
        <span className="mono">{t('zoom')}</span>
        <input type="range" min={1} max={4} step={0.25} value={zoom} onChange={e => setZoom(Number(e.target.value))} style={{ flex: 1 }} />
        <span className="mono">{zoom.toFixed(2)}x · r {circle.r.toFixed(3)}</span>
      </div>
      <div style={{ overflow: 'auto', maxHeight: 360, background: '#111', borderRadius: 8 }}>
        <div style={{ position: 'relative', width: `${zoom * 100}%`, aspectRatio: '1 / 1' }}>
          <img src={frameImageUrl(projectId, previewFrameIndex, lens)}
            style={{ width: '100%', height: '100%', display: 'block', objectFit: 'cover' }} alt={`lens${lens}`} />
          <svg ref={svgRef} viewBox="0 0 1 1" preserveAspectRatio="none"
            style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', touchAction: 'none' }}
            onPointerMove={onMove} onPointerUp={() => setDrag(null)} onPointerLeave={() => setDrag(null)}>
            <defs>
              <mask id={`hole-${projectId}`}>
                <rect x="0" y="0" width="1" height="1" fill="white" />
                <circle cx={circle.cx} cy={circle.cy} r={circle.r} fill="black" />
              </mask>
            </defs>
            <rect x="0" y="0" width="1" height="1" fill="rgba(0,0,0,0.55)" mask={`url(#hole-${projectId})`} />
            <circle cx={circle.cx} cy={circle.cy} r={circle.r} fill="none" stroke="#4fd1c5" strokeWidth={0.003 / zoom} />
            <circle cx={circle.cx + circle.r} cy={circle.cy} r={0.014 / zoom} fill="#4fd1c5"
              style={{ cursor: 'ew-resize' }}
              onPointerDown={e => { (e.target as Element).setPointerCapture(e.pointerId); setDrag('radius') }} />
            <circle cx={circle.cx} cy={circle.cy} r={0.011 / zoom} fill="#f6ad55"
              style={{ cursor: 'move' }}
              onPointerDown={e => { (e.target as Element).setPointerCapture(e.pointerId); setDrag('center') }} />
          </svg>
        </div>
      </div>
      {save.error && <div className="error">{String(save.error)}</div>}
    </div>
  )
}

const clamp01 = (v: number) => Math.max(0, Math.min(1, v))
