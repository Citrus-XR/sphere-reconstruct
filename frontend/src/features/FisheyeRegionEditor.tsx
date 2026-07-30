import { useEffect, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  api,
  frameImageUrl,
  type FisheyeRegion,
  type LensCircle,
  type RegionOperation,
} from '../api/client'
import { useSettings } from '../ui/settings'

// 魚眼の中心固定円と加算/減算 brush を sensor ごとに編集する。

type Handle = 'radius' | null
type RegionTool = 'radius' | 'add' | 'subtract'
const defaultCircle = (): LensCircle => ({ cx: 0.5, cy: 0.5, r: 0.459, operations: [] })
const defaultRegion = (): FisheyeRegion => ({ lens0: defaultCircle(), lens1: defaultCircle() })

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
  const [region, setRegion] = useState<FisheyeRegion>(() => defaultRegion())
  const [lens, setLens] = useState<0 | 1>(0)
  const [zoom, setZoom] = useState(1)
  const [tool, setTool] = useState<RegionTool>('radius')
  const [brushRadius, setBrushRadius] = useState(0.035)
  const [painting, setPainting] = useState(false)
  const [savedMsg, setSavedMsg] = useState(false)
  const [previewPosition, setPreviewPosition] = useState(() => Math.max(
    0,
    frameIndices.indexOf(initialFrameIndex),
  ))

  useEffect(() => { if (data) setRegion(normalizeRegion(data)) }, [data])
  useEffect(() => {
    setPreviewPosition(Math.max(0, frameIndices.indexOf(initialFrameIndex)))
  }, [frameIndices, initialFrameIndex])

  const save = useMutation({
    mutationFn: () => api.putFisheyeRegion(projectId, sourceId, region),
    onSuccess: r => {
      setRegion(normalizeRegion(r))
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
  const setCircle = (c: LensCircle) => setRegion(prev => ({
    ...prev,
    [lensKey]: { ...c, cx: 0.5, cy: 0.5 },
  }))

  const svgRef = useRef<SVGSVGElement>(null)
  const lastStampRef = useRef<{ x: number; y: number } | null>(null)
  const [drag, setDrag] = useState<Handle>(null)

  const toNorm = (e: { clientX: number; clientY: number }) => {
    const rect = svgRef.current?.getBoundingClientRect()
    if (!rect) return null
    return { x: clamp01((e.clientX - rect.left) / rect.width), y: clamp01((e.clientY - rect.top) / rect.height) }
  }
  const onMove = (e: React.PointerEvent) => {
    const n = toNorm(e); if (!n) return
    if (drag === 'radius') {
      setCircle({ ...circle, r: Math.max(0.05, Math.min(0.75, Math.hypot(n.x - 0.5, n.y - 0.5))) })
    } else if (painting && (tool === 'add' || tool === 'subtract')) {
      addStamp(n.x, n.y)
    }
  }
  const addStamp = (x: number, y: number) => {
    const previous = lastStampRef.current
    if (previous && Math.hypot(x - previous.x, y - previous.y) < brushRadius * 0.25) return
    lastStampRef.current = { x, y }
    const operation: RegionOperation = { mode: tool as 'add' | 'subtract', x, y, r: brushRadius }
    setRegion(previousRegion => {
      const previousCircle = previousRegion[lensKey]
      if ((previousCircle.operations?.length ?? 0) >= 2048) return previousRegion
      return {
        ...previousRegion,
        [lensKey]: {
          ...previousCircle,
          cx: 0.5,
          cy: 0.5,
          operations: [...(previousCircle.operations ?? []), operation],
        },
      }
    })
  }
  const stopInteraction = () => {
    setDrag(null)
    setPainting(false)
    lastStampRef.current = null
  }
  const maskId = `hole-${projectId}-${sourceId}-${lens}`

  return (
    <div>
      <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 4 }}>{t('regionTitle')}</div>
      <div className="hint" style={{ marginBottom: 8 }}>{t('regionHelp')}</div>
      <div style={{ display: 'flex', gap: 8, marginBottom: 8, alignItems: 'center', flexWrap: 'wrap' }}>
        <select className="input" style={{ maxWidth: 130 }} value={lens} onChange={e => setLens(Number(e.target.value) as 0 | 1)}>
          <option value={0}>lens0</option>
          <option value={1}>lens1</option>
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
      <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 6, flexWrap: 'wrap' }}>
        {(['radius', 'add', 'subtract'] as RegionTool[]).map(value => (
          <button key={value} type="button" className={`btn ${tool === value ? '' : 'btn-secondary'}`}
            aria-pressed={tool === value} onClick={() => setTool(value)}>
            {t(`regionTool_${value}`)}
          </button>
        ))}
        <button type="button" className="btn btn-secondary" disabled={!circle.operations?.length}
          onClick={() => setCircle({ ...circle, operations: circle.operations.slice(0, -1) })}>
          {t('regionUndo')}
        </button>
        <button type="button" className="btn btn-secondary" disabled={!circle.operations?.length}
          onClick={() => setCircle({ ...circle, operations: [] })}>
          {t('regionClearCustom')}
        </button>
        <span className="mono">{t('regionOperationCount')}: {circle.operations?.length ?? 0}</span>
      </div>
      {(tool === 'add' || tool === 'subtract') && (
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 6 }}>
          <span className="mono">{t('regionBrushSize')}</span>
          <input type="range" min={0.005} max={0.15} step={0.005} value={brushRadius}
            onChange={event => setBrushRadius(Number(event.target.value))} style={{ flex: 1 }} />
          <span className="mono">{brushRadius.toFixed(3)}</span>
        </div>
      )}
      <div style={{ overflow: 'auto', maxHeight: 360, background: '#111', borderRadius: 8 }}>
        <div style={{ position: 'relative', width: `${zoom * 100}%`, aspectRatio: '1 / 1' }}>
          <img src={frameImageUrl(projectId, previewFrameIndex, lens)}
            style={{ width: '100%', height: '100%', display: 'block', objectFit: 'cover' }} alt={`lens${lens}`} />
          <svg ref={svgRef} viewBox="0 0 1 1" preserveAspectRatio="none"
            style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', touchAction: 'none' }}
            onPointerDown={event => {
              if (tool === 'radius') return
              event.currentTarget.setPointerCapture(event.pointerId)
              setPainting(true)
              const point = toNorm(event)
              if (point) addStamp(point.x, point.y)
            }}
            onPointerMove={onMove} onPointerUp={stopInteraction} onPointerCancel={stopInteraction}
            onPointerLeave={() => { if (!painting) stopInteraction() }}>
            <defs>
              <mask id={maskId}>
                <rect x="0" y="0" width="1" height="1" fill="white" />
                <circle cx="0.5" cy="0.5" r={circle.r} fill="black" />
                {(circle.operations ?? []).map((operation, index) => (
                  <circle key={index} cx={operation.x} cy={operation.y} r={operation.r}
                    fill={operation.mode === 'add' ? 'black' : 'white'} />
                ))}
              </mask>
            </defs>
            <rect x="0" y="0" width="1" height="1" fill="rgba(0,0,0,0.55)" mask={`url(#${maskId})`} />
            <circle cx="0.5" cy="0.5" r={circle.r} fill="none" stroke="#4fd1c5" strokeWidth={0.003 / zoom} />
            {(circle.operations ?? []).map((operation, index) => (
              <circle key={index} cx={operation.x} cy={operation.y} r={operation.r} fill="none"
                stroke={operation.mode === 'add' ? '#63e6be' : '#ff6b6b'} strokeWidth={0.002 / zoom} />
            ))}
            <circle cx={0.5 + circle.r} cy="0.5" r={0.014 / zoom} fill="#4fd1c5"
              style={{ cursor: 'ew-resize' }}
              onPointerDown={e => {
                if (tool !== 'radius') return
                e.stopPropagation()
                const target = e.target as Element
                target.setPointerCapture(e.pointerId)
                setDrag('radius')
              }} />
            <line x1={0.485} y1="0.5" x2={0.515} y2="0.5" stroke="#f6ad55" strokeWidth={0.002 / zoom} />
            <line x1="0.5" y1={0.485} x2="0.5" y2={0.515} stroke="#f6ad55" strokeWidth={0.002 / zoom} />
          </svg>
        </div>
      </div>
      {save.error && <div className="error">{String(save.error)}</div>}
    </div>
  )
}

const clamp01 = (v: number) => Math.max(0, Math.min(1, v))

const normalizeCircle = (value: Partial<LensCircle> | undefined): LensCircle => ({
  cx: 0.5,
  cy: 0.5,
  r: typeof value?.r === 'number' ? value.r : 0.459,
  operations: Array.isArray(value?.operations) ? value.operations : [],
})

const normalizeRegion = (value: FisheyeRegion): FisheyeRegion => ({
  ...value,
  lens0: normalizeCircle(value.lens0),
  lens1: normalizeCircle(value.lens1),
})
