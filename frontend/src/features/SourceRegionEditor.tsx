import { useRef, useState, type SetStateAction } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  api, frameImageUrl, type ProjectSource, type RegionOperation, type RegionView, type SourceRegion,
} from '../api/client'
import { useSettings } from '../ui/settings'
import { AppIcon } from '../components/AppIcon'

type Point = { x: number; y: number }
type RegionTool = 'radius' | 'add' | 'subtract'
type Frame = { index: number; source_id: string }
const MAX_CIRCLE_RADIUS = 0.5

export const SourceRegionEditor = ({ projectId, sources, frames }: {
  projectId: string
  sources: ProjectSource[]
  frames: Frame[]
}) => {
  const { t } = useSettings()
  const [selectedSourceId, setSelectedSourceId] = useState('')
  const [drafts, setDrafts] = useState<Record<string, SourceRegion | undefined>>({})
  const source = sources.find(item => item.id === selectedSourceId)
    ?? sources.find(item => item.role === 'primary') ?? sources[0]
  if (!source) return null

  return <div style={{ maxWidth: 'calc(100vw - 2rem)' }}>
    <div className="ctl">
      <label htmlFor="region-source">{t('regionSource')}</label>
      <select id="region-source" className="input" value={source.id}
        onChange={event => setSelectedSourceId(event.target.value)}>
        {sources.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}
      </select>
    </div>
    <SourceRegionCanvas key={source.id} projectId={projectId} source={source}
      frameIndices={frames.filter(frame => frame.source_id === source.id).map(frame => frame.index)}
      draft={drafts[source.id]}
      setDraft={value => setDrafts(previous => ({ ...previous,
        [source.id]: typeof value === 'function' ? value(previous[source.id]) : value,
      }))} />
  </div>
}

const SourceRegionCanvas = ({ projectId, source, frameIndices, draft, setDraft }: {
  projectId: string
  source: ProjectSource
  frameIndices: number[]
  draft: SourceRegion | undefined
  setDraft: (value: SetStateAction<SourceRegion | undefined>) => void
}) => {
  const { t } = useSettings()
  const qc = useQueryClient()
  const { data, error } = useQuery({
    queryKey: ['source-region', projectId, source.id],
    queryFn: () => api.getSourceRegion(projectId, source.id), retry: false,
  })
  const [sensor, setSensor] = useState(source.projection === 'dual_fisheye' ? 'lens0' : 'main')
  const [previewPosition, setPreviewPosition] = useState(0)
  const [zoom, setZoom] = useState(1)
  const [tool, setTool] = useState<RegionTool>(source.projection === 'dual_fisheye' ? 'radius' : 'subtract')
  const [brushRadius, setBrushRadius] = useState(0.035)
  const [brushPreview, setBrushPreview] = useState<Point | null>(null)
  const [loadedImage, setLoadedImage] = useState<{ url: string; width: number; height: number } | null>(null)
  const [failedImage, setFailedImage] = useState<string | null>(null)
  const [saved, setSaved] = useState(false)
  const svgRef = useRef<SVGSVGElement>(null)
  const interaction = useRef<{
    pointerId: number; tool: RegionTool; strokeId: number; lastPoint: Point | null
  } | null>(null)
  const save = useMutation({
    mutationFn: (region: SourceRegion) => api.putSourceRegion(projectId, source.id, region),
    onSuccess: (region, submitted) => {
      qc.setQueryData(['source-region', projectId, source.id], region)
      setDraft(current => current === submitted ? undefined : current)
      setSaved(true)
      if (region.invalidated?.length) {
        for (const key of ['stages', 'reconstruction', 'masks', 'export-info', 'projects'])
          qc.invalidateQueries({ queryKey: key === 'projects' ? [key] : [key, projectId] })
      }
    },
  })

  if (error) return <div className="error">{String(error)}</div>
  if (!data) return <div className="hint">{t('regionLoading')}</div>
  const region = draft ?? data
  const view = region.views[sensor]
  const frameIndex = frameIndices[Math.min(previewPosition, Math.max(0, frameIndices.length - 1))]
  const imageUrl = frameIndex === undefined ? '' : frameImageUrl(projectId, frameIndex, sensor === 'lens1' ? 1 : 0)
  const imageReady = loadedImage?.url === imageUrl
  const aspect = loadedImage ? loadedImage.height / loadedImage.width : 1
  const hasChanges = JSON.stringify(region.views) !== JSON.stringify(data.views)
  const updateView = (next: SetStateAction<RegionView>) => {
    setDraft(previous => {
      const current = previous ?? data
      return { views: { ...current.views,
        [sensor]: typeof next === 'function' ? next(current.views[sensor]) : next,
      } }
    })
    setSaved(false)
  }
  const toNorm = (event: { clientX: number; clientY: number }): Point | null => {
    const rect = svgRef.current?.getBoundingClientRect()
    if (!rect || !imageReady || save.isPending) return null
    const x = (event.clientX - rect.left) / rect.width
    const y = (event.clientY - rect.top) / rect.height
    return x < 0 || x > 1 || y < 0 || y > 1 ? null : { x, y }
  }
  const paint = (point: Point) => {
    const active = interaction.current
    if (!active) return
    if (active.tool === 'radius') {
      if (view.kind === 'circle') updateView({ ...view,
        r: Math.max(0.01, Math.min(MAX_CIRCLE_RADIUS, Math.hypot(point.x - 0.5, (point.y - 0.5) * aspect))),
      })
      return
    }
    const previous = active.lastPoint
    if (previous && Math.hypot(point.x - previous.x, (point.y - previous.y) * aspect) < brushRadius * 0.25) return
    if (view.operations.length >= 2048) return
    active.lastPoint = point
    const operation: RegionOperation = { mode: active.tool, ...point, r: brushRadius, stroke_id: active.strokeId }
    updateView(current => current.operations.length >= 2048 ? current : {
      ...current, operations: [...current.operations, operation],
    })
  }
  const stopInteraction = () => { interaction.current = null }
  const maskId = `region-${projectId}-${source.id}-${sensor}`
  const hatchId = `${maskId}-hatch`
  const outlineId = `${maskId}-outline`
  const hatchSpacing = 0.035 / zoom
  const tools: RegionTool[] = view.kind === 'circle' ? ['radius', 'add', 'subtract'] : ['add', 'subtract']

  return <div>
    <div style={{ display: 'flex', gap: 8, marginBottom: 8, alignItems: 'center', flexWrap: 'wrap' }}>
      {Object.keys(region.views).length > 1 && <select className="input" aria-label={t('regionSensor')}
        style={{ maxWidth: 130 }} value={sensor} disabled={save.isPending}
        onChange={event => { setSensor(event.target.value); setBrushPreview(null) }}>
        {Object.keys(region.views).map(key => <option key={key} value={key}>{key}</option>)}
      </select>}
      <button type="button" className="btn" disabled={save.isPending} onClick={() => save.mutate(region)}>{t('save')}</button>
      <button type="button" className="btn btn-secondary" disabled={!hasChanges || save.isPending}
        onClick={() => { setDraft(undefined); setSaved(false) }}>{t('discard')}</button>
      {saved && <span className="mono" style={{ color: 'var(--accent)' }}>{t('saved')}</span>}
      {!data.saved && <span className="mono">{t('regionDefault')}</span>}
      {data.needs_review && <span className="mono" style={{ color: '#d69a2a' }}>{t('regionNeedsReview')}</span>}
    </div>
    {view.kind === 'circle' && <div className="ctl">
      <label>{t('regionRadius')}</label>
      <input type="range" aria-label={t('regionRadius')} min={0.01} max={MAX_CIRCLE_RADIUS} step={0.001}
        disabled={save.isPending} value={view.r}
        onChange={event => updateView({ ...view, r: Number(event.target.value) })} style={{ width: '100%' }} />
      <div className="hint">{t('regionRadiusHint')} · r {view.r.toFixed(3)}</div>
    </div>}
    {frameIndex === undefined ? <div className="editor-note" role="note" data-testid="source-region-no-preview">
      <AppIcon name="info" size={16} /><span>{t('regionAdvancedNeedsFrames')}</span>
    </div> : <>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 6 }}>
        <button type="button" className="btn btn-secondary icon-only" title={t('previousFrame')} aria-label={t('previousFrame')}
          disabled={previewPosition === 0} onClick={() => setPreviewPosition(position => Math.max(0, position - 1))}>
          <AppIcon name="chevronLeft" />
        </button>
        <input type="range" aria-label={t('regionPreviewFrame')} min={0} max={Math.max(0, frameIndices.length - 1)}
          step={1} value={previewPosition} onChange={event => setPreviewPosition(Number(event.target.value))}
          style={{ flex: 1, minWidth: 0 }} />
        <button type="button" className="btn btn-secondary icon-only" title={t('nextFrame')} aria-label={t('nextFrame')}
          disabled={previewPosition >= frameIndices.length - 1}
          onClick={() => setPreviewPosition(position => Math.min(frameIndices.length - 1, position + 1))}>
          <AppIcon name="chevronRight" />
        </button>
        <span className="mono">{previewPosition + 1}/{frameIndices.length}</span>
      </div>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 6 }}>
        <AppIcon name="zoom" />
        <input type="range" aria-label={t('zoom')} min={1} max={4} step={0.25} value={zoom}
          onChange={event => setZoom(Number(event.target.value))} style={{ flex: 1, minWidth: 0 }} />
        <span className="mono">{zoom.toFixed(2)}x</span>
      </div>
      <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 6, flexWrap: 'wrap' }}>
        {tools.map(value => <button key={value} type="button" className={`btn ${tool === value ? '' : 'btn-secondary'}`}
          disabled={save.isPending} aria-pressed={tool === value}
          onClick={() => { setTool(value); setBrushPreview(null) }}>{t(`regionTool_${value}`)}</button>)}
        <button type="button" className="btn btn-secondary icon-only" disabled={!view.operations.length || save.isPending}
          title={t('regionUndo')} aria-label={t('regionUndo')}
          onClick={() => updateView({ ...view, operations: removeLastStroke(view.operations) })}>
          <AppIcon name="undo" />
        </button>
        <button type="button" className="btn btn-secondary icon-only" disabled={!view.operations.length || save.isPending}
          title={t('regionClearCustom')} aria-label={t('regionClearCustom')}
          onClick={() => updateView({ ...view, operations: [] })}><AppIcon name="broom" /></button>
        <span className="mono">{t('regionOperationCount')}: {view.operations.length}</span>
      </div>
      {tool !== 'radius' && <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 6 }}>
        <span className="mono">{t('regionBrushSize')}</span>
        <input type="range" aria-label={t('regionBrushSize')} min={0.005} max={0.15} step={0.005} value={brushRadius}
          onChange={event => setBrushRadius(Number(event.target.value))} style={{ flex: 1, minWidth: 0 }} />
        <span className="mono">{brushRadius.toFixed(3)}</span>
      </div>}
      {failedImage === imageUrl && <div className="error" role="alert">{t('regionImageError')}</div>}
      <div style={{ overflow: 'auto', maxHeight: 360, background: '#111', borderRadius: 8 }}>
        <div style={{ position: 'relative', width: `${zoom * 100}%`, aspectRatio: `1 / ${aspect}` }}>
          <img key={imageUrl} src={imageUrl} alt={sensor} style={{ width: '100%', height: 'auto', display: 'block' }}
            onLoad={event => setLoadedImage({ url: imageUrl,
              width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight })}
            onError={() => setFailedImage(imageUrl)} />
          {imageReady && <svg ref={svgRef} viewBox={`0 0 1 ${aspect}`} preserveAspectRatio="none"
            data-testid="source-region-canvas"
            style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', touchAction: 'none',
              pointerEvents: 'all', cursor: tool === 'radius' ? 'default' : 'crosshair' }}
            onPointerDown={event => {
              if (event.button !== 0 || interaction.current) return
              const point = toNorm(event)
              if (!point) return
              event.currentTarget.setPointerCapture(event.pointerId)
              interaction.current = { pointerId: event.pointerId, tool,
                strokeId: Math.max(0, ...view.operations.map(operation => operation.stroke_id)) + 1, lastPoint: null }
              if (tool !== 'radius') setBrushPreview(point)
              paint(point)
            }}
            onPointerMove={event => {
              const point = toNorm(event)
              setBrushPreview(tool === 'radius' ? null : point)
              if (point && interaction.current?.pointerId === event.pointerId) paint(point)
            }}
            onPointerUp={event => {
              if (interaction.current?.pointerId !== event.pointerId) return
              stopInteraction()
              event.currentTarget.releasePointerCapture(event.pointerId)
            }}
            onPointerCancel={stopInteraction} onLostPointerCapture={stopInteraction}
            onPointerLeave={() => setBrushPreview(null)}>
            <defs><mask id={maskId} maskUnits="userSpaceOnUse" x="0" y="0" width="1" height={aspect}>
              <rect x="0" y="0" width="1" height={aspect} fill={view.kind === 'circle' ? 'white' : 'black'} />
              {view.kind === 'circle' && <circle cx="0.5" cy={aspect * 0.5} r={view.r} fill="black" />}
              {view.operations.map((operation, index) => <circle key={index} cx={operation.x} cy={operation.y * aspect}
                r={operation.r} fill={operation.mode === 'add' ? 'black' : 'white'} />)}
            </mask>
              <pattern id={hatchId} patternUnits="userSpaceOnUse" width={hatchSpacing} height={hatchSpacing}>
                <path d={`M0 ${hatchSpacing} L${hatchSpacing} 0`}
                  stroke="#ff91ab" strokeOpacity={0.65} strokeWidth={0.002 / zoom} />
              </pattern>
              <filter id={outlineId} filterUnits="userSpaceOnUse" x="0" y="0" width="1" height={aspect}
                colorInterpolationFilters="sRGB">
                <feMorphology in="SourceAlpha" operator="erode" radius={0.004 / zoom} result="interior" />
                <feComposite in="SourceAlpha" in2="interior" operator="out" result="edge" />
                <feFlood floodColor="#ff91ab" />
                <feComposite in2="edge" operator="in" />
              </filter>
            </defs>
            <g mask={`url(#${maskId})`} pointerEvents="none">
              <rect x="0" y="0" width="1" height={aspect} fill="rgba(0,0,0,0.55)" />
              <rect x="0" y="0" width="1" height={aspect} fill={`url(#${hatchId})`} />
            </g>
            {/* Filter the composed mask so overlapping stamps and restored areas have no stale outlines. */}
            <g filter={`url(#${outlineId})`} pointerEvents="none">
              <rect x="0" y="0" width="1" height={aspect} fill="white" mask={`url(#${maskId})`} />
            </g>
            {view.kind === 'circle' && <circle cx="0.5" cy={aspect * 0.5} r={view.r}
              fill="none" stroke="#4fd1c5" strokeWidth={0.003 / zoom} />}
            {brushPreview && tool !== 'radius' && <circle data-testid="source-region-brush-preview"
              cx={brushPreview.x} cy={brushPreview.y * aspect} r={brushRadius}
              fill={tool === 'add' ? 'rgba(99,230,190,0.2)' : 'rgba(255,107,107,0.2)'}
              stroke={tool === 'add' ? '#63e6be' : '#ff6b6b'} strokeWidth={0.002 / zoom} pointerEvents="none" />}
          </svg>}
        </div>
      </div>
    </>}
    {save.error && <div className="error">{String(save.error)}</div>}
  </div>
}

const removeLastStroke = (operations: RegionOperation[]) => {
  const last = operations.at(-1)
  if (!last) return operations
  let start = operations.length - 1
  while (start > 0 && operations[start - 1].stroke_id === last.stroke_id) start -= 1
  return operations.slice(0, start)
}
