import { useMemo, useState } from 'react'
import { frameReconMap, type FrameInfo, type FramesManifest, type ReconstructionData } from '../api/client'
import { useSettings } from '../ui/settings'

const PhotoSourceGroup = ({
  source, frames, expanded, reconstructionAvailable, registration,
  selectedFrameIndex, frameLabel, onToggle, onSelectFrame,
}: {
  source: FramesManifest['sources'][number]
  frames: FrameInfo[]
  expanded: boolean
  reconstructionAvailable: boolean
  registration: ReturnType<typeof frameReconMap>
  selectedFrameIndex: number | null
  frameLabel: string
  onToggle: () => void
  onSelectFrame: (index: number) => void
}) => (
  <div className="hier-source-group" data-source-id={source.id}>
    <button type="button" className="hier-item hier-row-button"
      style={{ paddingLeft: 22, fontWeight: 600 }} onClick={onToggle} aria-expanded={expanded}>
      <span style={{ flex: 1 }}>{expanded ? '▾' : '▸'} {source.label}</span>
      <span className="mono" style={{ fontSize: 10 }}>{source.count}</span>
    </button>
    {expanded && frames.map(frame => {
      const registered = registration.get(frame.index)
      const color = !reconstructionAvailable ? 'var(--border)' : registered ? '#3ad07a' : 'var(--error)'
      return (
        <button type="button" key={`${source.id}:${frame.index}`}
          className={`hier-item hier-row-button${selectedFrameIndex === frame.index ? ' sel' : ''}`}
          style={{ paddingLeft: 28, fontSize: 12 }} onClick={() => onSelectFrame(frame.index)}
          aria-current={selectedFrameIndex === frame.index ? 'true' : undefined}>
          <span className="hier-badge" style={{ background: color }} />
          <span style={{ flex: 1 }}>{frameLabel} {frame.source_index}</span>
          {frame.score && <span className="mono" style={{ fontSize: 10, color: 'var(--fg-mute)' }}>
            {frame.score.sharpness.toFixed(0)}
          </span>}
        </button>
      )
    })}
  </div>
)

// 左 Hierarchy: 抽出済み写真リスト (再構成前から) + 再構成の点群/カメラを表示・選択する.
export const SceneHierarchy = ({
  recon, frames, sources, showPoints, setShowPoints, showCams, setShowCams,
  selectedCameraId, onSelectCamera, selectedFrameIndex, onSelectFrame,
}: {
  recon: ReconstructionData | undefined
  frames: FrameInfo[] | undefined
  sources: FramesManifest['sources'] | undefined
  showPoints: boolean; setShowPoints: (v: boolean) => void
  showCams: boolean; setShowCams: (v: boolean) => void
  selectedCameraId: number | null; onSelectCamera: (id: number) => void
  selectedFrameIndex: number | null; onSelectFrame: (index: number) => void
}) => {
  const { t } = useSettings()
  const [expandCams, setExpandCams] = useState(false)
  const [expandPhotos, setExpandPhotos] = useState(false)
  const [collapsedPhotoSources, setCollapsedPhotoSources] = useState<Set<string>>(() => new Set())
  const framesBySource = useMemo(() => {
    const grouped = new Map<string, FrameInfo[]>()
    for (const frame of frames ?? []) {
      const sourceFrames = grouped.get(frame.source_id)
      if (sourceFrames) sourceFrames.push(frame)
      else grouped.set(frame.source_id, [frame])
    }
    return grouped
  }, [frames])
  const togglePhotoSource = (sourceId: string) => {
    setCollapsedPhotoSources(current => {
      const next = new Set(current)
      if (next.has(sourceId)) next.delete(sourceId)
      else next.add(sourceId)
      return next
    })
  }
  if (!recon && !frames?.length) return null
  const reg = frameReconMap(recon)

  return (
    <div>
      {!!frames?.length && (
        <>
          <button type="button" className="hier-item hier-row-button" style={{ fontWeight: 600 }}
            onClick={() => setExpandPhotos(v => !v)} aria-expanded={expandPhotos}>
            <span style={{ flex: 1 }}>{expandPhotos ? '▾' : '▸'} {t('photos')}</span>
            <span className="mono" style={{ fontSize: 10, color: 'var(--fg-mute)' }}>{frames.length}</span>
          </button>
          {expandPhotos && (sources ?? []).map(source => (
            <PhotoSourceGroup key={source.id} source={source}
              frames={framesBySource.get(source.id) ?? []}
              expanded={!collapsedPhotoSources.has(source.id)} reconstructionAvailable={!!recon}
              registration={reg} selectedFrameIndex={selectedFrameIndex} frameLabel={t('frameLabel')}
              onToggle={() => togglePhotoSource(source.id)} onSelectFrame={onSelectFrame} />
          ))}
        </>
      )}

      {recon && (
        <>
          <div className="hier-item" style={{ fontWeight: 600 }}>▾ {t('sceneRoot')}</div>
          <label className="hier-item" style={{ paddingLeft: 22 }}>
            <input type="checkbox" checked={showPoints} onChange={e => setShowPoints(e.target.checked)}
              aria-label={t('togglePointCloud')} />
            <span style={{ flex: 1 }}>{t('pointCloud')}</span>
            <span className="mono" style={{ fontSize: 10, color: 'var(--fg-mute)' }}>{recon.stats.num_points3D.toLocaleString()}</span>
          </label>
          <div className="hier-item" style={{ paddingLeft: 22 }}>
            <input type="checkbox" checked={showCams} onChange={e => setShowCams(e.target.checked)}
              aria-label={t('toggleCameras')} />
            <button type="button" className="hier-inline-button" onClick={() => setExpandCams(v => !v)}
              aria-expanded={expandCams}>
              {expandCams ? '▾' : '▸'} {t('datasetCameras')}
            </button>
            <span className="mono" style={{ fontSize: 10, color: 'var(--fg-mute)' }}>{recon.images.length}</span>
          </div>
          {expandCams && recon.images.map(img => (
            <button type="button" key={img.id}
              className={`hier-item hier-row-button${selectedCameraId === img.id ? ' sel' : ''}`}
              style={{ paddingLeft: 44, fontSize: 12 }} onClick={() => onSelectCamera(img.id)}
              aria-current={selectedCameraId === img.id ? 'true' : undefined}>
              <span className="hier-badge" style={{ background: '#3ad07a' }} />
              <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{img.name}</span>
            </button>
          ))}
        </>
      )}
    </div>
  )
}
