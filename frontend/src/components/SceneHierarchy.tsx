import { useState } from 'react'
import { frameReconMap, type FrameInfo, type ReconstructionData } from '../api/client'
import { useSettings } from '../ui/settings'

// 左 Hierarchy: 抽出済み写真リスト (再構成前から) + 再構成の点群/カメラを表示・選択する.
export const SceneHierarchy = ({
  recon, frames, showPoints, setShowPoints, showCams, setShowCams,
  selectedCameraId, onSelectCamera, selectedFrameIndex, onSelectFrame,
}: {
  recon: ReconstructionData | undefined
  frames: FrameInfo[] | undefined
  showPoints: boolean; setShowPoints: (v: boolean) => void
  showCams: boolean; setShowCams: (v: boolean) => void
  selectedCameraId: number | null; onSelectCamera: (id: number) => void
  selectedFrameIndex: number | null; onSelectFrame: (index: number) => void
}) => {
  const { t } = useSettings()
  const [expandCams, setExpandCams] = useState(false)
  const [expandPhotos, setExpandPhotos] = useState(true)
  if (!recon && !frames?.length) return null
  const reg = frameReconMap(recon)

  return (
    <div>
      {!!frames?.length && (
        <>
          <div className="hier-item" style={{ fontWeight: 600 }} onClick={() => setExpandPhotos(v => !v)}>
            <span style={{ flex: 1 }}>{expandPhotos ? '▾' : '▸'} {t('photos')}</span>
            <span className="mono" style={{ fontSize: 10, color: 'var(--fg-mute)' }}>{frames.length}</span>
          </div>
          {expandPhotos && frames.map(f => {
            // 状態: 再構成前=灰, 登録済=緑, 未登録(失敗)=赤.
            const r = reg.get(f.index)
            const color = !recon ? 'var(--border)' : r ? '#3ad07a' : 'var(--error)'
            return (
              <div key={f.index} className={`hier-item${selectedFrameIndex === f.index ? ' sel' : ''}`}
                style={{ paddingLeft: 28, fontSize: 12 }} onClick={() => onSelectFrame(f.index)}>
                <span className="hier-badge" style={{ background: color }} />
                <span style={{ flex: 1 }}>frame {f.index}</span>
                {f.score && <span className="mono" style={{ fontSize: 10, color: 'var(--fg-mute)' }}>{f.score.sharpness.toFixed(0)}</span>}
              </div>
            )
          })}
        </>
      )}

      {recon && (
        <>
          <div className="hier-item" style={{ fontWeight: 600 }}>▾ Scene</div>
          <div className="hier-item" style={{ paddingLeft: 22 }}>
            <input type="checkbox" checked={showPoints} onChange={e => setShowPoints(e.target.checked)}
              onClick={e => e.stopPropagation()} />
            <span style={{ flex: 1 }}>Point Cloud</span>
            <span className="mono" style={{ fontSize: 10, color: 'var(--fg-mute)' }}>{recon.stats.num_points3D.toLocaleString()}</span>
          </div>
          <div className="hier-item" style={{ paddingLeft: 22 }} onClick={() => setExpandCams(v => !v)}>
            <input type="checkbox" checked={showCams} onChange={e => setShowCams(e.target.checked)}
              onClick={e => e.stopPropagation()} />
            <span style={{ flex: 1 }}>{expandCams ? '▾' : '▸'} Dataset · Cameras</span>
            <span className="mono" style={{ fontSize: 10, color: 'var(--fg-mute)' }}>{recon.images.length}</span>
          </div>
          {expandCams && recon.images.map(img => (
            <div key={img.id} className={`hier-item${selectedCameraId === img.id ? ' sel' : ''}`}
              style={{ paddingLeft: 44, fontSize: 12 }} onClick={() => onSelectCamera(img.id)}>
              <span className="hier-badge" style={{ background: '#3ad07a' }} />
              <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{img.name}</span>
            </div>
          ))}
        </>
      )}
    </div>
  )
}
