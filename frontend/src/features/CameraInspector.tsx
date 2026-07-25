import { useState } from 'react'
import { frameImageUrl, fisheyeMaskUrl, parseImageName, type ReconstructionImage } from '../api/client'
import { useSettings } from '../ui/settings'

// カメラ選択時に Inspector に表示: 対応フレーム画像 + 適用マスクの重ね表示トグル.
export const CameraInspector = ({ projectId, image }: { projectId: string; image: ReconstructionImage }) => {
  const { t } = useSettings()
  const [showMask, setShowMask] = useState(false)
  const parsed = parseImageName(image.name)

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
        <strong style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis' }}>{image.name}</strong>
        {parsed && (
          <button className={`btn ${showMask ? '' : 'btn-secondary'}`} onClick={() => setShowMask(v => !v)}>
            {t('maskOverlay')}
          </button>
        )}
      </div>
      <div className="mono" style={{ marginBottom: 8, fontSize: 11 }}>
        pos [{image.position.map(v => v.toFixed(2)).join(', ')}] · pts {image.num_points}
      </div>
      {parsed ? (
        <div style={{ position: 'relative', background: '#111', borderRadius: 6, overflow: 'hidden' }}>
          <img src={frameImageUrl(projectId, parsed.index, parsed.lens)} style={{ width: '100%', display: 'block' }} alt={image.name} />
          {showMask && (
            <img src={fisheyeMaskUrl(projectId, parsed.index, parsed.lens)}
              style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', opacity: 0.5, mixBlendMode: 'screen' }}
              alt="mask" onError={e => { (e.target as HTMLImageElement).style.display = 'none' }} />
          )}
        </div>
      ) : (
        <div className="hint">{t('noCamImage')}</div>
      )}
    </div>
  )
}
