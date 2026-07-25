import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api, frameImageUrl, fisheyeMaskUrl, frameReconMap, type FrameInfo, type ReconstructionData } from '../api/client'
import { useSettings } from '../ui/settings'

// Hierarchy で写真を選んだときの Inspector: フレーム画像 + 抽出スコア + 処理結果 (再構成の
// 登録可否 / 3D 点数) + SAM3 マスク (あれば 原画/マスク/重ね を切替, 動体被覆率も表示).
export const FrameInspector = ({
  projectId, frameIndex, frames, recon, sourceKind,
}: {
  projectId: string
  frameIndex: number
  frames: FrameInfo[] | undefined
  recon: ReconstructionData | undefined
  sourceKind: string | null
}) => {
  const { t } = useSettings()
  const [lens, setLens] = useState(0)
  const [view, setView] = useState<'orig' | 'mask' | 'overlay'>('orig')
  const info = frames?.find(f => f.index === frameIndex)
  const isInsv = sourceKind === 'insv'
  const reg = recon ? frameReconMap(recon).get(frameIndex) : undefined

  // SAM3 マスク (native fisheye). 無ければ 404 で undefined.
  const { data: masks } = useQuery({
    queryKey: ['masks', projectId], queryFn: () => api.getMasks(projectId), retry: false,
  })
  const maskRec = masks?.kind === 'sam3_fisheye_masks'
    ? masks.frames.find(f => f.index === frameIndex)?.lenses?.find(l => l.lens === lens)
    : undefined
  // マスクが無い時は必ず原画表示 (壊れた画像を防ぐ).
  const eff = maskRec ? view : 'orig'

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
        <strong style={{ flex: 1 }}>frame {frameIndex}</strong>
        {isInsv && (
          <span className="seg">
            <button className={lens === 0 ? 'on' : ''} onClick={() => setLens(0)}>L0</button>
            <button className={lens === 1 ? 'on' : ''} onClick={() => setLens(1)}>L1</button>
          </span>
        )}
      </div>

      {maskRec && (
        <div className="seg" style={{ marginBottom: 8 }}>
          <button className={view === 'orig' ? 'on' : ''} onClick={() => setView('orig')}>{t('viewOrig')}</button>
          <button className={view === 'mask' ? 'on' : ''} onClick={() => setView('mask')}>{t('viewMask')}</button>
          <button className={view === 'overlay' ? 'on' : ''} onClick={() => setView('overlay')}>{t('viewOverlay')}</button>
        </div>
      )}

      {eff === 'mask'
        ? <img src={fisheyeMaskUrl(projectId, frameIndex, lens)} alt="mask"
            style={{ width: '100%', display: 'block', borderRadius: 6, background: '#111' }} />
        : <div style={{ position: 'relative', lineHeight: 0 }}>
            <img src={frameImageUrl(projectId, frameIndex, lens)} alt={`frame ${frameIndex}`}
              style={{ width: '100%', display: 'block', borderRadius: 6, background: '#111' }} />
            {eff === 'overlay' && (
              // マスクは 使う所=255 / 動体=0. multiply で動体 (除外) が黒く落ちて見える.
              <img src={fisheyeMaskUrl(projectId, frameIndex, lens)} alt="mask overlay"
                style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', borderRadius: 6, mixBlendMode: 'multiply' }} />
            )}
          </div>}

      {maskRec && (
        <div className="mono" style={{ fontSize: 11, marginTop: 6, color: maskRec.coverage_warning ? 'var(--error)' : 'var(--fg-mute)' }}>
          {t('maskCoverage')}: {(maskRec.coverage * 100).toFixed(1)}%{maskRec.coverage_warning ? ' ⚠' : ''}
        </div>
      )}

      <div className="ctl" style={{ marginTop: 10 }}>
        <label>{t('frameScore')}</label>
        <div className="mono" style={{ fontSize: 12 }}>
          {info?.score
            ? `${t('sharpness')} ${info.score.sharpness.toFixed(1)}`
              + (info.score.features != null ? ` · ${t('features')} ${info.score.features}` : '')
            : <span style={{ color: 'var(--fg-mute)' }}>{t('noScore')}</span>}
        </div>
      </div>

      {recon && (
        <div className="ctl">
          <label>{t('reconResult')}</label>
          {reg
            ? <div className="mono" style={{ fontSize: 12, color: '#3ad07a' }}>
                ✓ {t('registered')} · {t('points')} {reg.numPoints.toLocaleString()} · {reg.images} img
              </div>
            : <div className="mono" style={{ fontSize: 12, color: 'var(--error)' }}>⚠ {t('notRegistered')}</div>}
        </div>
      )}
    </div>
  )
}
