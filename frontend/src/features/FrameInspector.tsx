import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  api,
  fisheyeMaskUrl,
  frameImageUrl,
  frameReconMap,
  type FrameInfo,
  type ReconstructionData,
} from '../api/client'
import { useSettings } from '../ui/settings'
import {
  CapturePreview,
  CaptureSummary,
  InspectorField,
  InspectorFields,
  InspectorHeader,
  type CaptureView,
} from './CaptureInspectorParts'

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
  const [view, setView] = useState<CaptureView>('orig')
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
      <InspectorHeader title={`${t('frameLabel')} ${frameIndex}`} actions={isInsv ? (
          <span className="seg" role="group" aria-label={t('lensLabel')}>
            <button type="button" aria-pressed={lens === 0} className={lens === 0 ? 'on' : ''} onClick={() => setLens(0)}>L0</button>
            <button type="button" aria-pressed={lens === 1} className={lens === 1 ? 'on' : ''} onClick={() => setLens(1)}>L1</button>
          </span>
        ) : undefined} />

      <CapturePreview originalUrl={frameImageUrl(projectId, frameIndex, lens)}
        maskUrl={maskRec ? fisheyeMaskUrl(projectId, frameIndex, lens) : null}
        view={eff} onViewChange={setView} alt={`${t('frameLabel')} ${frameIndex}`} />

      <CaptureSummary frameIndex={frameIndex} lens={isInsv ? lens : null}
        pointsLabel={t('framePoints')}
        registration={recon ? {
          registered: !!reg,
          points: reg?.numPoints ?? 0,
          images: reg?.images ?? 0,
        } : null} />

      {maskRec && (
        <InspectorFields>
          <InspectorField label={t('maskCoverage')} tone={maskRec.coverage_warning ? 'error' : 'muted'}>
            {(maskRec.coverage * 100).toFixed(1)}%{maskRec.coverage_warning ? ' ⚠' : ''}
          </InspectorField>
        </InspectorFields>
      )}

      <div className="inspector-section">
        <div className="inspector-section-title">{t('frameScore')}</div>
        <div className="mono">
          {info?.score
            ? `${t('sharpness')} ${info.score.sharpness.toFixed(1)}`
              + (info.score.features != null ? ` · ${t('features')} ${info.score.features}` : '')
            : <span style={{ color: 'var(--fg-mute)' }}>{t('noScore')}</span>}
        </div>
      </div>
    </div>
  )
}
