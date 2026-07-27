import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  api,
  frameImageUrl,
  frameReconMap,
  preparedMaskUrl,
  type FrameInfo,
  type MaskPurpose,
  type MasksManifest,
  type ProjectSource,
  type ReconstructionData,
} from '../api/client'
import { useSettings } from '../ui/settings'
import {
  CapturePreview,
  CaptureSummary,
  InspectorField,
  InspectorFields,
  InspectorHeader,
  MaskPurposeTabs,
  type CaptureView,
} from './CaptureInspectorParts'

// Hierarchy で写真を選んだときの Inspector: フレーム画像 + 抽出スコア + 処理結果 (再構成の
// 登録可否 / 3D 点数) + feature/training SAM3 マスクの用途・重ね表示を揃える.
export const FrameInspector = ({
  projectId, frameIndex, frames, recon, sources, featureMaskRunning, trainingMaskRunning,
}: {
  projectId: string
  frameIndex: number
  frames: FrameInfo[] | undefined
  recon: ReconstructionData | undefined
  sources: ProjectSource[]
  featureMaskRunning: boolean
  trainingMaskRunning: boolean
}) => {
  const { t } = useSettings()
  const [lens, setLens] = useState(0)
  const [view, setView] = useState<CaptureView>('orig')
  const [maskPurpose, setMaskPurpose] = useState<MaskPurpose>('training')
  const info = frames?.find(f => f.index === frameIndex)
  const source = sources.find(item => item.id === info?.source_id)
  const isInsv = source?.projection === 'dual_fisheye'
  const reg = recon ? frameReconMap(recon).get(frameIndex) : undefined

  const { data: featureMasks } = useQuery({
    queryKey: ['masks', projectId, 'feature'],
    queryFn: () => api.getMasks(projectId, 'feature'), retry: false,
    refetchInterval: featureMaskRunning ? 1000 : false,
  })
  const { data: trainingMasks } = useQuery({
    queryKey: ['masks', projectId, 'training'],
    queryFn: () => api.getMasks(projectId, 'training'), retry: false,
    refetchInterval: trainingMaskRunning ? 1000 : false,
  })
  const findMask = (manifest: MasksManifest | undefined) => {
    const candidates = manifest?.images.filter(
      record => record.source_id === info?.source_id && record.capture_index === info?.source_index,
    ) ?? []
    return isInsv
      ? candidates.find(record => new RegExp(`(^|/)${lens === 0 ? 'front' : 'back'}/`).test(record.name))
      : candidates[0]
  }
  const masksByPurpose = {
    feature: findMask(featureMasks),
    training: findMask(trainingMasks),
  }
  const availablePurposes = (['feature', 'training'] as MaskPurpose[])
    .filter(purpose => masksByPurpose[purpose] !== undefined)
  const effectivePurpose = masksByPurpose[maskPurpose]
    ? maskPurpose
    : masksByPurpose.training ? 'training' : 'feature'
  const maskRec = masksByPurpose[effectivePurpose]
  const maskManifest = effectivePurpose === 'feature' ? featureMasks : trainingMasks
  const eff = maskRec ? view : 'orig'
  const actions = isInsv || availablePurposes.length > 1 ? (
    <span style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
      {isInsv && (
        <span className="seg" role="group" aria-label={t('lensLabel')}>
          <button type="button" aria-pressed={lens === 0} className={lens === 0 ? 'on' : ''}
            onClick={() => setLens(0)}>L0</button>
          <button type="button" aria-pressed={lens === 1} className={lens === 1 ? 'on' : ''}
            onClick={() => setLens(1)}>L1</button>
        </span>
      )}
      <MaskPurposeTabs available={availablePurposes} selected={effectivePurpose}
        onSelect={setMaskPurpose} />
    </span>
  ) : undefined

  return (
    <div>
      <InspectorHeader
        title={`${source?.label ?? t('frameLabel')} · ${t('frameLabel')} ${info?.source_index ?? frameIndex}`}
        actions={actions} />

      <CapturePreview originalUrl={frameImageUrl(projectId, frameIndex, lens)}
        maskUrl={maskRec
          ? preparedMaskUrl(projectId, maskRec.name, effectivePurpose, maskManifest?.revision)
          : null}
        view={eff} onViewChange={setView} alt={`${t('frameLabel')} ${frameIndex}`} />

      <CaptureSummary frameIndex={info?.source_index ?? frameIndex} lens={isInsv ? lens : null}
        pointsLabel={t('framePoints')}
        registration={recon ? {
          registered: !!reg,
          points: reg?.numPoints ?? 0,
          images: reg?.images ?? 0,
        } : null} />

      {maskRec && (
        <InspectorFields>
          <InspectorField
            label={`${t(effectivePurpose === 'feature' ? 'featureMask' : 'trainingMask')} · ${t('maskCoverage')}`}
            tone={maskRec.coverage_warning ? 'error' : 'muted'}>
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
