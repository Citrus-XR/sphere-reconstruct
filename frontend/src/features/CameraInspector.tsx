import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  api,
  parseImageName,
  preparedImageUrl,
  preparedMaskUrl,
  type MaskPurpose,
  type ProjectSource,
  type ReconstructionImage,
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

// カメラ選択時に Inspector に表示: 対応画像 + feature/training マスクの用途・重ね表示.
export const CameraInspector = ({
  projectId, image, sources, featureMaskRunning, trainingMaskRunning,
}: {
  projectId: string
  image: ReconstructionImage
  sources: ProjectSource[]
  featureMaskRunning: boolean
  trainingMaskRunning: boolean
}) => {
  const { t } = useSettings()
  const [view, setView] = useState<CaptureView>('orig')
  const [maskPurpose, setMaskPurpose] = useState<MaskPurpose>('training')
  const parsed = parseImageName(image.name)
  const source = sources.find(item => item.id === parsed?.sourceId)
  const frameMatch = image.name.match(/frame_(\d+)/)
  const frameIndex = parsed?.index ?? (frameMatch ? Number(frameMatch[1]) : null)
  const namedLens = image.name.match(/lens(\d+)/)
  const lens = parsed?.lens ?? (namedLens ? Number(namedLens[1]) : null)
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
  const masksByPurpose = {
    feature: featureMasks?.images.find(record => record.name === image.name),
    training: trainingMasks?.images.find(record => record.name === image.name),
  }
  const availablePurposes = (['feature', 'training'] as MaskPurpose[])
    .filter(purpose => masksByPurpose[purpose] !== undefined)
  const effectivePurpose = masksByPurpose[maskPurpose]
    ? maskPurpose
    : masksByPurpose.training ? 'training' : 'feature'
  const maskRec = masksByPurpose[effectivePurpose]
  const maskManifest = effectivePurpose === 'feature' ? featureMasks : trainingMasks
  const originalUrl = parsed ? preparedImageUrl(projectId, image.name) : null
  const maskUrl = maskRec
    ? preparedMaskUrl(projectId, image.name, effectivePurpose, maskManifest?.revision)
    : null

  return (
    <div>
      <InspectorHeader title={image.name}
        actions={<MaskPurposeTabs available={availablePurposes} selected={effectivePurpose}
          onSelect={setMaskPurpose} />} />
      {parsed && originalUrl ? (
        <CapturePreview originalUrl={originalUrl} maskUrl={maskUrl}
          view={view} onViewChange={setView} alt={image.name} />
      ) : (
        <div className="hint">{t('noCamImage')}</div>
      )}
      {frameIndex !== null && (
        <CaptureSummary frameIndex={frameIndex} lens={lens}
          pointsLabel={t('imagePoints')}
          registration={{ registered: true, points: image.num_points }} />
      )}
      <InspectorFields>
        {source && <InspectorField label={t('sourceLabel')} icon="folder">{source.label}</InspectorField>}
        <InspectorField label={t('datasetImage')} icon="image">{image.name}</InspectorField>
        <InspectorField label={t('cameraPosition')} icon="camera">
          [{image.position.map(value => value.toFixed(3)).join(', ')}]
        </InspectorField>
        {maskRec && (
          <InspectorField
            label={`${t(effectivePurpose === 'feature' ? 'featureMask' : 'trainingMask')} · ${t('maskCoverage')}`}
            icon="mask"
            tone={maskRec.coverage_warning ? 'error' : 'muted'}>
            {(maskRec.coverage * 100).toFixed(1)}%{maskRec.coverage_warning ? ' ⚠' : ''}
          </InspectorField>
        )}
      </InspectorFields>
    </div>
  )
}
