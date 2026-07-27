import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  api,
  parseImageName,
  preparedImageUrl,
  preparedMaskUrl,
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
  type CaptureView,
} from './CaptureInspectorParts'

// カメラ選択時に Inspector に表示: 対応フレーム画像 + 適用マスクの重ね表示トグル.
export const CameraInspector = ({ projectId, image, sources }: {
  projectId: string
  image: ReconstructionImage
  sources: ProjectSource[]
}) => {
  const { t } = useSettings()
  const [view, setView] = useState<CaptureView>('orig')
  const parsed = parseImageName(image.name)
  const source = sources.find(item => item.id === parsed?.sourceId)
  const frameMatch = image.name.match(/frame_(\d+)/)
  const frameIndex = parsed?.index ?? (frameMatch ? Number(frameMatch[1]) : null)
  const namedLens = image.name.match(/lens(\d+)/)
  const lens = parsed?.lens ?? (namedLens ? Number(namedLens[1]) : null)
  const { data: masks } = useQuery({
    queryKey: ['masks', projectId], queryFn: () => api.getMasks(projectId), retry: false,
  })
  const maskRec = masks?.images.find(record => record.name === image.name)
  const originalUrl = parsed ? preparedImageUrl(projectId, image.name) : null
  const maskUrl = maskRec ? preparedMaskUrl(projectId, image.name) : null

  return (
    <div>
      <InspectorHeader title={image.name} />
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
        {source && <InspectorField label={t('sourceLabel')}>{source.label}</InspectorField>}
        <InspectorField label={t('datasetImage')}>{image.name}</InspectorField>
        <InspectorField label={t('cameraPosition')}>
          [{image.position.map(value => value.toFixed(3)).join(', ')}]
        </InspectorField>
        {maskRec && (
          <InspectorField label={t('maskCoverage')} tone={maskRec.coverage_warning ? 'error' : 'muted'}>
            {(maskRec.coverage * 100).toFixed(1)}%{maskRec.coverage_warning ? ' ⚠' : ''}
          </InspectorField>
        )}
      </InspectorFields>
    </div>
  )
}
