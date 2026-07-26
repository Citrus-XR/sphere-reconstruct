import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  api,
  fisheyeMaskUrl,
  frameImageUrl,
  parseImageName,
  pinholeImageUrl,
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
export const CameraInspector = ({ projectId, image }: {
  projectId: string
  image: ReconstructionImage
}) => {
  const { t } = useSettings()
  const [view, setView] = useState<CaptureView>('orig')
  const parsed = parseImageName(image.name)
  const frameMatch = image.name.match(/frame_(\d+)/)
  const frameIndex = parsed?.index ?? (frameMatch ? Number(frameMatch[1]) : null)
  const namedLens = image.name.match(/lens(\d+)/)
  const lens = parsed?.lens ?? (namedLens ? Number(namedLens[1]) : null)
  const { data: masks } = useQuery({
    queryKey: ['masks', projectId], queryFn: () => api.getMasks(projectId), retry: false,
  })
  const maskFrame = parsed ? masks?.frames.find(frame => frame.index === parsed.index) : undefined
  const maskRec = parsed?.kind === 'native' && masks?.kind === 'sam3_fisheye_masks'
    ? maskFrame?.lenses?.find(item => item.lens === parsed.lens)
    : parsed?.kind === 'pinhole' && masks?.kind === 'sam3_pinhole_masks'
    ? maskFrame?.views?.find(item => item.lens === parsed.lens && item.view === parsed.view)
    : undefined
  const originalUrl = parsed?.kind === 'pinhole'
    ? pinholeImageUrl(projectId, parsed.index, parsed.view, parsed.lens)
    : parsed ? frameImageUrl(projectId, parsed.index, parsed.lens) : null
  const maskUrl = parsed?.kind === 'pinhole' && maskRec
    ? pinholeImageUrl(projectId, parsed.index, parsed.view, parsed.lens, true)
    : parsed?.kind === 'native' && maskRec ? fisheyeMaskUrl(projectId, parsed.index, parsed.lens) : null

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
