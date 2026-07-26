import type { ReactNode } from 'react'
import { useSettings } from '../ui/settings'

export const InspectorHeader = ({ title, actions }: { title: ReactNode; actions?: ReactNode }) => (
  <div className="inspector-header">
    <strong className="inspector-title">{title}</strong>
    {actions}
  </div>
)

export const InspectorFields = ({ children, testId }: { children: ReactNode; testId?: string }) => (
  <dl className="inspector-fields" data-testid={testId}>{children}</dl>
)

export const InspectorField = ({ label, children, tone }: {
  label: string
  children: ReactNode
  tone?: 'success' | 'error' | 'muted'
}) => (
  <div className={`inspector-field ${tone ?? ''}`.trim()}>
    <dt>{label}</dt>
    <dd>{children}</dd>
  </div>
)

export const CaptureSummary = ({
  frameIndex,
  lens,
  registration,
  pointsLabel,
}: {
  frameIndex: number
  lens: number | null
  registration: { registered: boolean; points: number; images?: number } | null
  pointsLabel: string
}) => {
  const { t } = useSettings()
  return (
    <InspectorFields testId="capture-summary">
      <InspectorField label={t('frameLabel')}>{frameIndex}</InspectorField>
      {lens !== null && <InspectorField label={t('lensLabel')}>L{lens}</InspectorField>}
      {registration && (
        <>
          <InspectorField label={t('registrationLabel')} tone={registration.registered ? 'success' : 'error'}>
            {registration.registered ? `✓ ${t('registered')}` : `⚠ ${t('notRegistered')}`}
          </InspectorField>
          <InspectorField label={pointsLabel} tone={registration.registered ? undefined : 'muted'}>
            {registration.registered ? registration.points.toLocaleString() : '—'}
          </InspectorField>
          {registration.images !== undefined && (
            <InspectorField label={t('registeredImages')}>
              {registration.images.toLocaleString()}
            </InspectorField>
          )}
        </>
      )}
    </InspectorFields>
  )
}

export const CapturePreview = ({
  originalUrl,
  maskUrl,
  denoisedUrl,
  view,
  onViewChange,
  alt,
}: {
  originalUrl: string
  maskUrl: string | null
  denoisedUrl: string | null
  view: CaptureView
  onViewChange: (view: CaptureView) => void
  alt: string
}) => {
  const { t } = useSettings()
  const maskAvailable = maskUrl !== null
  const denoisedAvailable = denoisedUrl !== null
  const effectiveView = view === 'denoised'
    ? (denoisedAvailable ? view : 'orig')
    : (maskAvailable ? view : 'orig')
  return (
    <>
      {maskAvailable && (
        <div className="seg inspector-preview-tabs" role="tablist" aria-label={t('previewVariant')}>
          <button type="button" role="tab" aria-selected={effectiveView === 'orig'} className={effectiveView === 'orig' ? 'on' : ''}
            onClick={() => onViewChange('orig')}>{t('viewOrig')}</button>
          <button type="button" role="tab" aria-selected={effectiveView === 'mask'} className={effectiveView === 'mask' ? 'on' : ''}
            onClick={() => onViewChange('mask')}>{t('viewMask')}</button>
          <button type="button" role="tab" aria-selected={effectiveView === 'overlay'} className={effectiveView === 'overlay' ? 'on' : ''}
            onClick={() => onViewChange('overlay')}>{t('viewOverlay')}</button>
          {denoisedAvailable && <button type="button" role="tab" aria-selected={effectiveView === 'denoised'}
            className={effectiveView === 'denoised' ? 'on' : ''} onClick={() => onViewChange('denoised')}>{t('viewDenoised')}</button>}
        </div>
      )}
      {!maskAvailable && denoisedAvailable && (
        <div className="seg inspector-preview-tabs" role="tablist" aria-label={t('previewVariant')}>
          <button type="button" role="tab" aria-selected={effectiveView === 'orig'} className={effectiveView === 'orig' ? 'on' : ''}
            onClick={() => onViewChange('orig')}>{t('viewOrig')}</button>
          <button type="button" role="tab" aria-selected={effectiveView === 'denoised'} className={effectiveView === 'denoised' ? 'on' : ''}
            onClick={() => onViewChange('denoised')}>{t('viewDenoised')}</button>
        </div>
      )}
      {effectiveView === 'mask'
        ? <img className="inspector-preview-image" src={maskUrl ?? ''} alt={t('viewMask')} />
        : effectiveView === 'denoised'
        ? <img className="inspector-preview-image" src={denoisedUrl ?? ''} alt={t('viewDenoised')} />
        : <div className="inspector-preview">
            <img className="inspector-preview-image" src={originalUrl} alt={alt} />
            {effectiveView === 'overlay' && (
              <img className="inspector-preview-mask" src={maskUrl ?? ''} alt={t('viewOverlay')} />
            )}
          </div>}
    </>
  )
}

export type CaptureView = 'orig' | 'mask' | 'overlay' | 'denoised'
