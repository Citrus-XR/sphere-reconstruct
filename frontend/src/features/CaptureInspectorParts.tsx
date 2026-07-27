import type { ReactNode } from 'react'
import type { MaskPurpose } from '../api/client'
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
  view,
  onViewChange,
  alt,
}: {
  originalUrl: string
  maskUrl: string | null
  view: CaptureView
  onViewChange: (view: CaptureView) => void
  alt: string
}) => {
  const { t } = useSettings()
  const maskAvailable = maskUrl !== null
  const effectiveView = maskAvailable ? view : 'orig'
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
        </div>
      )}
      {effectiveView === 'mask'
        ? <img className="inspector-preview-image" src={maskUrl ?? ''} alt={t('viewMask')} />
        : <div className="inspector-preview">
            <img className="inspector-preview-image" src={originalUrl} alt={alt} />
            {effectiveView === 'overlay' && (
              <img className="inspector-preview-mask" src={maskUrl ?? ''} alt={t('viewOverlay')} />
            )}
          </div>}
    </>
  )
}

export const MaskPurposeTabs = ({ available, selected, onSelect }: {
  available: MaskPurpose[]
  selected: MaskPurpose
  onSelect: (purpose: MaskPurpose) => void
}) => {
  const { t } = useSettings()
  if (available.length < 2) return null
  return (
    <span className="seg" role="group" aria-label={t('maskPurpose')}>
      {available.map(purpose => (
        <button type="button" key={purpose} aria-pressed={selected === purpose}
          className={selected === purpose ? 'on' : ''} onClick={() => onSelect(purpose)}>
          {t(purpose === 'feature' ? 'featureMask' : 'trainingMask')}
        </button>
      ))}
    </span>
  )
}

export type CaptureView = 'orig' | 'mask' | 'overlay'
