import type { ReactNode } from 'react'
import type { MaskPurpose } from '../api/client'
import { useSettings } from '../ui/settings'
import { AppIcon, type AppIconName } from '../components/AppIcon'

export const InspectorHeader = ({ title, actions, icon = 'camera' }: {
  title: ReactNode
  actions?: ReactNode
  icon?: AppIconName
}) => (
  <div className="inspector-header">
    <AppIcon name={icon} size={17} />
    <strong className="inspector-title">{title}</strong>
    {actions}
  </div>
)

export const InspectorFields = ({ children, testId }: { children: ReactNode; testId?: string }) => (
  <dl className="inspector-fields" data-testid={testId}>{children}</dl>
)

export const InspectorField = ({ label, children, tone, icon = 'layer' }: {
  label: string
  children: ReactNode
  tone?: 'success' | 'error' | 'muted'
  icon?: AppIconName
}) => (
  <div className={`inspector-field ${tone ?? ''}`.trim()}>
    <dt><AppIcon name={icon} size={13} /> {label}</dt>
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
      <InspectorField label={t('frameLabel')} icon="image">{frameIndex}</InspectorField>
      {lens !== null && <InspectorField label={t('lensLabel')} icon="camera">L{lens}</InspectorField>}
      {registration && (
        <>
          <InspectorField label={t('registrationLabel')} icon={registration.registered ? 'checkmark' : 'warning'}
            tone={registration.registered ? 'success' : 'error'}>
            {registration.registered ? t('registered') : t('notRegistered')}
          </InspectorField>
          <InspectorField label={pointsLabel} icon="points" tone={registration.registered ? undefined : 'muted'}>
            {registration.registered ? registration.points.toLocaleString() : '—'}
          </InspectorField>
          {registration.images !== undefined && (
            <InspectorField label={t('registeredImages')} icon="image">
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
  circular = false,
}: {
  originalUrl: string
  maskUrl: string | null
  view: CaptureView
  onViewChange: (view: CaptureView) => void
  alt: string
  circular?: boolean
}) => {
  const { t } = useSettings()
  const maskAvailable = maskUrl !== null
  const effectiveView = maskAvailable ? view : 'orig'
  return (
    <>
      {maskAvailable && (
        <div className="seg inspector-preview-tabs" role="tablist" aria-label={t('previewVariant')}>
          <button type="button" role="tab" aria-selected={effectiveView === 'orig'} className={effectiveView === 'orig' ? 'on' : ''}
            onClick={() => onViewChange('orig')}><AppIcon name="image" size={14} /> {t('viewOrig')}</button>
          <button type="button" role="tab" aria-selected={effectiveView === 'mask'} className={effectiveView === 'mask' ? 'on' : ''}
            onClick={() => onViewChange('mask')}><AppIcon name="mask" size={14} /> {t('viewMask')}</button>
          <button type="button" role="tab" aria-selected={effectiveView === 'overlay'} className={effectiveView === 'overlay' ? 'on' : ''}
            onClick={() => onViewChange('overlay')}><AppIcon name="overlay" size={14} /> {t('viewOverlay')}</button>
        </div>
      )}
      <div className={`inspector-preview${circular ? ' circular' : ''}`}>
        {effectiveView === 'mask'
          ? <img className="inspector-preview-image" src={maskUrl ?? ''} alt={t('viewMask')} />
          : <>
              <img className="inspector-preview-image" src={originalUrl} alt={alt} />
              {effectiveView === 'overlay' && (
                <img className="inspector-preview-mask" src={maskUrl ?? ''} alt={t('viewOverlay')} />
              )}
            </>}
      </div>
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
          <AppIcon name={purpose === 'feature' ? 'target' : 'image'} size={14} />
          {t(purpose === 'feature' ? 'featureMask' : 'trainingMask')}
        </button>
      ))}
    </span>
  )
}

export type CaptureView = 'orig' | 'mask' | 'overlay'
