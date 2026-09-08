import { useEffect, useMemo } from 'react'
import type { StageStatus } from '../api/client'
import { useSettings } from '../ui/settings'
import { AppIcon } from './AppIcon'

export const CLEARABLE_STAGE_NAMES = [
  'inspect_source',
  'extract_frames',
  'prepare_images',
  'rectify_fisheye',
  'generate_feature_masks',
  'generate_training_masks',
  'extract_features',
  'match_features',
  'reconstruct',
  'align_reconstruction',
  'restore_metric_scale',
  'scene_alignment',
  'cleanup_sparse',
  'dense_initialization',
  'export_dataset',
] as const

const CONSUMERS: Record<string, string[]> = {
  inspect_source: ['extract_frames'],
  extract_frames: ['prepare_images'],
  prepare_images: ['rectify_fisheye'],
  rectify_fisheye: ['generate_feature_masks', 'generate_training_masks', 'extract_features'],
  generate_feature_masks: ['extract_features', 'export_dataset'],
  generate_training_masks: ['export_dataset'],
  extract_features: ['match_features'],
  match_features: ['reconstruct'],
  reconstruct: ['align_reconstruction'],
  align_reconstruction: ['restore_metric_scale'],
  restore_metric_scale: ['scene_alignment'],
  scene_alignment: ['cleanup_sparse'],
  cleanup_sparse: ['dense_initialization', 'export_dataset'],
  dense_initialization: ['export_dataset'],
  export_dataset: [],
}

const consumerClosure = (requested: Set<string>): Set<string> => {
  const result = new Set(requested)
  const pending = [...requested]
  while (pending.length) {
    const stage = pending.pop() as string
    for (const consumer of CONSUMERS[stage] ?? []) {
      if (result.has(consumer)) continue
      result.add(consumer)
      pending.push(consumer)
    }
  }
  return result
}

export const ClearOutputsDialog = ({
  stages,
  selectedStages,
  pending,
  error,
  onSelectedStagesChange,
  onClose,
  onClear,
}: {
  stages: StageStatus[]
  selectedStages: string[]
  pending: boolean
  error: unknown
  onSelectedStagesChange: (stages: string[]) => void
  onClose: () => void
  onClear: (stages: string[]) => void
}) => {
  const { t } = useSettings()
  const requested = useMemo(() => new Set(selectedStages), [selectedStages])
  const effective = useMemo(() => consumerClosure(requested), [requested])
  const statuses = useMemo(() => new Map(stages.map(stage => [stage.stage, stage])), [stages])
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !pending) onClose()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [onClose, pending])

  const toggle = (stage: string) => {
    const next = new Set(requested)
    if (next.has(stage)) next.delete(stage)
    else next.add(stage)
    onSelectedStagesChange([...next])
  }

  return (
    <div className="modal-back" onClick={() => !pending && onClose()}>
      <div className="modal clear-outputs-modal" onClick={event => event.stopPropagation()}
        role="dialog" aria-modal="true" aria-labelledby="clear-outputs-title">
        <div className="modal-head">
          <strong id="clear-outputs-title" style={{ flex: 1 }}>{t('clearOutputsTitle')}</strong>
          <button type="button" className="btn btn-secondary" onClick={onClose}
            disabled={pending} aria-label={t('close')}><AppIcon name="dismiss" /></button>
        </div>
        <div className="clear-output-shortcuts">
          <button type="button" className="clear-output-shortcut" disabled={pending}
            onClick={() => onClear(['inspect_source'])} data-testid="clear-all-outputs">
            <AppIcon name="delete" />
            <span><b>{t('clearAllOutputs')}</b><small>{t('clearAllOutputsHint')}</small></span>
          </button>
          <button type="button" className="clear-output-shortcut" disabled={pending}
            onClick={() => onClear(['prepare_images'])} data-testid="clear-after-frames">
            <AppIcon name="image" />
            <span><b>{t('clearAfterFrames')}</b><small>{t('clearAfterFramesHint')}</small></span>
          </button>
        </div>
        <div className="inspector-section-title clear-output-section-title">{t('clearCustom')}</div>
        <div className="hint clear-output-hint">{t('clearDependencyHint')}</div>
        <div className="modal-list clear-output-list">
          {CLEARABLE_STAGE_NAMES.map(stage => {
            const selected = effective.has(stage)
            const implied = selected && !requested.has(stage)
            const status = statuses.get(stage)
            return (
              <label key={stage} className={`clear-output-row${implied ? ' implied' : ''}`}>
                <input type="checkbox" checked={selected} disabled={pending || implied}
                  onChange={() => toggle(stage)} />
                <span>{t(`st_${stage}`)}</span>
                <small>{status?.has_output ? t('done') : t('notrun')}</small>
              </label>
            )
          })}
        </div>
        <div className="clear-output-footer">
          <span className="mono">{t('clearSelectedCount').replace('{count}', String(effective.size))}</span>
          <button type="button" className="btn btn-secondary" onClick={onClose} disabled={pending}>
            {t('cancel')}
          </button>
          <button type="button" className="btn clear-output-confirm" disabled={pending || requested.size === 0}
            onClick={() => onClear([...requested])} data-testid="clear-selected-outputs">
            <AppIcon name="delete" /> {t('clearSelected')}
          </button>
        </div>
        {error != null && <div className="error" style={{ margin: '0 12px 12px' }}>{String(error)}</div>}
      </div>
    </div>
  )
}
