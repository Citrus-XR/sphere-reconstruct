import { useEffect, useId, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  api,
  type FrameSelection,
  type ProjectSource,
  type Projection,
  type SourceCreate,
  type SourceInfo,
  type StageStatus,
} from '../api/client'
import { paramsForStage, QUALITY_PRESETS, type ReconMode, type StageParams } from './stageParams'
import { ProgressRing } from '../components/ProgressRing'
import { PathText } from '../components/PathText'
import { useSettings } from '../ui/settings'

type SourcePreset = 'insv' | 'erp_video' | 'erp_images' | 'perspective_video' | 'perspective_images'

const SOURCE_PRESETS: Record<SourcePreset, Omit<SourceCreate, 'path' | 'label' | 'role'>> = {
  insv: { adapter: 'insta360_insv', media_kind: 'video', projection: 'dual_fisheye' },
  erp_video: { adapter: 'generic_video', media_kind: 'video', projection: 'equirectangular' },
  erp_images: { adapter: 'generic_images', media_kind: 'images', projection: 'equirectangular' },
  perspective_video: { adapter: 'generic_video', media_kind: 'video', projection: 'perspective' },
  perspective_images: { adapter: 'generic_images', media_kind: 'images', projection: 'perspective' },
}

// 経過秒を mm:ss (1h 以上は h:mm:ss) に整形.
const fmtDur = (sec: number): string => {
  const s = Math.max(0, Math.floor(sec))
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), ss = s % 60
  const p = (n: number) => String(n).padStart(2, '0')
  return h > 0 ? `${h}:${p(m)}:${p(ss)}` : `${m}:${p(ss)}`
}

// 右パネル「生成設定」: 選択中工程のパラメータ + 再生成 / クリア. 各コントロール下に説明.
export const StageSettings = ({
  projectId, stage, status, sourceInfo, reconMode, setReconMode, params, setParams, onJob, hasSource,
  sources, resultMode, primaryProjection, processing, stageIsRunning, onStop,
  onSelectSource, onDeleteSource, onMakePrimarySource, sourceMutationError, frameSelection,
  stageProgress, stageStartedAt, stageProgressMsg, blockedReason,
}: {
  projectId: string
  stage: string
  status: StageStatus | undefined
  sourceInfo: SourceInfo | undefined
  reconMode: ReconMode
  setReconMode: (m: ReconMode) => void
  params: StageParams
  setParams: (p: Partial<StageParams>) => void
  onJob: (jobId: string) => void
  hasSource: boolean
  sources: ProjectSource[]
  resultMode: 'native' | 'pinhole' | 'equirect' | 'mixed' | null
  primaryProjection: Projection | null
  processing: boolean
  stageIsRunning: boolean
  onStop: () => void
  onSelectSource: (source: Omit<SourceCreate, 'path'>) => void
  onDeleteSource: (sourceId: string) => void
  onMakePrimarySource: (sourceId: string) => void
  sourceMutationError: unknown
  frameSelection: FrameSelection | null | undefined
  stageProgress: number | null
  stageStartedAt: string | null
  stageProgressMsg: string
  blockedReason: string | null
}) => {
  const { t } = useSettings()
  const qc = useQueryClient()
  const { data: doctor } = useQuery({ queryKey: ['doctor'], queryFn: api.getDoctor, staleTime: 30_000 })
  const [advOpen, setAdvOpen] = useState(false)
  const [colmapAdvOpen, setColmapAdvOpen] = useState(false)
  const [sourcePreset, setSourcePreset] = useState<SourcePreset>('insv')
  // 実行中は 1s 毎に now を進めて経過/予測終了を更新する.
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    if (!stageIsRunning) return
    const id = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(id)
  }, [stageIsRunning])

  const run = useMutation({
    mutationFn: () => api.rerunStage(projectId, stage, { [stage]: paramsForStage(stage, params, reconMode) }),
    onSuccess: r => { onJob(r.job_id); qc.invalidateQueries({ queryKey: ['stages', projectId] }) },
  })
  const clear = useMutation({
    mutationFn: () => api.clearStage(projectId, stage),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['stages', projectId] })
      // 成果物が消えると 404 になるクエリは remove で破棄 (invalidate だと前回 data が残る).
      for (const key of ['reconstruction', 'frames', 'fisheye-region', 'masks', 'export-info']) {
        qc.removeQueries({ queryKey: [key, projectId] })
      }
    },
  })
  // export_dataset の出力先 (絶対パス) を Inspector に表示する.
  const { data: exportInfo } = useQuery({
    queryKey: ['export-info', projectId], queryFn: () => api.getExportInfo(projectId),
    enabled: stage === 'export_dataset' && !!status?.has_output, retry: false,
  })

  const predBase = sourceInfo?.duration_sec ? Math.floor(sourceInfo.duration_sec * params.fps) : null
  const predCapped = predBase != null && params.maxFrames > 0 ? Math.min(predBase, params.maxFrames) : predBase
  const maskPurpose = stage === 'generate_feature_masks'
    ? 'feature'
    : stage === 'generate_training_masks' ? 'training' : null
  const featureMask = maskPurpose === 'feature'
  const maskEnabled = maskPurpose === null
    ? true
    : featureMask ? params.featureMaskEnabled : params.trainingMaskEnabled
  const maskDownsample = featureMask ? params.featureMaskDownsampleOn : params.trainingMaskDownsampleOn
  const maskSize = featureMask ? params.featureMaskSize : params.trainingMaskSize
  const maskDilateOn = featureMask ? params.featureMaskDilateOn : params.trainingMaskDilateOn
  const maskDilate = featureMask ? params.featureMaskDilate : params.trainingMaskDilate
  const maskPrompt = featureMask ? params.featureMaskPrompt : params.trainingMaskPrompt
  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
        <strong style={{ flex: 1 }}>{t(`st_${stage}`)}</strong>
        {processing && stageIsRunning
          ? <button className="btn stop" onClick={onStop}>■ {t('stop')}</button>
          : <button className="btn"
              disabled={run.isPending || !hasSource || processing || !!blockedReason || !maskEnabled}
              title={!hasSource ? t('noSource') : processing ? t('otherRunning')
                : !maskEnabled ? t('maskStepDisabled') : blockedReason ?? ''}
              onClick={() => run.mutate()}>
              {status?.has_output ? t('regenerate') : t('generate')}
            </button>}
        <button className="btn btn-secondary" disabled={clear.isPending || !status?.has_output || processing}
          onClick={() => clear.mutate()}>{t('clear')}</button>
      </div>
      {stageIsRunning && (() => {
        // 停止ボタン下の実行状況行: 環形進捗 + 経過時間 + 現在処理中の項目.
        const startMs = stageStartedAt ? new Date(stageStartedAt).getTime() : now
        const elapsed = Math.max(0, (now - startMs) / 1000)
        return (
          <div style={{ marginBottom: 10 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <ProgressRing value={stageProgress} size={24} stroke={3} showLabel
                label={t(`st_${stage}`)} />
              <div className="mono" style={{ fontSize: 11, color: 'var(--fg-mute)' }}>
                {t('elapsed')}: {fmtDur(elapsed)}
              </div>
            </div>
            {stageProgressMsg && (
              <div className="mono" style={{ fontSize: 11, color: 'var(--fg-mute)', marginTop: 4,
                overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {stageProgressMsg}
              </div>
            )}
          </div>
        )
      })()}
      {!hasSource && stage !== 'inspect_source' && <div className="hint" style={{ color: '#d69a2a', marginBottom: 8 }}>{t('needSource')}</div>}
      {blockedReason && <div className="hint" style={{ color: '#d69a2a', marginBottom: 8 }}>{blockedReason}</div>}
      {run.error && <div className="error">{String(run.error)}</div>}
      {clear.error && <div className="error">{String(clear.error)}</div>}
      {status?.status === 'failed' && status.error_text && <div className="error">{status.error_text}</div>}
      {status?.has_output && status.extra && <StageResult key={stage} stage={stage} extra={status.extra} />}

      {stage === 'extract_frames' && (
        <>
          <div className="ctl">
            <label>{t('lbl_method')}</label>
            <select className="input" value={params.method}
              onChange={e => setParams({ method: e.target.value as StageParams['method'] })}>
              <option value="interval">{t('method_interval')}</option>
              <option value="sharpness">{t('method_sharpness')}</option>
              <option value="spatial">{t('method_spatial')}</option>
            </select>
            <div className="hint">{t('hint_method')}</div>
          </div>

          {params.method !== 'spatial' ? (
            <>
              <Slider label={t('lbl_fps')} hint={t('hint_fps')} min={0.1} max={5} step={0.1} value={params.fps}
                onChange={v => setParams({ fps: v })} fmt={v => `${v.toFixed(1)} fps`} />
              {params.method === 'sharpness' && (
                <div className="ctl">
                  <label>{t('lbl_sharpLevel')}</label>
                  <select className="input" value={params.sharpnessLevel}
                    onChange={e => setParams({ sharpnessLevel: e.target.value as StageParams['sharpnessLevel'] })}>
                    <option value="basic">Basic</option><option value="better">Better</option><option value="best">Best</option>
                  </select>
                  <div className="hint">{t('hint_sharpLevel')}</div>
                </div>
              )}
              <div className="predict">
                {t('predFrames')}: {predCapped == null ? '—' :
                  params.method === 'interval' ? `~${predCapped}` : `~${predCapped}–${Math.floor(predCapped * 1.2)}`}
                {sourceInfo?.duration_sec ? ` (${sourceInfo.duration_sec.toFixed(0)}s)` : ''}
              </div>
            </>
          ) : (
            <>
              <Slider label={`① ${t('lbl_sharpThreshold')}`} hint={t('hint_sharpThreshold')} min={0} max={300} step={10}
                value={params.minSharpness} onChange={v => setParams({ minSharpness: Math.round(v) })}
                fmt={v => v === 0 ? t('off') : `${v}`} />
              <Slider label={`② ${t('lbl_motion')}`} hint={t('hint_motion')} min={0.5} max={5} step={0.25}
                value={params.targetMotion} onChange={v => setParams({ targetMotion: v })} fmt={v => `${v.toFixed(2)}`} />
              <div className="predict">{t('predSpatial')}</div>
              {frameSelection?.reasons && (
                <div className="hint" style={{ marginTop: 4 }}>
                  {t('selReasons')}: 🌫 {frameSelection.reasons.blur} · ☀ {frameSelection.reasons.exposure} · ✦ {frameSelection.reasons.few_features} · RS {frameSelection.reasons.rolling_shutter ?? 0}
                  {' '}({t('selKept')}: {frameSelection.selected}/{frameSelection.candidates ?? '—'})
                  {frameSelection.fallback && ` · ${t('selFallback')}`}
                </div>
              )}
            </>
          )}

          <div className="ctl">
            <label>{t('lbl_maxframes')}</label>
            <input className="input" type="number" min={0} value={params.maxFrames}
              onChange={e => setParams({ maxFrames: Number(e.target.value) })} />
            <div className="hint">{t('hint_maxframes')}</div>
          </div>

          <Slider label={t('lbl_rsMotion')} hint={t('hint_rsMotion')} min={0.1} max={2} step={0.1}
            value={params.maxRollingShutterMotion}
            onChange={v => setParams({ maxRollingShutterMotion: v })} fmt={v => `${v.toFixed(1)}°`} />

          {params.method === 'spatial' && (
            <div className="ctl">
              <div style={{ cursor: 'pointer', userSelect: 'none', color: 'var(--fg-mute)', fontSize: 12 }}
                onClick={() => setAdvOpen(v => !v)}>{advOpen ? '▾' : '▸'} {t('lbl_advanced')}</div>
              {advOpen && (
                <div style={{ marginTop: 6 }}>
                  <Slider label={t('lbl_candidateFps')} hint={t('hint_candidateFps')} min={1} max={10} step={0.5}
                    value={params.candidateFps} onChange={v => setParams({ candidateFps: v })} fmt={v => `${v.toFixed(1)} fps`} />
                  <div className="ctl">
                    <label>{t('lbl_minFeatures')}</label>
                    <input className="input" type="number" min={0} value={params.minFeatures}
                      onChange={e => setParams({ minFeatures: Number(e.target.value) })} />
                    <div className="hint">{t('hint_minFeatures')}</div>
                  </div>
                  <Slider label={t('lbl_maxClip')} hint={t('hint_maxClip')} min={0} max={1} step={0.05}
                    value={params.maxClip} onChange={v => setParams({ maxClip: v })} fmt={v => v.toFixed(2)} />
                </div>
              )}
            </div>
          )}
        </>
      )}

      {maskPurpose && (
        <>
          <div className="ctl">
            <label style={{ display: 'inline-flex', gap: 6, alignItems: 'center' }}>
              <input type="checkbox" checked={maskEnabled} onChange={() => setParams(featureMask
                ? { featureMaskEnabled: !params.featureMaskEnabled }
                : { trainingMaskEnabled: !params.trainingMaskEnabled })} />
              {t(featureMask ? 'enableFeatureMasks' : 'enableTrainingMasks')}
            </label>
            <div className="hint">{t(featureMask ? 'hintFeatureMasks' : 'hintTrainingMasks')}</div>
            {featureMask && reconMode === 'native_fisheye' && <div className="hint">{t('hint_sam3circle')}</div>}
          </div>
          <div className="ctl">
            <label style={{ display: 'inline-flex', gap: 6, alignItems: 'center' }}>
              <input type="checkbox" checked={maskDownsample} onChange={() => setParams(featureMask
                ? { featureMaskDownsampleOn: !params.featureMaskDownsampleOn }
                : { trainingMaskDownsampleOn: !params.trainingMaskDownsampleOn })} /> {t('lbl_downsample')}
            </label>
            <div className="hint">{t('hint_downsample')}</div>
          </div>
          {maskDownsample && (
            <Slider label={t('lbl_masksize')} hint={t('hint_masksize')} min={6} max={14} step={1}
              value={Math.round(Math.log2(maskSize))}
              onChange={v => setParams(featureMask
                ? { featureMaskSize: 2 ** v }
                : { trainingMaskSize: 2 ** v })} fmt={v => `${2 ** v}px`} />
          )}
          <div className="ctl">
            <label style={{ display: 'inline-flex', gap: 6, alignItems: 'center' }}>
              <input type="checkbox" checked={maskDilateOn} onChange={() => setParams(featureMask
                ? { featureMaskDilateOn: !params.featureMaskDilateOn }
                : { trainingMaskDilateOn: !params.trainingMaskDilateOn })} /> {t('lbl_dilate')}
            </label>
            <div className="hint">{t('hint_dilate')}</div>
          </div>
          {maskDilateOn && (
            <Slider label={t('lbl_dilate')} hint="" min={2} max={40} step={2}
              value={maskDilate} onChange={v => setParams(featureMask
                ? { featureMaskDilate: Math.round(v) }
                : { trainingMaskDilate: Math.round(v) })} fmt={v => `${v}px`} />
          )}
          <div className="ctl">
            <label>{t(featureMask ? 'lblFeaturePrompt' : 'lblTrainingPrompt')}</label>
            <input className="input" placeholder="person,animal,..." value={maskPrompt}
              onChange={e => setParams(featureMask
                ? { featureMaskPrompt: e.target.value }
                : { trainingMaskPrompt: e.target.value })} />
            <div className="hint">{t(featureMask ? 'hintFeaturePrompt' : 'hintTrainingPrompt')}</div>
          </div>
        </>
      )}

      {stage === 'extract_features' && (
        <>
          <div className="hint" style={{ marginBottom: 8 }}>
            {t('lbl_mode')}: {reconMode === 'native_fisheye' ? t('modeNative')
              : reconMode === 'equirectangular' ? t('modeEquirect') : t('modePinhole')} ({t('st_inspect_source')})
          </div>
          <div className="ctl">
            <label>{t('lbl_qualityPreset')}</label>
            <select className="input" value={params.qualityPreset}
              onChange={e => {
                const value = e.target.value as StageParams['qualityPreset']
                setParams(value === 'custom'
                  ? { qualityPreset: value }
                  : { qualityPreset: value, ...QUALITY_PRESETS[value] })
              }}>
              <option value="draft">{t('preset_draft')}</option>
              <option value="standard">{t('preset_standard')}</option>
              <option value="high">{t('preset_high')}</option>
              <option value="custom">{t('preset_custom')}</option>
            </select>
            <div className="hint">{t('hint_qualityPreset')}</div>
          </div>
          <div className="ctl">
            <label>{t('lbl_backend')}</label>
            <select className="input" value={params.featureType}
              onChange={e => {
                const featureType = e.target.value as StageParams['featureType']
                setParams({ featureType, loopClosure: featureType === 'SIFT', qualityPreset: 'custom' })
              }}>
              <option value="ALIKED_N16ROT">ALIKED N16ROT</option>
              <option value="ALIKED_N32">ALIKED N32</option>
              <option value="SIFT">SIFT</option>
            </select>
            <div className="hint">{t('hint_backend')}</div>
          </div>
          <NumField label={t('f_maxImageSize')} hint={t('hint_maxImageSize')} value={params.featureMaxImageSize}
            onChange={value => setParams({ featureMaxImageSize: value, qualityPreset: 'custom' })} />
          <NumField label={t('f_maxFeatures')} hint={t('hint_maxFeatures')} value={params.featureMaxNumFeatures}
            onChange={value => setParams({ featureMaxNumFeatures: value, qualityPreset: 'custom' })} />
          <label style={{ display: 'inline-flex', gap: 6, alignItems: 'center', margin: '4px 0' }}>
            <input type="checkbox" checked={params.featureUseGpu}
              onChange={() => setParams({ featureUseGpu: !params.featureUseGpu })} /> GPU
          </label>
          {params.featureType === 'SIFT' && <>
            <NumField label={t('f_peakThreshold')} hint={t('hint_peakThreshold')} value={params.siftPeakThreshold} step={0.0001}
              onChange={value => setParams({ siftPeakThreshold: value, qualityPreset: 'custom' })} />
            <NumField label={t('f_edgeThreshold')} hint={t('hint_edgeThreshold')} value={params.siftEdgeThreshold} step={0.5}
              onChange={value => setParams({ siftEdgeThreshold: value, qualityPreset: 'custom' })} />
            <div className="ctl">
              <label style={{ display: 'inline-flex', gap: 6, alignItems: 'center' }}>
                <input type="checkbox" checked={params.siftAffineDsp}
                  onChange={() => setParams({ siftAffineDsp: !params.siftAffineDsp, qualityPreset: 'custom' })} /> {t('f_affineDsp')}
              </label>
              <div className="hint">{t('hint_affineDsp')}</div>
            </div>
          </>}
        </>
      )}

      {stage === 'match_features' && (
        <>
          <div className="ctl">
            <label>{t('lbl_matcher')}</label>
            <select className="input" value={params.matcherType}
              onChange={e => setParams({ matcherType: e.target.value as StageParams['matcherType'], qualityPreset: 'custom' })}>
              <option value="bruteforce">Brute-force</option>
              <option value="lightglue">LightGlue</option>
            </select>
            <div className="hint">{t('hint_matcher')}</div>
          </div>
          <div className="ctl">
            <label>{t('lbl_pairing')}</label>
            <select className="input" value={params.pairing}
              onChange={e => setParams({ pairing: e.target.value as StageParams['pairing'] })}>
              <option value="auto">{t('matcher_auto')}</option>
              <option value="sequential">{t('matcher_sequential')}</option>
              <option value="exhaustive">{t('matcher_exhaustive')}</option>
              <option value="vocab_tree">{t('matcher_vocab')}</option>
            </select>
            <div className="hint">{t('hint_pairing')}</div>
          </div>
          {(params.pairing === 'sequential' || (params.pairing === 'auto' && sources.length === 1)) && <>
            <Slider label={t('lbl_overlap')} hint={t('hint_overlap')} min={2} max={20} step={1}
              value={params.overlap} onChange={value => setParams({ overlap: Math.round(value) })} fmt={value => `${value}`} />
            <div className="ctl">
              <label style={{ display: 'inline-flex', gap: 6, alignItems: 'center' }}>
                <input type="checkbox" checked={params.loopClosure}
                  onChange={() => setParams({ loopClosure: !params.loopClosure })} /> {t('lbl_loopClosure')}
              </label>
              <div className="hint">{t('hint_loopClosure')}</div>
            </div>
            <div className="ctl">
              <label style={{ display: 'inline-flex', gap: 6, alignItems: 'center' }}>
                <input type="checkbox" checked={params.transitiveMatching}
                  onChange={() => setParams({ transitiveMatching: !params.transitiveMatching })} /> {t('lbl_transitiveMatching')}
              </label>
              <div className="hint">{t('hint_transitiveMatching')}</div>
            </div>
          </>}
          <NumField label={t('f_maxMatches')} hint={t('hint_maxMatches')} value={params.maxNumMatches}
            onChange={value => setParams({ maxNumMatches: value, qualityPreset: 'custom' })} />
          <NumField label={t('f_twoViewInliers')} hint={t('hint_twoViewInliers')} value={params.twoViewMinInliers}
            onChange={value => setParams({ twoViewMinInliers: value })} />
          <div className="ctl">
            <label style={{ display: 'inline-flex', gap: 6, alignItems: 'center' }}>
              <input type="checkbox" checked={params.guidedMatching}
                onChange={() => setParams({ guidedMatching: !params.guidedMatching })} /> {t('f_guidedMatching')}
            </label>
            <div className="hint">{t('hint_guidedMatching')}</div>
          </div>
          <div className="ctl">
            <label style={{ display: 'inline-flex', gap: 6, alignItems: 'center' }}>
              <input type="checkbox" checked={params.rigVerification}
                onChange={() => setParams({ rigVerification: !params.rigVerification })} /> {t('lbl_rigVerification')}
            </label>
            <div className="hint">{t('hint_rigVerification')}</div>
          </div>
        </>
      )}

      {stage === 'reconstruct' && (
        <>
          <div className="ctl">
            <label>{t('lbl_mapper')}</label>
            <select className="input" value={params.mapper}
              onChange={e => setParams({
                mapper: e.target.value as StageParams['mapper'],
                viewGraphCalibration: e.target.value === 'global',
              })}>
              <option value="global">{t('mapperGlobal')}</option>
              <option value="incremental">{t('mapperIncremental')}</option>
            </select>
            <div className="hint">{t('hint_mapper')}</div>
          </div>
          {params.mapper === 'global' && <div className="ctl">
            <label style={{ display: 'inline-flex', gap: 6, alignItems: 'center' }}>
                <input type="checkbox" checked={params.viewGraphCalibration}
                onChange={() => setParams({ viewGraphCalibration: !params.viewGraphCalibration })} /> {t('viewGraphCalibration')}
            </label>
            <div className="hint">{t('hint_viewGraphCalibration')}</div>
          </div>}
          <div className="ctl">
            <label style={{ display: 'inline-flex', gap: 6, alignItems: 'center' }}>
              <input type="checkbox" checked={params.baUseGpu}
                disabled={!doctor?.checks.colmap?.capabilities?.gpu_bundle_adjustment}
                onChange={() => setParams({ baUseGpu: !params.baUseGpu })} /> {t('f_baUseGpu')}
            </label>
            <div className="hint">{t('hint_baUseGpu')}</div>
          </div>
          {!doctor?.checks.colmap?.capabilities?.gpu_bundle_adjustment && (
            <div className="hint" style={{ color: '#d69a2a' }}>
              {t('ceresCpuUnavailable')}
            </div>
          )}
          <div className="ctl">
            <div style={{ cursor: 'pointer', userSelect: 'none', color: 'var(--fg-mute)', fontSize: 12 }}
              onClick={() => setColmapAdvOpen(value => !value)}>{colmapAdvOpen ? '▾' : '▸'} {t('lbl_colmapAdvanced')}</div>
            {colmapAdvOpen && <div style={{ marginTop: 6 }}>
              <NumField label={t('f_mapperMinMatches')} value={params.mapperMinNumMatches} onChange={value => setParams({ mapperMinNumMatches: value })} />
              <NumField label="random_seed" value={params.mapperRandomSeed} onChange={value => setParams({ mapperRandomSeed: value })} />
              {params.mapper === 'incremental' && <>
                <NumField label={t('f_initMinInliers')} value={params.initMinNumInliers} onChange={value => setParams({ initMinNumInliers: value })} />
                <NumField label="init_image_id1" value={params.initImageId1} onChange={value => setParams({ initImageId1: value })} />
                <NumField label="init_image_id2" value={params.initImageId2} onChange={value => setParams({ initImageId2: value })} />
                <NumField label={t('f_absPoseMaxError')} value={params.absPoseMaxError} step={0.5} onChange={value => setParams({ absPoseMaxError: value })} />
                <NumField label={t('f_filterMaxReproj')} value={params.filterMaxReprojError} step={0.5} onChange={value => setParams({ filterMaxReprojError: value })} />
                <NumField label={t('f_filterMinTriAngle')} value={params.filterMinTriAngle} step={0.5} onChange={value => setParams({ filterMinTriAngle: value })} />
                <NumField label={t('f_baLocalIters')} value={params.baLocalIters} onChange={value => setParams({ baLocalIters: value })} />
                <NumField label={t('f_minModelSize')} value={params.minModelSize} onChange={value => setParams({ minModelSize: value })} />
              </>}
              <NumField label={t('f_baGlobalIters')} value={params.baGlobalIters} onChange={value => setParams({ baGlobalIters: value })} />
              <NumField label="min_points3D" value={params.minPoints3D} onChange={value => setParams({ minPoints3D: value })} />
            </div>}
          </div>
        </>
      )}

      {stage === 'align_reconstruction' && (
        <div className="ctl">
          <label>{t('lbl_alignment')}</label>
          <select className="input" value={params.alignmentMethod}
            onChange={e => setParams({ alignmentMethod: e.target.value as StageParams['alignmentMethod'] })}>
            <option value="auto">{t('alignmentAuto')}</option>
            <option value="imu">{t('alignmentRequired')}</option>
            <option value="none">{t('disabledOption')}</option>
          </select>
          <div className="hint">{t('hint_alignment')}</div>
        </div>
      )}

      {stage === 'restore_metric_scale' && (
        <div className="ctl">
          <label>{t('lbl_metricScale')}</label>
          <select className="input" value={params.metricScaleMethod}
            onChange={event => setParams({
              metricScaleMethod: event.target.value as StageParams['metricScaleMethod'],
            })}>
            <option value="auto">{t('metricScaleAuto')}</option>
            <option value="rig">{t('metricScaleRequired')}</option>
            <option value="none">{t('disabledOption')}</option>
          </select>
          <div className="hint">{t('hint_metricScale')}</div>
        </div>
      )}

      {stage === 'position_ground' && (
        <div className="ctl">
          <label>{t('lbl_groundPosition')}</label>
          <select className="input" value={params.groundPositionMethod}
            onChange={event => setParams({
              groundPositionMethod: event.target.value as StageParams['groundPositionMethod'],
            })}>
            <option value="auto">{t('groundPositionAuto')}</option>
            <option value="points">{t('groundPositionRequired')}</option>
            <option value="none">{t('disabledOption')}</option>
          </select>
          <div className="hint">{t('hint_groundPosition')}</div>
        </div>
      )}

      {stage === 'dense_initialization' && (
        <>
          <label className="ctl" style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
            <input type="checkbox" checked={params.denseEnabled}
              onChange={event => setParams({ denseEnabled: event.target.checked })} />
            {t('denseEnable')}
          </label>
          <div className="hint" style={{ marginBottom: 8 }}>{t('denseHint')}</div>
          <div className="ctl">
            <label>{t('denseQuality')}</label>
            <select className="input" value={params.denseQuality}
              disabled={!params.denseEnabled}
              onChange={event => setParams({ denseQuality: event.target.value as StageParams['denseQuality'] })}>
              <option value="turbo">Turbo (320px)</option>
              <option value="fast">Fast (512px)</option>
              <option value="base">Base (640px)</option>
              <option value="high">High (640/960px)</option>
            </select>
          </div>
          <Slider label={t('denseReferences')} hint={t('denseReferencesHint')} min={0.05} max={1} step={0.05}
            value={params.denseReferenceFraction} onChange={value => setParams({ denseReferenceFraction: value })}
            fmt={value => `${Math.round(value * 100)}%`} />
          <Slider label={t('denseNeighbors')} hint={t('denseNeighborsHint')} min={1} max={6} step={1}
            value={params.denseNeighbors} onChange={value => setParams({ denseNeighbors: Math.round(value) })}
            fmt={value => String(Math.round(value))} />
          <NumField label={t('denseMatches')} value={params.denseMatchesPerPair}
            onChange={value => setParams({ denseMatchesPerPair: value })} />
          <NumField label={t('denseMaximumPoints')} value={params.denseMaximumPoints}
            onChange={value => setParams({ denseMaximumPoints: value })} />
          <label className="ctl" style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
            <input type="checkbox" checked={params.denseUseFeatureMasks} disabled={!params.denseEnabled}
              onChange={event => setParams({ denseUseFeatureMasks: event.target.checked })} />
            {t('denseUseMasks')}
          </label>
          <div className="inspector-section-title">{t('advanced')}</div>
          <NumField label={t('denseConfidence')} value={params.denseConfidenceThreshold} step={0.05}
            onChange={value => setParams({ denseConfidenceThreshold: value })} />
          <NumField label={t('denseReprojection')} value={params.denseReprojectionThreshold} step={0.1}
            onChange={value => setParams({ denseReprojectionThreshold: value })} />
          <NumField label={t('denseParallax')} value={params.denseMinimumParallax} step={0.1}
            onChange={value => setParams({ denseMinimumParallax: value })} />
          <NumField label={t('denseVoxelRatio')} value={params.denseVoxelRatio} step={0.0001}
            onChange={value => setParams({ denseVoxelRatio: value })} />
        </>
      )}

      {stage === 'prepare_images' && reconMode === 'pinhole_rig' && (
        <Slider label={t('lblPinholeSize')} hint={t('hintPinholeSize')} min={512} max={2048} step={128}
          value={params.size} onChange={value => setParams({ size: Math.round(value) })} fmt={value => `${value}px`} />
      )}

      {stage === 'inspect_source' && (
        <>
          <div className="ctl">
            <label>{t('addSource')}</label>
            <select className="input" value={sourcePreset}
              onChange={event => setSourcePreset(event.target.value as SourcePreset)}>
              <option value="insv">{t('sourceTypeInsv')}</option>
              <option value="perspective_images">{t('sourceTypePhoneImages')}</option>
              <option value="perspective_video">{t('sourceTypePerspectiveVideo')}</option>
              <option value="erp_video">{t('sourceTypeErpVideo')}</option>
              <option value="erp_images">{t('sourceTypeErpImages')}</option>
            </select>
            <button className="btn" style={{ marginTop: 6 }} disabled={processing} onClick={() => onSelectSource({
              ...SOURCE_PRESETS[sourcePreset],
              role: sources.length ? 'supplemental' : 'primary',
            })}>{t('choosePath')}</button>
            <div className="hint">{t('hint_addSource')}</div>
          </div>
          <div className="source-list">
            {sources.map(source => (
              <div className="source-card" key={source.id}>
                <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                  <strong style={{ flex: 1 }}>{source.label}</strong>
                  <span className={`source-role ${source.role}`}>{source.role === 'primary'
                    ? t('primarySource') : t('supplementalSource')}</span>
                </div>
                <div className="hint">{t(`projection_${source.projection}`)} · {t(`media_${source.media_kind}`)}</div>
                <PathText path={source.path} compact />
                <div style={{ display: 'flex', gap: 6, marginTop: 6 }}>
                  {source.role !== 'primary' && (
                    <button className="btn btn-secondary" disabled={processing}
                      onClick={() => onMakePrimarySource(source.id)}>
                      {t('makePrimary')}
                    </button>
                  )}
                  <button className="btn btn-secondary" style={{ color: 'var(--error)' }} disabled={processing}
                    onClick={() => {
                      if (window.confirm(t('removeSourceConfirm'))) onDeleteSource(source.id)
                    }}>{t('removeSource')}</button>
                </div>
              </div>
            ))}
            {!sources.length && <div className="hint">{t('noSource')}</div>}
          </div>
          {sourceMutationError != null && <div className="error">{String(sourceMutationError)}</div>}
          <div style={{ marginTop: 12 }}>
            <label>{t('lbl_mode')}</label>
            <select className="input" value={reconMode} onChange={e => setReconMode(e.target.value as ReconMode)}>
              {primaryProjection === 'equirectangular'
                ? (<>
                    <option value="equirectangular">{t('modeEquirect')}</option>
                    <option value="pinhole_rig">{t('modePinhole')}</option>
                  </>)
                : (<>
                    <option value="native_fisheye">{t('modeNative')}</option>
                    <option value="pinhole_rig">{t('modePinhole')}</option>
                  </>)}
            </select>
            <div className="hint">{t('hint_mode')}</div>
            {reconMode === 'pinhole_rig' && <div className="hint">{t('hint_needsGen')}</div>}
            {(reconMode === 'native_fisheye' || reconMode === 'equirectangular') && <div className="hint">{t('hint_noGen')}</div>}
            <div className="mono" style={{ fontSize: 11, marginTop: 6 }}>
              {t('resultMode')}: {resultMode === 'native' ? t('modeNative')
                : resultMode === 'equirect' ? t('modeEquirect')
                : resultMode === 'pinhole' ? t('modePinhole')
                : resultMode === 'mixed' ? t('modeMixed') : t('modeNone')}
            </div>
          </div>
          {sourceInfo?.duration_sec_total != null && (
            <div className="mono" style={{ fontSize: 11, marginTop: 8 }}>
              {t('totalVideoDuration')}: {sourceInfo.duration_sec_total.toFixed(1)}s
            </div>
          )}
        </>
      )}
      {stage === 'export_dataset' && (
        <>
          <div className="ctl">
            <label style={{ display: 'inline-flex', gap: 6, alignItems: 'center' }}>
              <input type="checkbox" checked={params.emitTrainConfigs}
                onChange={() => setParams({ emitTrainConfigs: !params.emitTrainConfigs })} /> {t('lbl_emitTrainConfigs')}
            </label>
            <div className="hint">{t('hint_emitTrainConfigs')}</div>
          </div>
          <div className="ctl">
            <label style={{ display: 'inline-flex', gap: 6, alignItems: 'center' }}>
              <input type="checkbox" checked={params.optimizeFisheyeTrainingImages}
                onChange={() => setParams({
                  optimizeFisheyeTrainingImages: !params.optimizeFisheyeTrainingImages,
                })} /> {t('lbl_optimizeFisheyeTrainingImages')}
            </label>
            <div className="hint">{t('hint_optimizeFisheyeTrainingImages')}</div>
          </div>
          {exportInfo
            ? <div className="ctl">
                <label>{t('exportDir')}</label>
                <PathText path={exportInfo.dir} />
                {exportInfo.gui_integration && !exportInfo.gui_integration.train_configs_auto_applied && (
                  <div className="hint" style={{ marginTop: 10, color: '#d69a2a' }}>
                    {t('lfGuiConfigWarning')}
                    <div className="mono" style={{ marginTop: 4 }}>
                      {t('lfRequiredSettings')}: {exportInfo.gui_integration.required_settings.strategy.toUpperCase()}
                      {' · '}GUT={String(exportInfo.gui_integration.required_settings.gut)}
                      {' · '}mask={exportInfo.gui_integration.required_settings.mask_mode}
                      {' · '}PPISP={String(exportInfo.gui_integration.required_settings.ppisp)}
                      {' · '}controller={String(exportInfo.gui_integration.required_settings.ppisp_controller)}
                      {exportInfo.gui_integration.required_settings.max_width > 0
                        ? ` · max-width=${exportInfo.gui_integration.required_settings.max_width}` : ''}
                    </div>
                  </div>
                )}
              </div>
            : <div className="hint">{t('noParams')}</div>}
        </>
      )}
    </div>
  )
}

// COLMAP 詳細用の数値入力 (0 = COLMAP 既定). placeholder で「既定」を示す.
const NumField = ({ label, hint, value, step = 1, onChange }: {
  label: string; hint?: string; value: number; step?: number; onChange: (v: number) => void
}) => {
  const { t } = useSettings()
  const inputId = useId()
  return (
    <div className="ctl" style={{ marginBottom: 6 }}>
      <label htmlFor={inputId} style={{ fontSize: 11 }}>{label}</label>
      <input id={inputId} className="input" type="number" min={0} step={step} value={value || ''}
        placeholder={t('defaultZero')} onChange={e => onChange(Number(e.target.value) || 0)} />
      {hint && <div className="hint">{hint}</div>}
    </div>
  )
}

const StageResult = ({ stage, extra }: { stage: string; extra: Record<string, unknown> }) => {
  const { t } = useSettings()
  const [expanded, setExpanded] = useState(false)
  const statisticsId = useId()
  const rows: Array<[string, string, string]> = []
  const visit = (value: unknown, path: string[]) => {
    if (value == null) return
    if (Array.isArray(value)) {
      const simple = value.every(item => ['string', 'number', 'boolean'].includes(typeof item))
      if (simple) {
        rows.push([path.join('.'), statPathLabel(t, path), value.length ? value.join(', ') : '0'])
        return
      }
      value.forEach((item, index) => visit(item, [...path, `#${index + 1}`]))
      return
    }
    if (typeof value === 'object') {
      for (const [key, child] of Object.entries(value as Record<string, unknown>)) visit(child, [...path, key])
      return
    }
    if (typeof value !== 'string' && typeof value !== 'number' && typeof value !== 'boolean') return
    const leaf = path.at(-1) ?? ''
    rows.push([path.join('.'), statPathLabel(t, path), formatStatistic(t, leaf, value)])
  }
  visit(extra, [stage])
  if (!rows.length) return null
  return <div className="stage-statistics">
    <button type="button" className="stage-statistics-toggle" aria-expanded={expanded}
      aria-controls={statisticsId} onClick={() => setExpanded(value => !value)}>
      <span aria-hidden="true">{expanded ? '▾' : '▸'}</span> {t('resultStatistics')}
    </button>
    {expanded && <dl id={statisticsId} className="inspector-fields">
        {rows.map(([path, label, value]) => (
          <div className="inspector-field" key={path}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>}
  </div>
}

const statPathLabel = (t: (key: string) => string, path: string[]): string => {
  const contexts = path.slice(1, -1).flatMap((segment, index, segments) => {
    if (!segment.startsWith('#') || index === 0) return []
    return [`${statLabel(t, segments[index - 1])} ${segment}`]
  })
  return [...contexts, statLabel(t, path.at(-1) ?? '')].join(' · ')
}

const statLabel = (t: (key: string) => string, key: string): string => {
  const translationKey = `stat_${key}`
  const translated = t(translationKey)
  return translated === translationKey
    ? key.replaceAll('_', ' ').replace(/\b\w/g, char => char.toUpperCase())
    : translated
}

const formatStatistic = (t: (key: string) => string, key: string, value: string | number | boolean): string => {
  if (key === 'lfstudio_training_metrics' && value === 'external') return t('notAvailableExternalTraining')
  if (key === 'rig_verification_scope' && typeof value === 'string') {
    const translationKey = `stat_${value}`
    const translated = t(translationKey)
    return translated === translationKey ? value : translated
  }
  if ((key === 'purpose' || key === 'mask_source') && typeof value === 'string') {
    if (value === 'feature') return t('featureMask')
    if (value === 'training') return t('trainingMask')
    if (value === 'physical') return t('physicalMask')
  }
  if (typeof value === 'boolean') return value ? t('yes') : t('no')
  if (typeof value === 'string') return value
  if (key === 'file_size' || key === 'footer_size' || key.endsWith('_bytes')) {
    const units = ['B', 'KiB', 'MiB', 'GiB']
    let amount = value, index = 0
    while (amount >= 1024 && index < units.length - 1) { amount /= 1024; index += 1 }
    return `${amount.toFixed(index ? 2 : 0)} ${units[index]}`
  }
  if (key === 'maximum_to_p95_ratio' || key === 'trajectory_maximum_to_p95_ratio'
    || key === 'max_step_ratio_limit')
    return `${Number(value.toPrecision(6)).toLocaleString()}×`
  if (key.endsWith('_ratio') || key.endsWith('_coverage')) return `${(value * 100).toFixed(2)}%`
  if (key === 'relative_mad') return `${(value * 100).toFixed(4)}%`
  if (key.endsWith('_m')) return `${value.toFixed(3)} m`
  if (key === 'scale_factor') return `${Number(value.toPrecision(8)).toLocaleString()}×`
  if (key.includes('reprojection_error')) return `${value.toFixed(4)} px`
  if (key.endsWith('_px')) return `${value.toFixed(3)} px`
  if (key.endsWith('_deg')) return `${value.toFixed(3)}°`
  if (key.endsWith('_sec')) return `${value.toFixed(3)} s`
  if (Number.isInteger(value)) return value.toLocaleString()
  return Number(value.toPrecision(6)).toLocaleString()
}

const Slider = ({
  label, hint, min, max, step, value, onChange, fmt,
}: {
  label: string; hint?: string; min: number; max: number; step: number; value: number
  onChange: (v: number) => void; fmt: (v: number) => string
}) => {
  const inputId = useId()
  return (
    <div className="ctl">
      <label htmlFor={inputId}>{label}</label>
      <div className="row">
        <input id={inputId} type="range" min={min} max={max} step={step} value={value}
          onChange={e => onChange(Number(e.target.value))} />
        <span className="val">{fmt(value)}</span>
      </div>
      {hint && <div className="hint">{hint}</div>}
    </div>
  )
}
