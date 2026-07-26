import { useEffect, useId, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api, type FrameSelection, type SourceInfo, type StageStatus } from '../api/client'
import { paramsForStage, QUALITY_PRESETS, type ReconMode, type StageParams } from './stageParams'
import { ProgressRing } from '../components/ProgressRing'
import { PathText } from '../components/PathText'
import { useSettings } from '../ui/settings'

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
  sourcePath, resultMode, sourceKind, processing, stageIsRunning, onStop, stageDisabled, onToggleStage, onSelectSource, frameSelection,
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
  sourcePath: string | null
  resultMode: 'native' | 'pinhole' | 'equirect' | null
  sourceKind: string | null
  processing: boolean
  stageIsRunning: boolean
  onStop: () => void
  stageDisabled: boolean
  onToggleStage: () => void
  onSelectSource: () => void
  frameSelection: FrameSelection | null | undefined
  stageProgress: number
  stageStartedAt: string | null
  stageProgressMsg: string
  blockedReason: string | null
}) => {
  const { t } = useSettings()
  const qc = useQueryClient()
  const { data: doctor } = useQuery({ queryKey: ['doctor'], queryFn: api.getDoctor, staleTime: 30_000 })
  const [advOpen, setAdvOpen] = useState(false)
  const [colmapAdvOpen, setColmapAdvOpen] = useState(false)
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
      for (const key of ['reconstruction', 'frames', 'fisheye-region', 'masks', 'denoise', 'export-info']) {
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
  const stageConfigDisabled = stage === 'denoise_frames'
    && (params.denoiseMethod === 'off' || reconMode === 'pinhole_rig' || sourceKind === 'erp_images')
  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
        <strong style={{ flex: 1 }}>{t(`st_${stage}`)}</strong>
        {processing && stageIsRunning
          ? <button className="btn stop" onClick={onStop}>■ {t('stop')}</button>
          : <button className="btn" disabled={run.isPending || !hasSource || processing || stageConfigDisabled || !!blockedReason}
              title={!hasSource ? t('noSource') : processing ? t('otherRunning')
                : stageConfigDisabled ? (reconMode === 'pinhole_rig' ? t('denoisePinholeUnsupported')
                  : sourceKind === 'erp_images' ? t('denoiseVideoOnly') : t('denoiseEnableFirst'))
                : blockedReason ?? ''}
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
              <ProgressRing value={stageProgress} size={24} stroke={3} showLabel />
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
      {status?.has_output && status.extra && <StageResult stage={stage} extra={status.extra} />}

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
                  {t('selReasons')}: 🌫 {frameSelection.reasons.blur} · ☀ {frameSelection.reasons.exposure} · ✦ {frameSelection.reasons.few_features}
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

      {stage === 'generate_masks' && (
        <>
          <div className="ctl">
            <label style={{ display: 'inline-flex', gap: 6, alignItems: 'center' }}>
              <input type="checkbox" checked={!stageDisabled} onChange={onToggleStage} /> {t('enableStage')}
            </label>
            <div className="hint">{t('hint_sam3enable')}</div>
            {reconMode === 'native_fisheye' && <div className="hint">{t('hint_sam3circle')}</div>}
          </div>
          <div className="ctl">
            <label style={{ display: 'inline-flex', gap: 6, alignItems: 'center' }}>
              <input type="checkbox" checked={params.downsampleOn} onChange={() => setParams({ downsampleOn: !params.downsampleOn })} /> {t('lbl_downsample')}
            </label>
            <div className="hint">{t('hint_downsample')}</div>
          </div>
          {params.downsampleOn && (
            <Slider label={t('lbl_masksize')} hint={t('hint_masksize')} min={6} max={14} step={1}
              value={Math.round(Math.log2(params.maskSize))}
              onChange={v => setParams({ maskSize: 2 ** v })} fmt={v => `${2 ** v}px`} />
          )}
          <div className="ctl">
            <label style={{ display: 'inline-flex', gap: 6, alignItems: 'center' }}>
              <input type="checkbox" checked={params.dilateOn} onChange={() => setParams({ dilateOn: !params.dilateOn })} /> {t('lbl_dilate')}
            </label>
            <div className="hint">{t('hint_dilate')}</div>
          </div>
          {params.dilateOn && (
            <Slider label={t('lbl_dilate')} hint="" min={2} max={40} step={2}
              value={params.dilate} onChange={v => setParams({ dilate: Math.round(v) })} fmt={v => `${v}px`} />
          )}
          <div className="ctl">
            <label>{t('lbl_prompt')}</label>
            <input className="input" placeholder="person,tripod,..." value={params.prompt}
              onChange={e => setParams({ prompt: e.target.value })} />
            <div className="hint">{t('hint_prompt')}</div>
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
              onChange={e => setParams({ featureType: e.target.value as StageParams['featureType'], qualityPreset: 'custom' })}>
              <option value="ALIKED_N16ROT">ALIKED N16ROT</option>
              <option value="ALIKED_N32">ALIKED N32</option>
              <option value="SIFT">SIFT</option>
            </select>
            <div className="hint">{t('hint_backend')}</div>
          </div>
          <NumField label={t('f_maxImageSize')} value={params.featureMaxImageSize}
            onChange={value => setParams({ featureMaxImageSize: value, qualityPreset: 'custom' })} />
          <NumField label={t('f_maxFeatures')} value={params.featureMaxNumFeatures}
            onChange={value => setParams({ featureMaxNumFeatures: value, qualityPreset: 'custom' })} />
          <label style={{ display: 'inline-flex', gap: 6, alignItems: 'center', margin: '4px 0' }}>
            <input type="checkbox" checked={params.featureUseGpu}
              onChange={() => setParams({ featureUseGpu: !params.featureUseGpu })} /> GPU
          </label>
          {params.featureType === 'SIFT' && <>
            <NumField label={t('f_peakThreshold')} value={params.siftPeakThreshold} step={0.0001}
              onChange={value => setParams({ siftPeakThreshold: value, qualityPreset: 'custom' })} />
            <NumField label={t('f_edgeThreshold')} value={params.siftEdgeThreshold} step={0.5}
              onChange={value => setParams({ siftEdgeThreshold: value, qualityPreset: 'custom' })} />
            <label style={{ display: 'inline-flex', gap: 6, alignItems: 'center', margin: '4px 0' }}>
              <input type="checkbox" checked={params.siftAffineDsp}
                onChange={() => setParams({ siftAffineDsp: !params.siftAffineDsp, qualityPreset: 'custom' })} /> {t('f_affineDsp')}
            </label>
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
          </div>
          <div className="ctl">
            <label>{t('lbl_pairing')}</label>
            <select className="input" value={params.pairing}
              onChange={e => setParams({ pairing: e.target.value as StageParams['pairing'] })}>
              <option value="sequential">{t('matcher_sequential')}</option>
              <option value="exhaustive">{t('matcher_exhaustive')}</option>
              <option value="vocab_tree">{t('matcher_vocab')}</option>
            </select>
          </div>
          {params.pairing === 'sequential' && <>
            <Slider label={t('lbl_overlap')} hint={t('hint_overlap')} min={2} max={20} step={1}
              value={params.overlap} onChange={value => setParams({ overlap: Math.round(value) })} fmt={value => `${value}`} />
            <label style={{ display: 'inline-flex', gap: 6, alignItems: 'center' }}>
              <input type="checkbox" checked={params.loopClosure}
                onChange={() => setParams({ loopClosure: !params.loopClosure })} /> {t('lbl_loopClosure')}
            </label>
          </>}
          <NumField label={t('f_maxMatches')} value={params.maxNumMatches}
            onChange={value => setParams({ maxNumMatches: value, qualityPreset: 'custom' })} />
          <NumField label={t('f_twoViewInliers')} value={params.twoViewMinInliers}
            onChange={value => setParams({ twoViewMinInliers: value })} />
          <label style={{ display: 'inline-flex', gap: 6, alignItems: 'center' }}>
            <input type="checkbox" checked={params.guidedMatching}
              onChange={() => setParams({ guidedMatching: !params.guidedMatching })} /> {t('f_guidedMatching')}
          </label>
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
              <option value="global">Global Mapper</option>
              <option value="incremental">Incremental Mapper</option>
            </select>
          </div>
          {params.mapper === 'global' && <label style={{ display: 'inline-flex', gap: 6, alignItems: 'center' }}>
              <input type="checkbox" checked={params.viewGraphCalibration}
              onChange={() => setParams({ viewGraphCalibration: !params.viewGraphCalibration })} /> {t('viewGraphCalibration')}
          </label>}
          <label style={{ display: 'inline-flex', gap: 6, alignItems: 'center' }}>
            <input type="checkbox" checked={params.baUseGpu}
              disabled={!doctor?.checks.colmap.capabilities?.gpu_bundle_adjustment}
              onChange={() => setParams({ baUseGpu: !params.baUseGpu })} /> {t('f_baUseGpu')}
          </label>
          {!doctor?.checks.colmap.capabilities?.gpu_bundle_adjustment && (
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

      {stage === 'denoise_frames' && (
        <>
          <div className="ctl">
            <label htmlFor="denoise-method">{t('lbl_denoiseMethod')}</label>
            <select id="denoise-method" className="input" value={params.denoiseMethod}
              onChange={e => setParams({ denoiseMethod: e.target.value as StageParams['denoiseMethod'] })}>
              <option value="off">{t('denoiseOff')}</option>
              <option value="fastdvdnet">{t('denoiseFastDvdnet')}</option>
              <option value="ffmpeg_adaptive">{t('denoiseFfmpeg')}</option>
            </select>
            <div className="hint">{t('hint_denoiseMethod')}</div>
          </div>
          {sourceKind === 'erp_images' && params.denoiseMethod !== 'off' && (
            <div className="hint" style={{ color: '#d69a2a' }}>{t('denoiseVideoOnly')}</div>
          )}
          {reconMode === 'pinhole_rig' && params.denoiseMethod !== 'off' && (
            <div className="hint" style={{ color: '#d69a2a' }}>{t('denoisePinholeUnsupported')}</div>
          )}
          {params.denoiseMethod === 'fastdvdnet' && (
            <>
              <Slider label={t('lbl_denoiseSigma')} hint={t('hint_denoiseSigma')}
                min={5} max={20} step={1} value={params.denoiseSigma}
                onChange={value => setParams({ denoiseSigma: Math.round(value) })} fmt={value => `${value}`} />
              <div className="ctl">
                <label htmlFor="denoise-tile">{t('lbl_denoiseTile')}</label>
                <select id="denoise-tile" className="input" value={params.denoiseTileSize}
                  onChange={e => setParams({ denoiseTileSize: Number(e.target.value) })}>
                  <option value={256}>256 px</option>
                  <option value={512}>512 px</option>
                  <option value={768}>768 px</option>
                </select>
                <div className="hint">{t('hint_denoiseTile')}</div>
              </div>
            </>
          )}
          {params.denoiseMethod === 'ffmpeg_adaptive' && (
            <>
              <div className="ctl">
                <label htmlFor="denoise-window">{t('lbl_denoiseWindow')}</label>
                <select id="denoise-window" className="input" value={params.denoiseTemporalWindow}
                  onChange={e => setParams({ denoiseTemporalWindow: Number(e.target.value) })}>
                  <option value={5}>5</option><option value={9}>9</option><option value={13}>13</option>
                </select>
                <div className="hint">{t('hint_denoiseWindow')}</div>
              </div>
              <label style={{ display: 'inline-flex', gap: 6, alignItems: 'center' }}>
                <input type="checkbox" checked={params.denoiseLumaOnly}
                  onChange={() => setParams({ denoiseLumaOnly: !params.denoiseLumaOnly })} /> {t('lbl_denoiseLuma')}
              </label>
            </>
          )}
          <div className="predict">{params.denoiseMethod === 'off' || reconMode === 'pinhole_rig'
            ? t('denoiseExportOriginal') : t('denoiseExportProcessed')}</div>
        </>
      )}

      {stage === 'reproject_views' && (
        <Slider label={t('lblPinholeSize')} hint="" min={512} max={2048} step={128}
          value={params.size} onChange={v => setParams({ size: Math.round(v) })} fmt={v => `${v}px`} />
      )}

      {stage === 'inspect_source' && (
        <div className="ctl">
          <button className="btn" onClick={onSelectSource}>{t('selectSource')}</button>
          <div className="hint">{t('hint_selectsource')}</div>
          <div style={{ marginTop: 8 }}>
            <div className="hint">{t('source')}</div>
            {sourcePath
              ? <PathText path={sourcePath} />
              : <div className="mono">{t('noSource')}</div>}
            {sourceInfo?.duration_sec != null && (
              <div className="mono" style={{ fontSize: 11 }}>
                {sourceInfo.width}×{sourceInfo.height} · {sourceInfo.duration_sec.toFixed(0)}s
                {sourceInfo.fps ? ` · ${sourceInfo.fps.toFixed(2)}fps` : ''}
              </div>
            )}
          </div>
          <div style={{ marginTop: 12 }}>
            <label>{t('lbl_mode')}</label>
            <select className="input" value={reconMode} onChange={e => setReconMode(e.target.value as ReconMode)}>
              {sourceKind === 'erp_video' || sourceKind === 'erp_images'
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
                : resultMode === 'pinhole' ? t('modePinhole') : t('modeNone')}
            </div>
          </div>
        </div>
      )}
      {stage === 'export_dataset' && (
        <>
          <div className="ctl">
            <label style={{ display: 'inline-flex', gap: 6, alignItems: 'center' }}>
              <input type="checkbox" checked={params.emitTrainConfigs}
                onChange={() => setParams({ emitTrainConfigs: !params.emitTrainConfigs })} /> {t('lbl_emitTrainConfigs')}
            </label>
            <div className="hint">{t('hint_emitTrainConfigs')}</div>
            <div className="hint">{params.denoiseMethod === 'off' || reconMode === 'pinhole_rig'
              ? t('denoiseExportOriginal') : t('denoiseExportProcessed')}</div>
          </div>
          {exportInfo
            ? <div className="ctl">
                <label>{t('exportDir')}</label>
                <PathText path={exportInfo.dir} />
                {exportInfo.dataset_dir && (
                  <div style={{ marginTop: 6 }}>
                    <div className="hint">{t('exportDataset')}</div>
                    <PathText path={exportInfo.dataset_dir} />
                  </div>
                )}
                {exportInfo.preview_dir && (
                  <div style={{ marginTop: 6 }}>
                    <div className="hint">{t('exportPreview')}</div>
                    <PathText path={exportInfo.preview_dir} />
                  </div>
                )}
                {exportInfo.train_configs_dir && (
                  <div style={{ marginTop: 6 }}>
                    <div className="hint">{t('exportTrainConfigs')}</div>
                    <PathText path={exportInfo.train_configs_dir} />
                  </div>
                )}
                <div style={{ marginTop: 6 }}>
                  <div className="hint">{t('exportTrainingOutput')}</div>
                  <PathText path={exportInfo.training_output_dir} />
                </div>
                {exportInfo.gui_integration && !exportInfo.gui_integration.train_configs_auto_applied && (
                  <div className="hint" style={{ marginTop: 10, color: '#d69a2a' }}>
                    {t('lfGuiConfigWarning')}
                    <div className="mono" style={{ marginTop: 4 }}>
                      {t('lfRequiredSettings')}: {exportInfo.gui_integration.required_settings.strategy.toUpperCase()}
                      {' · '}GUT={String(exportInfo.gui_integration.required_settings.gut)}
                      {' · '}mask={exportInfo.gui_integration.required_settings.mask_mode}
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
const NumField = ({ label, value, step = 1, onChange }: {
  label: string; value: number; step?: number; onChange: (v: number) => void
}) => {
  const { t } = useSettings()
  const inputId = useId()
  return (
    <div className="ctl" style={{ marginBottom: 6 }}>
      <label htmlFor={inputId} style={{ fontSize: 11 }}>{label}</label>
      <input id={inputId} className="input" type="number" min={0} step={step} value={value || ''}
        placeholder={t('defaultZero')} onChange={e => onChange(Number(e.target.value) || 0)} />
    </div>
  )
}

const StageResult = ({ stage, extra }: { stage: string; extra: Record<string, unknown> }) => {
  const { t } = useSettings()
  const rows: Array<[string, unknown]> = stage === 'extract_features'
    ? [[t('resultImages'), extra.images], [t('resultAvgKeypoints'), Math.round(Number(extra.average_keypoints ?? 0))]]
    : stage === 'match_features'
    ? [[t('resultVerifiedPairs'), extra.verified_pairs], [t('resultAvgInliers'), Number(extra.average_inliers ?? 0).toFixed(1)]]
    : stage === 'reconstruct'
    ? [[t('resultRegistered'), `${extra.num_images ?? 0}`], [t('resultPoints'), extra.num_points3D], [t('resultError'), `${Number(extra.mean_reprojection_error ?? 0).toFixed(3)} px`]]
    : stage === 'align_reconstruction'
    ? [[t('resultApplied'), String(extra.applied)], [t('resultSpread'), `${Number(extra.spread_deg ?? 0).toFixed(2)}°`], [t('resultInliers'), extra.inlier_count], [t('resultTimeOffset'), `${Number(extra.time_offset_sec ?? 0).toFixed(3)} s`]]
    : stage === 'denoise_frames'
    ? [[t('resultMethod'), extra.method], [t('resultImages'), extra.images], [t('resultDevice'), extra.device ?? '—'],
      [t('resultSigma'), extra.sigma ?? '—'], [t('resultMeanDelta'), extra.mean_abs_delta ?? '—']]
    : []
  if (!rows.length) return null
  return <div className="predict" style={{ marginBottom: 10 }}>
    {rows.map(([label, value]) => <div key={label}><span className="hint">{label}: </span>{String(value ?? '—')}</div>)}
  </div>
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
