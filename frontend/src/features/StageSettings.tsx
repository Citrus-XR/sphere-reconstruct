import { useEffect, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api, type FrameSelection, type SourceInfo, type StageStatus } from '../api/client'
import { paramsForStage, COLMAP_PRESETS, type ReconMode, type StageParams } from './stageParams'
import { ProgressRing } from '../components/ProgressRing'
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
  stageProgress, stageStartedAt, stageProgressMsg,
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
}) => {
  const { t } = useSettings()
  const qc = useQueryClient()
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
  // 源の検査ボタンを隠す (緑扱い) のは「ERP ソース かつ equirectangular モード」だけ.
  // ERP+pinhole は reproject という処理が要るので, 源ステップも通常の生成ボタンを出す.
  const hideRun = stage === 'inspect_source' && sourceKind !== 'insv' && reconMode === 'equirectangular'

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
        <strong style={{ flex: 1 }}>{t(`st_${stage}`)}</strong>
        {!hideRun && (processing && stageIsRunning
          ? <button className="btn stop" onClick={onStop}>■ {t('stop')}</button>
          : <button className="btn" disabled={run.isPending || !hasSource || processing}
              title={!hasSource ? t('noSource') : processing ? t('otherRunning') : ''}
              onClick={() => run.mutate()}>
              {status?.has_output ? t('regenerate') : t('generate')}
            </button>)}
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
      {run.error && <div className="error">{String(run.error)}</div>}
      {clear.error && <div className="error">{String(clear.error)}</div>}

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

      {stage === 'reconstruct' && (
        <>
          <div className="hint" style={{ marginBottom: 8 }}>
            {t('lbl_mode')}: {reconMode === 'native_fisheye' ? t('modeNative')
              : reconMode === 'equirectangular' ? t('modeEquirect') : t('modePinhole')} ({t('st_inspect_source')})
          </div>
          <div className="ctl">
            <label>{t('lbl_backend')}</label>
            <select className="input" value={params.backend} onChange={e => setParams({ backend: e.target.value as StageParams['backend'] })}>
              <option value="sift">SIFT</option><option value="aliked">ALIKED</option>
            </select>
            <div className="hint">{t('hint_backend')}</div>
          </div>
          {params.backend === 'sift' && (
            <>
              <div className="ctl">
                <label>{t('lbl_matcher')}</label>
                <select className="input" value={params.matcher}
                  onChange={e => setParams({ matcher: e.target.value as StageParams['matcher'] })}>
                  <option value="sequential">{t('matcher_sequential')}</option>
                  <option value="exhaustive">{t('matcher_exhaustive')}</option>
                  <option value="vocab_tree">{t('matcher_vocab')}</option>
                </select>
                <div className="hint">{t('hint_matcher')}</div>
              </div>
              {params.matcher === 'sequential' && (
                <div className="ctl">
                  <label style={{ display: 'inline-flex', gap: 6, alignItems: 'center' }}>
                    <input type="checkbox" checked={params.loopClosure}
                      onChange={() => setParams({ loopClosure: !params.loopClosure })} /> {t('lbl_loopClosure')}
                  </label>
                  <div className="hint">{t('hint_loopClosure')}</div>
                </div>
              )}
            </>
          )}
          {params.backend === 'aliked' && (
            <div className="ctl">
              <label>{t('lbl_device')}</label>
              <select className="input" value={params.device} onChange={e => setParams({ device: e.target.value as StageParams['device'] })}>
                <option value="auto">auto</option><option value="cuda">GPU</option><option value="cpu">CPU</option>
              </select>
              <div className="hint">{t('hint_device')}</div>
              {params.device === 'cuda' && <div className="hint" style={{ color: '#d69a2a' }}>{t('oomWarn')}</div>}
            </div>
          )}
          <Slider label={t('lbl_overlap')} hint={t('hint_overlap')} min={2} max={20} step={1}
            value={params.overlap} onChange={v => setParams({ overlap: Math.round(v) })} fmt={v => `${v}`} />
          {params.backend === 'aliked' && (
            <div className="ctl">
              <div style={{ cursor: 'pointer', userSelect: 'none', color: 'var(--fg-mute)', fontSize: 12 }}
                onClick={() => setAdvOpen(v => !v)}>{advOpen ? '▾' : '▸'} {t('lbl_advanced')}</div>
              {advOpen && (
                <div style={{ marginTop: 6 }}>
                  <div className="ctl">
                    <label style={{ display: 'inline-flex', gap: 6, alignItems: 'center' }}>
                      <input type="checkbox" checked={params.extractCapOn}
                        onChange={() => setParams({ extractCapOn: !params.extractCapOn })} /> {t('lbl_extractCap')}
                    </label>
                    <div className="hint">{t('hint_extractCap')}</div>
                  </div>
                  {params.extractCapOn && (
                    <Slider label={t('lbl_extractCap')} hint="" min={6} max={14} step={1}
                      value={Math.round(Math.log2(params.extractMaxSize))}
                      onChange={v => setParams({ extractMaxSize: 2 ** v })} fmt={v => `${2 ** v}px`} />
                  )}
                </div>
              )}
            </div>
          )}

          <div className="ctl">
            <label>{t('lbl_qualityPreset')}</label>
            <select className="input" value={params.colmapPreset}
              onChange={e => {
                const v = e.target.value as StageParams['colmapPreset']
                setParams(v === 'custom' ? { colmapPreset: v } : { colmapPreset: v, ...COLMAP_PRESETS[v] })
              }}>
              <option value="draft">{t('preset_draft')}</option>
              <option value="standard">{t('preset_standard')}</option>
              <option value="high">{t('preset_high')}</option>
              <option value="custom">{t('preset_custom')}</option>
            </select>
            <div className="hint">{t('hint_qualityPreset')}</div>
          </div>

          <div className="ctl">
            <div style={{ cursor: 'pointer', userSelect: 'none', color: 'var(--fg-mute)', fontSize: 12 }}
              onClick={() => setColmapAdvOpen(v => !v)}>{colmapAdvOpen ? '▾' : '▸'} {t('lbl_colmapAdvanced')}</div>
            {colmapAdvOpen && (() => {
              // 詳細フィールドを編集したらプリセットは custom 扱いにする.
              const setC = (patch: Partial<StageParams>) => setParams({ ...patch, colmapPreset: 'custom' })
              return (
                <div style={{ marginTop: 6 }}>
                  <div className="hint" style={{ marginBottom: 6 }}>{t('hint_colmapZeroDefault')}</div>
                  {params.backend === 'sift' && <>
                    <div className="hint" style={{ fontWeight: 600 }}>{t('grp_features')}</div>
                    <NumField label={t('f_maxFeatures')} value={params.siftMaxFeatures} onChange={v => setC({ siftMaxFeatures: v })} />
                    <NumField label={t('f_maxImageSize')} value={params.siftMaxImageSize} onChange={v => setC({ siftMaxImageSize: v })} />
                    <NumField label={t('f_peakThreshold')} value={params.siftPeakThreshold} step={0.0001} onChange={v => setC({ siftPeakThreshold: v })} />
                    <NumField label={t('f_edgeThreshold')} value={params.siftEdgeThreshold} step={0.5} onChange={v => setC({ siftEdgeThreshold: v })} />
                    <label style={{ display: 'inline-flex', gap: 6, alignItems: 'center', margin: '4px 0' }}>
                      <input type="checkbox" checked={params.siftAffineDsp} onChange={() => setC({ siftAffineDsp: !params.siftAffineDsp })} /> {t('f_affineDsp')}
                    </label>
                  </>}
                  <div className="hint" style={{ fontWeight: 600, marginTop: 6 }}>{t('grp_matching')}</div>
                  {params.backend === 'sift' && <>
                    <NumField label={t('f_maxMatches')} value={params.maxNumMatches} onChange={v => setC({ maxNumMatches: v })} />
                    <label style={{ display: 'inline-flex', gap: 6, alignItems: 'center', margin: '4px 0' }}>
                      <input type="checkbox" checked={params.guidedMatching} onChange={() => setC({ guidedMatching: !params.guidedMatching })} /> {t('f_guidedMatching')}
                    </label>
                  </>}
                  <NumField label={t('f_twoViewInliers')} value={params.twoViewMinInliers} onChange={v => setC({ twoViewMinInliers: v })} />
                  <div className="hint" style={{ fontWeight: 600, marginTop: 6 }}>{t('grp_mapper')}</div>
                  <label style={{ display: 'inline-flex', gap: 6, alignItems: 'center', margin: '4px 0' }}>
                    <input type="checkbox" checked={params.baUseGpu} onChange={() => setC({ baUseGpu: !params.baUseGpu })} /> {t('f_baUseGpu')}
                  </label>
                  <NumField label={t('f_mapperMinMatches')} value={params.mapperMinNumMatches} onChange={v => setC({ mapperMinNumMatches: v })} />
                  <NumField label={t('f_initMinInliers')} value={params.initMinNumInliers} onChange={v => setC({ initMinNumInliers: v })} />
                  <NumField label={t('f_absPoseMaxError')} value={params.absPoseMaxError} step={0.5} onChange={v => setC({ absPoseMaxError: v })} />
                  <NumField label={t('f_filterMaxReproj')} value={params.filterMaxReprojError} step={0.5} onChange={v => setC({ filterMaxReprojError: v })} />
                  <NumField label={t('f_filterMinTriAngle')} value={params.filterMinTriAngle} step={0.5} onChange={v => setC({ filterMinTriAngle: v })} />
                  <NumField label={t('f_baLocalIters')} value={params.baLocalIters} onChange={v => setC({ baLocalIters: v })} />
                  <NumField label={t('f_baGlobalIters')} value={params.baGlobalIters} onChange={v => setC({ baGlobalIters: v })} />
                  <NumField label={t('f_minModelSize')} value={params.minModelSize} onChange={v => setC({ minModelSize: v })} />
                </div>
              )
            })()}
          </div>
        </>
      )}

      {stage === 'reproject_views' && (
        <Slider label="Pinhole size" hint="" min={512} max={2048} step={128}
          value={params.size} onChange={v => setParams({ size: Math.round(v) })} fmt={v => `${v}px`} />
      )}

      {stage === 'inspect_source' && (
        <div className="ctl">
          <button className="btn" onClick={onSelectSource}>{t('selectSource')}</button>
          <div className="hint">{t('hint_selectsource')}</div>
          <div style={{ marginTop: 8 }}>
            <div className="hint">{t('source')}</div>
            <div className="mono" style={{ wordBreak: 'break-all', fontSize: 11, color: sourcePath ? 'var(--fg)' : 'var(--fg-mute)' }}>
              {sourcePath ?? t('noSource')}
            </div>
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
          </div>
          {exportInfo
            ? <div className="ctl">
                <label>{t('exportDir')}</label>
                <div className="mono" style={{ wordBreak: 'break-all', fontSize: 11 }}>{exportInfo.dir}</div>
                {exportInfo.dataset_dir && (
                  <div style={{ marginTop: 6 }}>
                    <div className="hint">{t('exportDataset')}</div>
                    <div className="mono" style={{ wordBreak: 'break-all', fontSize: 11 }}>{exportInfo.dataset_dir}</div>
                  </div>
                )}
                {exportInfo.preview_dir && (
                  <div style={{ marginTop: 6 }}>
                    <div className="hint">{t('exportPreview')}</div>
                    <div className="mono" style={{ wordBreak: 'break-all', fontSize: 11 }}>{exportInfo.preview_dir}</div>
                  </div>
                )}
                {exportInfo.train_configs_dir && (
                  <div style={{ marginTop: 6 }}>
                    <div className="hint">{t('exportTrainConfigs')}</div>
                    <div className="mono" style={{ wordBreak: 'break-all', fontSize: 11 }}>{exportInfo.train_configs_dir}</div>
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
}) => (
  <div className="ctl" style={{ marginBottom: 6 }}>
    <label style={{ fontSize: 11 }}>{label}</label>
    <input className="input" type="number" min={0} step={step} value={value || ''}
      placeholder="default (0)" onChange={e => onChange(Number(e.target.value) || 0)} />
  </div>
)

const Slider = ({
  label, hint, min, max, step, value, onChange, fmt,
}: {
  label: string; hint?: string; min: number; max: number; step: number; value: number
  onChange: (v: number) => void; fmt: (v: number) => string
}) => (
  <div className="ctl">
    <label>{label}</label>
    <div className="row">
      <input type="range" min={min} max={max} step={step} value={value} onChange={e => onChange(Number(e.target.value))} />
      <span className="val">{fmt(value)}</span>
    </div>
    {hint && <div className="hint">{hint}</div>}
  </div>
)
