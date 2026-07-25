// ステージパラメータの型・既定値・params_by_stage 生成. StageSettings と Run All で共用.

export type ReconMode = 'native_fisheye' | 'pinhole_rig' | 'equirectangular'

export interface StageParams {
  fps: number
  method: 'interval' | 'sharpness' | 'spatial'
  sharpnessLevel: 'basic' | 'better' | 'best'
  maxFrames: number
  targetMotion: number
  // spatial 抽帧の高级参数.
  candidateFps: number
  minSharpness: number
  minFeatures: number
  maxClip: number
  maskSize: number
  downsampleOn: boolean
  dilate: number
  dilateOn: boolean
  prompt: string
  backend: 'sift' | 'aliked'
  device: 'auto' | 'cuda' | 'cpu'
  matcher: 'sequential' | 'exhaustive' | 'vocab_tree'
  loopClosure: boolean
  baUseGpu: boolean
  extractCapOn: boolean
  extractMaxSize: number
  overlap: number
  size: number
  emitTrainConfigs: boolean
  // COLMAP 詳細 (高级选项). 0/false = COLMAP 既定. プリセットが「大」値を snap する.
  colmapPreset: 'draft' | 'standard' | 'high' | 'custom'
  siftMaxFeatures: number
  siftMaxImageSize: number
  siftPeakThreshold: number
  siftEdgeThreshold: number
  siftAffineDsp: boolean
  maxNumMatches: number
  guidedMatching: boolean
  twoViewMinInliers: number
  mapperMinNumMatches: number
  initMinNumInliers: number
  absPoseMaxError: number
  filterMaxReprojError: number
  filterMinTriAngle: number
  baLocalIters: number
  baGlobalIters: number
  minModelSize: number
}

// COLMAP 品質プリセット: 「大」ノブを snap する. SIFT 特徴/マッチ数は backend=sift のみ効くが,
// BA 反復数は両 backend 共通なので, ALIKED でもプリセット変更が可視・有効になるよう含める.
export const COLMAP_PRESETS: Record<'draft' | 'standard' | 'high', Partial<StageParams>> = {
  draft: { siftMaxFeatures: 4096, siftMaxImageSize: 2048, maxNumMatches: 16384, baLocalIters: 15, baGlobalIters: 25 },
  standard: { siftMaxFeatures: 8192, siftMaxImageSize: 3200, maxNumMatches: 32768, baLocalIters: 0, baGlobalIters: 0 },
  high: { siftMaxFeatures: 16384, siftMaxImageSize: 4096, maxNumMatches: 65536, baLocalIters: 40, baGlobalIters: 75 },
}

export const DEFAULT_PARAMS: StageParams = {
  fps: 1.0,
  method: 'sharpness',
  sharpnessLevel: 'basic',
  maxFrames: 0,
  targetMotion: 1.5,
  candidateFps: 3.0,
  minSharpness: 0,
  minFeatures: 50,
  maxClip: 0.25,
  maskSize: 1024,
  downsampleOn: true,
  dilate: 8,
  dilateOn: true,
  prompt: '',
  backend: 'sift',
  device: 'auto',
  matcher: 'sequential',
  loopClosure: false,
  baUseGpu: false,
  extractCapOn: false,
  extractMaxSize: 2048,
  overlap: 10,
  size: 1024,
  emitTrainConfigs: false,
  colmapPreset: 'standard',
  siftMaxFeatures: 8192,
  siftMaxImageSize: 3200,
  siftPeakThreshold: 0,
  siftEdgeThreshold: 0,
  siftAffineDsp: false,
  maxNumMatches: 32768,
  guidedMatching: false,
  twoViewMinInliers: 0,
  mapperMinNumMatches: 0,
  initMinNumInliers: 0,
  absPoseMaxError: 0,
  filterMaxReprojError: 0,
  filterMinTriAngle: 0,
  baLocalIters: 0,
  baGlobalIters: 0,
  minModelSize: 0,
}

const sharpnessCandidates = (s: StageParams['sharpnessLevel']) =>
  s === 'better' ? 5 : s === 'best' ? 8 : 3

export const paramsForStage = (
  stage: string,
  p: StageParams,
  reconMode: ReconMode,
): Record<string, unknown> => {
  switch (stage) {
    case 'extract_frames':
      if (p.method === 'spatial') {
        // 密集候補 → 品質門閾(清晰度/曝光/特徴) → 光流等間隔.
        return {
          selection_mode: 'spatial',
          candidate_fps: p.candidateFps,
          target_motion: p.targetMotion,
          min_sharpness: p.minSharpness,
          min_features: p.minFeatures,
          max_clip: p.maxClip,
          max_frames: p.maxFrames,
        }
      }
      return {
        interval_sec: 1 / p.fps,
        selection_mode: p.method, // 'interval' | 'sharpness'
        sharpness_candidates: p.method === 'sharpness' ? sharpnessCandidates(p.sharpnessLevel) : 1,
        max_frames: p.maxFrames,
      }
    case 'generate_masks':
      return {
        max_inference_size: p.downsampleOn ? p.maskSize : 0, // 0 = 縮小しない.
        dilate_px: p.dilateOn ? p.dilate : 0,                // 0 = 膨張しない.
        prompt: p.prompt, // 常に送る (空なら SAM3 スキップ = 円マスクのみ).
        // レイアウトは再構成モードで決まる: fisheye(生魚眼) / pinhole(再投影像) / erp(生 ERP).
        layout: reconMode === 'native_fisheye' ? 'fisheye' : reconMode === 'pinhole_rig' ? 'pinhole' : 'erp',
      }
    case 'reconstruct':
      return {
        reconstruction_mode: reconMode,
        feature_backend: p.backend,
        matcher: p.matcher,
        overlap: p.overlap,
        loop_closure: p.loopClosure,
        ba_use_gpu: p.baUseGpu,
        ...(p.backend === 'aliked' ? { extraction_device: p.device, extract_max_size: p.extractCapOn ? p.extractMaxSize : 0 } : {}),
        // COLMAP 詳細 (0/false = 既定).
        sift_max_num_features: p.siftMaxFeatures,
        sift_max_image_size: p.siftMaxImageSize,
        sift_peak_threshold: p.siftPeakThreshold,
        sift_edge_threshold: p.siftEdgeThreshold,
        sift_affine_dsp: p.siftAffineDsp,
        max_num_matches: p.maxNumMatches,
        guided_matching: p.guidedMatching,
        two_view_min_num_inliers: p.twoViewMinInliers,
        mapper_min_num_matches: p.mapperMinNumMatches,
        init_min_num_inliers: p.initMinNumInliers,
        abs_pose_max_error: p.absPoseMaxError,
        filter_max_reproj_error: p.filterMaxReprojError,
        filter_min_tri_angle: p.filterMinTriAngle,
        ba_local_max_num_iterations: p.baLocalIters,
        ba_global_max_num_iterations: p.baGlobalIters,
        min_model_size: p.minModelSize,
      }
    case 'reproject_views':
      return { size: p.size, fov_deg: 90 }
    case 'export_dataset':
      return { emit_train_configs: p.emitTrainConfigs }
    default:
      return {}
  }
}

// Run All 用: 全ステージ分の params_by_stage.
export const allParams = (p: StageParams, reconMode: ReconMode): Record<string, Record<string, unknown>> => {
  const stages = ['extract_frames', 'generate_masks', 'reconstruct', 'reproject_views', 'export_dataset']
  const out: Record<string, Record<string, unknown>> = {}
  for (const s of stages) out[s] = paramsForStage(s, p, reconMode)
  return out
}
