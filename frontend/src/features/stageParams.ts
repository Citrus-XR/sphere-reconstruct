// 各 stage の UI state と API parameter 変換. 重い処理を独立再実行できる粒度に保つ.

export type ReconMode = 'native_fisheye' | 'pinhole_rig' | 'equirectangular'
export type QualityPreset = 'draft' | 'standard' | 'high' | 'custom'

export interface StageParams {
  fps: number
  method: 'interval' | 'sharpness' | 'spatial'
  sharpnessLevel: 'basic' | 'better' | 'best'
  maxFrames: number
  targetMotion: number
  candidateFps: number
  minSharpness: number
  minFeatures: number
  maxClip: number
  maskSize: number
  downsampleOn: boolean
  dilate: number
  dilateOn: boolean
  prompt: string
  qualityPreset: QualityPreset
  featureType: 'SIFT' | 'ALIKED_N16ROT' | 'ALIKED_N32'
  featureUseGpu: boolean
  featureMaxImageSize: number
  featureMaxNumFeatures: number
  siftPeakThreshold: number
  siftEdgeThreshold: number
  siftAffineDsp: boolean
  matcherType: 'bruteforce' | 'lightglue'
  pairing: 'sequential' | 'exhaustive' | 'vocab_tree'
  matchingUseGpu: boolean
  overlap: number
  loopClosure: boolean
  maxNumMatches: number
  guidedMatching: boolean
  twoViewMinInliers: number
  mapper: 'global' | 'incremental'
  viewGraphCalibration: boolean
  baUseGpu: boolean
  mapperRandomSeed: number
  mapperMinNumMatches: number
  initMinNumInliers: number
  initImageId1: number
  initImageId2: number
  absPoseMaxError: number
  filterMaxReprojError: number
  filterMinTriAngle: number
  baLocalIters: number
  baGlobalIters: number
  minModelSize: number
  minRegisteredRatio: number
  minPoints3D: number
  alignmentMethod: 'auto' | 'imu' | 'none'
  size: number
  emitTrainConfigs: boolean
}

export const QUALITY_PRESETS: Record<Exclude<QualityPreset, 'custom'>, Partial<StageParams>> = {
  draft: {
    featureType: 'SIFT',
    featureMaxImageSize: 1536,
    featureMaxNumFeatures: 4096,
    matcherType: 'bruteforce',
    maxNumMatches: 8192,
    baLocalIters: 15,
    baGlobalIters: 50,
  },
  standard: {
    featureType: 'SIFT',
    featureMaxImageSize: 2048,
    featureMaxNumFeatures: 8192,
    matcherType: 'bruteforce',
    maxNumMatches: 16384,
    baLocalIters: 25,
    baGlobalIters: 100,
  },
  high: {
    featureType: 'SIFT',
    featureMaxImageSize: 3072,
    featureMaxNumFeatures: 16384,
    matcherType: 'bruteforce',
    maxNumMatches: 32768,
    baLocalIters: 40,
    baGlobalIters: 200,
  },
}

export const DEFAULT_PARAMS: StageParams = {
  fps: 1,
  method: 'sharpness',
  sharpnessLevel: 'better',
  maxFrames: 0,
  targetMotion: 8,
  candidateFps: 1.5,
  minSharpness: 0,
  minFeatures: 50,
  maxClip: 0.25,
  maskSize: 1024,
  downsampleOn: true,
  dilate: 8,
  dilateOn: true,
  prompt: '',
  qualityPreset: 'standard',
  featureType: 'SIFT',
  featureUseGpu: true,
  featureMaxImageSize: 2048,
  featureMaxNumFeatures: 8192,
  siftPeakThreshold: 0,
  siftEdgeThreshold: 0,
  siftAffineDsp: false,
  matcherType: 'bruteforce',
  pairing: 'sequential',
  matchingUseGpu: true,
  overlap: 4,
  loopClosure: false,
  maxNumMatches: 16384,
  guidedMatching: false,
  twoViewMinInliers: 15,
  mapper: 'global',
  viewGraphCalibration: true,
  baUseGpu: false,
  mapperRandomSeed: 0,
  mapperMinNumMatches: 0,
  initMinNumInliers: 0,
  initImageId1: 0,
  initImageId2: 0,
  absPoseMaxError: 0,
  filterMaxReprojError: 0,
  filterMinTriAngle: 0,
  baLocalIters: 25,
  baGlobalIters: 100,
  minModelSize: 0,
  minRegisteredRatio: 0.8,
  minPoints3D: 100,
  alignmentMethod: 'auto',
  size: 1024,
  emitTrainConfigs: true,
}

const sharpnessCandidates = (level: StageParams['sharpnessLevel']) =>
  level === 'better' ? 5 : level === 'best' ? 8 : 3

export const paramsForStage = (
  stage: string,
  params: StageParams,
  mode: ReconMode,
): Record<string, unknown> => {
  switch (stage) {
    case 'extract_frames':
      if (params.method === 'spatial') {
        return {
          selection_mode: 'spatial',
          candidate_fps: params.candidateFps,
          target_motion: params.targetMotion,
          min_sharpness: params.minSharpness,
          min_features: params.minFeatures,
          max_clip: params.maxClip,
          max_frames: params.maxFrames,
        }
      }
      return {
        interval_sec: 1 / params.fps,
        selection_mode: params.method,
        sharpness_candidates: params.method === 'sharpness'
          ? sharpnessCandidates(params.sharpnessLevel)
          : 1,
        max_frames: params.maxFrames,
      }
    case 'reproject_views':
      return { size: params.size, fov_deg: 90 }
    case 'generate_masks':
      return {
        max_inference_size: params.downsampleOn ? params.maskSize : 0,
        dilate_px: params.dilateOn ? params.dilate : 0,
        prompt: params.prompt,
        layout: mode === 'native_fisheye' ? 'fisheye' : mode === 'pinhole_rig' ? 'pinhole' : 'erp',
      }
    case 'extract_features':
      return {
        reconstruction_mode: mode,
        feature_type: params.featureType,
        use_gpu: params.featureUseGpu,
        use_masks: true,
        max_image_size: params.featureMaxImageSize,
        max_num_features: params.featureMaxNumFeatures,
        sift_peak_threshold: params.siftPeakThreshold,
        sift_edge_threshold: params.siftEdgeThreshold,
        sift_affine_dsp: params.siftAffineDsp,
      }
    case 'match_features':
      return {
        feature_type: params.featureType,
        matcher_type: params.matcherType,
        pairing: params.pairing,
        use_gpu: params.matchingUseGpu,
        overlap: params.overlap,
        loop_closure: params.loopClosure,
        max_num_matches: params.maxNumMatches,
        guided_matching: params.guidedMatching,
        min_num_inliers: params.twoViewMinInliers,
      }
    case 'reconstruct':
      return {
        mapper: params.mapper,
        view_graph_calibration: params.viewGraphCalibration,
        ba_use_gpu: params.baUseGpu,
        random_seed: params.mapperRandomSeed,
        mapper_min_num_matches: params.mapperMinNumMatches,
        init_min_num_inliers: params.initMinNumInliers,
        init_image_id1: params.initImageId1,
        init_image_id2: params.initImageId2,
        abs_pose_max_error: params.absPoseMaxError,
        filter_max_reproj_error: params.filterMaxReprojError,
        filter_min_tri_angle: params.filterMinTriAngle,
        ba_local_max_num_iterations: params.baLocalIters,
        ba_global_max_num_iterations: params.baGlobalIters,
        min_model_size: params.minModelSize,
        min_registered_ratio: params.minRegisteredRatio,
        min_points3D: params.minPoints3D,
      }
    case 'align_reconstruction':
      return { method: params.alignmentMethod, normalize_scale: true }
    case 'export_dataset':
      return {
        emit_train_configs: params.emitTrainConfigs,
      }
    default:
      return {}
  }
}

export const allParams = (
  params: StageParams,
  mode: ReconMode,
): Record<string, Record<string, unknown>> => {
  const stages = [
    'extract_frames',
    'reproject_views',
    'generate_masks',
    'extract_features',
    'match_features',
    'reconstruct',
    'align_reconstruction',
    'export_dataset',
  ]
  return Object.fromEntries(stages.map(stage => [stage, paramsForStage(stage, params, mode)]))
}
