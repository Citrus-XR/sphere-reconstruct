// 各 stage の UI state と API parameter 変換. 重い処理を独立再実行できる粒度に保つ.

import type { Projection } from '../api/client'

export type ReconMode = 'native_fisheye' | 'pinhole_rig' | 'equirectangular'
export type QualityPreset = 'draft' | 'standard' | 'high' | 'custom'
export type ColmapTriangulationPreset = 'default' | 'standard' | 'strict' | 'custom'

export interface ColmapTriangulationValues {
  filterMaxReprojError: number
  filterMinTriAngle: number
  triCreateMaxAngleError: number
  triContinueMaxAngleError: number
  triMergeMaxReprojError: number
  triCompleteMaxReprojError: number
  triMinAngle: number
}

export const COLMAP_TRIANGULATION_PRESETS: Record<Exclude<ColmapTriangulationPreset, 'custom'>, ColmapTriangulationValues> = {
  default: {
    filterMaxReprojError: 4,
    filterMinTriAngle: 1.5,
    triCreateMaxAngleError: 2,
    triContinueMaxAngleError: 2,
    triMergeMaxReprojError: 4,
    triCompleteMaxReprojError: 4,
    triMinAngle: 1.5,
  },
  standard: {
    filterMaxReprojError: 1.5,
    filterMinTriAngle: 3,
    triCreateMaxAngleError: 1,
    triContinueMaxAngleError: 1,
    triMergeMaxReprojError: 1.5,
    triCompleteMaxReprojError: 1.5,
    triMinAngle: 3,
  },
  strict: {
    filterMaxReprojError: 1,
    filterMinTriAngle: 5,
    triCreateMaxAngleError: 0.75,
    triContinueMaxAngleError: 0.75,
    triMergeMaxReprojError: 1,
    triCompleteMaxReprojError: 1,
    triMinAngle: 5,
  },
}

export const triangulationPresetForValues = (values: ColmapTriangulationValues): ColmapTriangulationPreset => (
  (Object.entries(COLMAP_TRIANGULATION_PRESETS) as [Exclude<ColmapTriangulationPreset, 'custom'>, ColmapTriangulationValues][])
    .find(([, preset]) => (Object.keys(preset) as (keyof ColmapTriangulationValues)[])
      .every(key => values[key] === preset[key]))?.[0] ?? 'custom'
)

const RECON_MODES_BY_PROJECTION: Record<Projection, readonly ReconMode[]> = {
  dual_fisheye: ['native_fisheye', 'pinhole_rig'],
  equirectangular: ['equirectangular', 'pinhole_rig'],
  perspective: ['pinhole_rig'],
}

export const reconModesForProjection = (projection: Projection | null): readonly ReconMode[] => (
  projection === null
    ? ['native_fisheye', 'pinhole_rig', 'equirectangular']
    : RECON_MODES_BY_PROJECTION[projection]
)

export const defaultReconModeForProjection = (projection: Projection | null): ReconMode => (
  reconModesForProjection(projection)[0]
)

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
  maxTemporalGap: number
  continuityStrategy: 'quality' | 'parallax' | 'balanced'
  maxRollingShutterMotion: number
  featureMaskEnabled: boolean
  featureMaskSizeAuto: boolean
  featureMaskSize: number
  featureMaskDownsampleOn: boolean
  featureMaskDilate: number
  featureMaskDilateOn: boolean
  featureMaskPrompt: string
  trainingMaskEnabled: boolean
  trainingMaskSizeAuto: boolean
  trainingMaskSize: number
  trainingMaskDownsampleOn: boolean
  trainingMaskDilate: number
  trainingMaskDilateOn: boolean
  trainingMaskPrompt: string
  qualityPreset: QualityPreset
  featureType: 'SIFT' | 'ALIKED_N16ROT' | 'ALIKED_N32'
  featureUseGpu: boolean
  featureMaxImageSizeAuto: boolean
  featureMaxImageSize: number
  featureMaxNumFeaturesAuto: boolean
  featureMaxNumFeatures: number
  siftPeakThreshold: number
  siftEdgeThreshold: number
  siftAffineDsp: boolean
  matcherType: 'bruteforce' | 'lightglue'
  pairing: 'auto' | 'sequential' | 'exhaustive' | 'vocab_tree'
  matchingUseGpu: boolean
  overlap: number
  loopClosure: boolean
  transitiveMatching: boolean
  maxNumMatches: number
  guidedMatching: boolean
  twoViewMinInliers: number
  rigVerification: boolean
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
  triCreateMaxAngleError: number
  triContinueMaxAngleError: number
  triMergeMaxReprojError: number
  triCompleteMaxReprojError: number
  triMinAngle: number
  baLocalIters: number
  baGlobalIters: number
  minModelSize: number
  minRegisteredRatio: number
  minPoints3D: number
  alignmentMethod: 'auto' | 'imu' | 'none'
  metricScaleMethod: 'auto' | 'rig' | 'none'
  sceneAlignmentMethod: 'auto' | 'required' | 'none'
  cleanupSparseEnabled: boolean
  cleanupRelativeError: number
  cleanupPixelSigma: number
  cleanupMaxCrossError: number
  denseEnabled: boolean
  denseQuality: 'turbo' | 'fast' | 'base' | 'high'
  denseReferenceFraction: number
  denseNeighbors: number
  denseMatchesPerPair: number
  denseConfidenceThreshold: number
  denseReprojectionThreshold: number
  denseMinimumParallax: number
  denseMaximumPoints: number
  denseVoxelRatio: number
  denseUseFeatureMasks: boolean
  sizeAuto: boolean
  size: number
  emitTrainConfigs: boolean
  optimizeFisheyeTrainingImages: boolean
}

export const QUALITY_PRESETS: Record<Exclude<QualityPreset, 'custom'>, Partial<StageParams>> = {
  draft: {
    featureType: 'SIFT',
    matcherType: 'bruteforce',
    maxNumMatches: 8192,
    baLocalIters: 15,
    baGlobalIters: 50,
  },
  standard: {
    featureType: 'SIFT',
    matcherType: 'bruteforce',
    maxNumMatches: 16384,
    baLocalIters: 25,
    baGlobalIters: 100,
  },
  high: {
    featureType: 'SIFT',
    matcherType: 'bruteforce',
    maxNumMatches: 32768,
    baLocalIters: 40,
    baGlobalIters: 200,
  },
}

export const DEFAULT_PARAMS: StageParams = {
  fps: 1,
  method: 'spatial',
  sharpnessLevel: 'better',
  maxFrames: 0,
  targetMotion: 2,
  candidateFps: 1.5,
  minSharpness: 0,
  minFeatures: 50,
  maxClip: 0.25,
  maxTemporalGap: 4,
  continuityStrategy: 'balanced',
  maxRollingShutterMotion: 0.8,
  featureMaskEnabled: true,
  featureMaskSizeAuto: true,
  featureMaskSize: 2048,
  featureMaskDownsampleOn: true,
  featureMaskDilate: 8,
  featureMaskDilateOn: true,
  featureMaskPrompt: '',
  trainingMaskEnabled: true,
  trainingMaskSizeAuto: true,
  trainingMaskSize: 2048,
  trainingMaskDownsampleOn: true,
  trainingMaskDilate: 8,
  trainingMaskDilateOn: true,
  trainingMaskPrompt: '',
  qualityPreset: 'standard',
  featureType: 'SIFT',
  featureUseGpu: true,
  featureMaxImageSizeAuto: true,
  featureMaxImageSize: 2048,
  featureMaxNumFeaturesAuto: true,
  featureMaxNumFeatures: 8192,
  siftPeakThreshold: 0,
  siftEdgeThreshold: 0,
  siftAffineDsp: false,
  matcherType: 'bruteforce',
  pairing: 'auto',
  matchingUseGpu: true,
  overlap: 4,
  loopClosure: false,
  transitiveMatching: false,
  maxNumMatches: 16384,
  guidedMatching: false,
  twoViewMinInliers: 15,
  rigVerification: true,
  mapper: 'incremental',
  viewGraphCalibration: false,
  baUseGpu: false,
  mapperRandomSeed: 0,
  mapperMinNumMatches: 0,
  initMinNumInliers: 0,
  initImageId1: 0,
  initImageId2: 0,
  absPoseMaxError: 0,
  ...COLMAP_TRIANGULATION_PRESETS.strict,
  baLocalIters: 25,
  baGlobalIters: 100,
  minModelSize: 0,
  minRegisteredRatio: 0.8,
  minPoints3D: 100,
  alignmentMethod: 'auto',
  metricScaleMethod: 'auto',
  sceneAlignmentMethod: 'auto',
  cleanupSparseEnabled: true,
  cleanupRelativeError: 0.02,
  cleanupPixelSigma: 1,
  cleanupMaxCrossError: 2,
  denseEnabled: false,
  denseQuality: 'turbo',
  denseReferenceFraction: 0.25,
  denseNeighbors: 2,
  denseMatchesPerPair: 2000,
  denseConfidenceThreshold: 0.2,
  denseReprojectionThreshold: 1.5,
  denseMinimumParallax: 0.5,
  denseMaximumPoints: 200000,
  denseVoxelRatio: 0.0005,
  denseUseFeatureMasks: true,
  sizeAuto: true,
  size: 1024,
  emitTrainConfigs: true,
  optimizeFisheyeTrainingImages: true,
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
          sensor_order: 'calibration_lens_order',
          selection_mode: 'spatial',
          candidate_fps: params.candidateFps,
          target_motion: params.targetMotion,
          min_sharpness: params.minSharpness,
          min_features: params.minFeatures,
          max_clip: params.maxClip,
          max_temporal_gap_sec: params.maxTemporalGap,
          continuity_strategy: params.continuityStrategy,
          max_rolling_shutter_motion_deg: params.maxRollingShutterMotion,
          max_frames: params.maxFrames,
        }
      }
      return {
        sensor_order: 'calibration_lens_order',
        interval_sec: 1 / params.fps,
        selection_mode: params.method,
        sharpness_candidates: params.method === 'sharpness'
          ? sharpnessCandidates(params.sharpnessLevel)
          : 1,
        max_frames: params.maxFrames,
        max_rolling_shutter_motion_deg: params.maxRollingShutterMotion,
      }
    case 'prepare_images':
      return mode === 'pinhole_rig'
        ? { reconstruction_mode: mode, size: params.size, fov_deg: 90 }
        : { reconstruction_mode: mode }
    case 'rectify_fisheye':
      return { projection_contract: 'radtan_pro_v2' }
    case 'generate_feature_masks':
      return {
        max_inference_size: params.featureMaskDownsampleOn ? params.featureMaskSize : 0,
        dilate_px: params.featureMaskDilateOn ? params.featureMaskDilate : 0,
        prompt: params.featureMaskPrompt,
      }
    case 'generate_training_masks':
      return {
        max_inference_size: params.trainingMaskDownsampleOn ? params.trainingMaskSize : 0,
        dilate_px: params.trainingMaskDilateOn ? params.trainingMaskDilate : 0,
        prompt: params.trainingMaskPrompt,
      }
    case 'extract_features':
      return {
        reconstruction_mode: mode,
        feature_type: params.featureType,
        use_gpu: params.featureUseGpu,
        use_feature_masks: params.featureMaskEnabled,
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
        transitive_matching: params.transitiveMatching,
        max_num_matches: params.maxNumMatches,
        guided_matching: params.guidedMatching,
        min_num_inliers: params.twoViewMinInliers,
        rig_verification: params.rigVerification,
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
        tri_create_max_angle_error: params.triCreateMaxAngleError,
        tri_continue_max_angle_error: params.triContinueMaxAngleError,
        tri_merge_max_reproj_error: params.triMergeMaxReprojError,
        tri_complete_max_reproj_error: params.triCompleteMaxReprojError,
        tri_min_angle: params.triMinAngle,
        ba_local_max_num_iterations: params.baLocalIters,
        ba_global_max_num_iterations: params.baGlobalIters,
        min_model_size: params.minModelSize,
        min_registered_ratio: params.minRegisteredRatio,
        min_points3D: params.minPoints3D,
      }
    case 'align_reconstruction':
      return { method: params.alignmentMethod }
    case 'restore_metric_scale':
      return { method: params.metricScaleMethod }
    case 'scene_alignment':
      return { method: params.sceneAlignmentMethod, align_manhattan_axes: true }
    case 'cleanup_sparse':
      return {
        enabled: params.cleanupSparseEnabled,
        relative_error: params.cleanupRelativeError,
        pixel_sigma: params.cleanupPixelSigma,
        max_cross_error: params.cleanupMaxCrossError,
      }
    case 'dense_initialization':
      return {
        enabled: params.denseEnabled,
        quality: params.denseQuality,
        reference_fraction: params.denseReferenceFraction,
        neighbors_per_reference: params.denseNeighbors,
        matches_per_pair: params.denseMatchesPerPair,
        confidence_threshold: params.denseConfidenceThreshold,
        reprojection_threshold_px: params.denseReprojectionThreshold,
        minimum_parallax_deg: params.denseMinimumParallax,
        maximum_new_points: params.denseMaximumPoints,
        voxel_size_ratio: params.denseVoxelRatio,
        use_feature_masks: params.denseUseFeatureMasks,
      }
    case 'export_dataset':
      return {
        emit_train_configs: params.emitTrainConfigs,
        optimize_fisheye_training_images: params.optimizeFisheyeTrainingImages,
        feature_masks_enabled: params.featureMaskEnabled,
        training_masks_enabled: params.trainingMaskEnabled,
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
    'export_dataset',
  ]
  return Object.fromEntries(stages.map(stage => [stage, paramsForStage(stage, params, mode)]))
}
