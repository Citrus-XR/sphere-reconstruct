import { expect, test } from '@playwright/test'
import {
  COLMAP_TRIANGULATION_PRESETS,
  DEFAULT_PARAMS,
  paramsForStage,
  triangulationPresetForValues,
} from '../src/features/stageParams'

test('COLMAP triangulation presets contain the documented default, standard and strict values', () => {
  expect(COLMAP_TRIANGULATION_PRESETS.default).toEqual({
    filterMaxReprojError: 4,
    filterMinTriAngle: 1.5,
    triCreateMaxAngleError: 2,
    triContinueMaxAngleError: 2,
    triMergeMaxReprojError: 4,
    triCompleteMaxReprojError: 4,
    triMinAngle: 1.5,
  })
  expect(COLMAP_TRIANGULATION_PRESETS.standard).toEqual({
    filterMaxReprojError: 1.5,
    filterMinTriAngle: 3,
    triCreateMaxAngleError: 1,
    triContinueMaxAngleError: 1,
    triMergeMaxReprojError: 1.5,
    triCompleteMaxReprojError: 1.5,
    triMinAngle: 3,
  })
  expect(COLMAP_TRIANGULATION_PRESETS.strict).toEqual({
    filterMaxReprojError: 1,
    filterMinTriAngle: 5,
    triCreateMaxAngleError: 0.75,
    triContinueMaxAngleError: 0.75,
    triMergeMaxReprojError: 1,
    triCompleteMaxReprojError: 1,
    triMinAngle: 5,
  })
})

test('new reconstruction settings select strict gates and retain selectable COLMAP defaults', () => {
  expect(triangulationPresetForValues(DEFAULT_PARAMS)).toBe('strict')
  expect(triangulationPresetForValues({ ...DEFAULT_PARAMS, filterMaxReprojError: 0 })).toBe('custom')
  expect(paramsForStage('reconstruct', DEFAULT_PARAMS, 'native_fisheye')).toMatchObject({
    filter_max_reproj_error: 1,
    filter_min_tri_angle: 5,
    tri_create_max_angle_error: 0.75,
    tri_continue_max_angle_error: 0.75,
    tri_merge_max_reproj_error: 1,
    tri_complete_max_reproj_error: 1,
    tri_min_angle: 5,
  })
  expect(paramsForStage('reconstruct', {
    ...DEFAULT_PARAMS,
    ...COLMAP_TRIANGULATION_PRESETS.default,
  }, 'native_fisheye')).toMatchObject({
    filter_max_reproj_error: 4,
    filter_min_tri_angle: 1.5,
    tri_create_max_angle_error: 2,
    tri_continue_max_angle_error: 2,
    tri_merge_max_reproj_error: 4,
    tri_complete_max_reproj_error: 4,
    tri_min_angle: 1.5,
  })
})

test('full-track cleanup sends only its current controls for fisheye and perspective workflows', () => {
  for (const mode of ['native_fisheye', 'pinhole_rig'] as const) {
    expect(paramsForStage('cleanup_sparse', DEFAULT_PARAMS, mode)).toEqual({
      enabled: true,
      relative_error: 0.02,
      pixel_sigma: 1,
      max_cross_error: 2,
    })
  }
})
