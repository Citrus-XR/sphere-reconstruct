import { expect, test } from '@playwright/test'
import type { ProjectSource, SourceInfo } from '../src/api/client'
import { recommendSourceResolutions } from '../src/features/resolutionPolicy'

const source = (id: string, projection: ProjectSource['projection']): ProjectSource => ({
  id, projection, label: id, role: id === 'primary' ? 'primary' : 'supplemental',
  adapter: 'test', media_kind: 'video', path: id, ordinal: 0, enabled: true,
})

const info = (...sources: Array<{ id: string; width: number; height: number }>): SourceInfo => ({
  duration_sec: 1,
  sources: sources.map(item => ({ ...item, duration_sec: 1 })),
})

test('ERP preserves horizontal feature resolution and derives a quarter-width pinhole view', () => {
  const recommendation = recommendSourceResolutions(
    [source('primary', 'equirectangular')],
    info({ id: 'primary', width: 7680, height: 3840 }),
    'equirectangular',
  )

  expect(recommendation).toEqual({
    featureMaxImageSize: 7680,
    featureMaxNumFeatures: 32768,
    maskMaxInferenceSize: 4096,
    pinholeViewSize: 2048,
  })
})

test('dual fisheye keeps native features and raises SAM inference above 2048', () => {
  const recommendation = recommendSourceResolutions(
    [source('primary', 'dual_fisheye')],
    info({ id: 'primary', width: 5376, height: 5376 }),
    'native_fisheye',
  )

  expect(recommendation).toEqual({
    featureMaxImageSize: 5376,
    featureMaxNumFeatures: 16384,
    maskMaxInferenceSize: 3072,
    pinholeViewSize: 2816,
  })
})

test('mixed sources use the largest requirement for each resolution option', () => {
  const recommendation = recommendSourceResolutions(
    [source('primary', 'perspective'), source('erp', 'equirectangular'), source('fisheye', 'dual_fisheye')],
    info(
      { id: 'primary', width: 6000, height: 4000 },
      { id: 'erp', width: 8192, height: 4096 },
      { id: 'fisheye', width: 5376, height: 5376 },
    ),
    'equirectangular',
  )

  expect(recommendation).toEqual({
    featureMaxImageSize: 8192,
    featureMaxNumFeatures: 32768,
    maskMaxInferenceSize: 4096,
    pinholeViewSize: 2816,
  })
})
