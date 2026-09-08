import type { ProjectSource, SourceInfo } from '../api/client'
import type { ReconMode } from './stageParams'

export interface ResolutionRecommendation {
  featureMaxImageSize: number
  featureMaxNumFeatures: number
  maskMaxInferenceSize: number
  pinholeViewSize: number
}

const roundUp = (value: number, step = 256) => Math.ceil(value / step) * step
const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value))

export const recommendSourceResolutions = (
  sources: ProjectSource[],
  sourceInfo: SourceInfo,
  reconMode: ReconMode,
): ResolutionRecommendation => {
  const dimensions = new Map((sourceInfo.sources ?? []).map(source => [source.id, source]))
  let featureMaxImageSize = 2048
  let featureMaxNumFeatures = 8192
  let maskMaxInferenceSize = 2048
  let pinholeViewSize = 1024

  for (const source of sources.filter(source => source.enabled)) {
    const info = dimensions.get(source.id)
    if (!info?.width || !info.height) continue
    const longestEdge = Math.max(info.width, info.height)
    const sphericalViewSize = source.projection === 'equirectangular'
      ? roundUp(info.width / 4)
      : source.projection === 'dual_fisheye'
        ? roundUp(longestEdge / 2)
        : 0
    if (sphericalViewSize > 0)
      pinholeViewSize = Math.max(pinholeViewSize, clamp(sphericalViewSize, 1024, 4096))

    const featureRequirement = source.projection === 'perspective'
      ? 2048
      : reconMode === 'pinhole_rig'
        ? Math.max(2048, clamp(sphericalViewSize, 1024, 4096))
        : source.projection === 'equirectangular'
          ? Math.max(2048, roundUp(info.width))
          : Math.max(2048, roundUp(longestEdge))
    featureMaxImageSize = Math.max(featureMaxImageSize, featureRequirement)
    const featureCountRequirement = source.projection === 'perspective' || reconMode === 'pinhole_rig'
      ? 8192
      : source.projection === 'equirectangular' ? 32768 : 16384
    featureMaxNumFeatures = Math.max(featureMaxNumFeatures, featureCountRequirement)

    const nativeMaskRequirement = source.projection === 'equirectangular'
      ? Math.max(2048, Math.min(4096, roundUp(info.width)))
      : source.projection === 'dual_fisheye'
        ? Math.max(2048, Math.min(3072, roundUp(longestEdge)))
        : 2048
    const maskRequirement = reconMode === 'pinhole_rig' && source.projection !== 'perspective'
      ? Math.max(2048, Math.min(3072, sphericalViewSize))
      : nativeMaskRequirement
    maskMaxInferenceSize = Math.max(maskMaxInferenceSize, maskRequirement)
  }

  return { featureMaxImageSize, featureMaxNumFeatures, maskMaxInferenceSize, pinholeViewSize }
}
