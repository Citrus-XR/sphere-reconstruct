import * as THREE from 'three'
import type { ViewerCameraPose } from '../api/client'

export type RobustSceneBounds = {
  center: [number, number, number]
  scale: number
  minimumY: number
}

const quantile = (values: number[], fraction: number) => {
  if (values.length === 0) return 0
  const position = fraction * (values.length - 1)
  const low = Math.floor(position)
  const high = Math.min(values.length - 1, low + 1)
  return values[low] * (1 - (position - low)) + values[high] * (position - low)
}

/** Fit the viewer to the scene body while ignoring sparse reconstruction teleports. */
export const robustSceneBounds = (positions: ArrayLike<number>, count = Math.floor(positions.length / 3)):
  RobustSceneBounds => {
  const axes: [number[], number[], number[]] = [[], [], []]
  const validPoints: [number, number, number][] = []
  const limit = Math.min(count, Math.floor(positions.length / 3))
  for (let index = 0; index < limit; index += 1) {
    const x = Number(positions[index * 3])
    const y = Number(positions[index * 3 + 1])
    const z = Number(positions[index * 3 + 2])
    if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
      axes[0].push(x); axes[1].push(y); axes[2].push(z)
      validPoints.push([x, y, z])
    }
  }
  if (axes[0].length === 0) return { center: [0, 0, 0], scale: 10, minimumY: 0 }
  axes.forEach(axis => axis.sort((left, right) => left - right))
  const center: [number, number, number] = axes.map(axis => quantile(axis, 0.5)) as [number, number, number]
  const radii: number[] = []
  for (const point of validPoints) {
    const dx = point[0] - center[0]
    const dy = point[1] - center[1]
    const dz = point[2] - center[2]
    radii.push(Math.hypot(dx, dy, dz))
  }
  radii.sort((left, right) => left - right)
  const radius = quantile(radii, 0.95)
  const fallback = Math.hypot(
    quantile(axes[0], 0.975) - quantile(axes[0], 0.025),
    quantile(axes[1], 0.975) - quantile(axes[1], 0.025),
    quantile(axes[2], 0.975) - quantile(axes[2], 0.025),
  )
  const scale = Math.max(radius * 2, fallback, 1e-3)
  return {
    center,
    scale: Number.isFinite(scale) ? scale : 10,
    minimumY: quantile(axes[1], 0.025),
  }
}

export const readViewPose = (camera: THREE.Camera): ViewerCameraPose => ({
  position: [camera.position.x, camera.position.y, camera.position.z],
  quaternion: [camera.quaternion.x, camera.quaternion.y, camera.quaternion.z, camera.quaternion.w],
})

export const equalViewPose = (left: ViewerCameraPose, right: ViewerCameraPose) => (
  left.position.every((value, index) => Math.abs(value - right.position[index]) < 1e-10)
  && left.quaternion.every((value, index) => Math.abs(value - right.quaternion[index]) < 1e-10)
)

export const applyViewPose = (camera: THREE.Camera, pose: ViewerCameraPose) => {
  camera.position.fromArray(pose.position)
  camera.quaternion.fromArray(pose.quaternion).normalize()
  camera.updateMatrixWorld()
}

export const fitInitialView = (camera: THREE.Camera, center: THREE.Vector3, scale: number) => {
  camera.position.copy(center).add(new THREE.Vector3(0.55, 0.4, 0.8).multiplyScalar(scale))
  camera.up.set(0, 1, 0)
  camera.lookAt(center)
  camera.updateMatrixWorld()
}
