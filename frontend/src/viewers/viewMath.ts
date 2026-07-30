import * as THREE from 'three'

export const MOVEMENT_SPEED_MULTIPLIERS = [
  0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16,
] as const

export const stepMovementSpeed = (current: number, direction: -1 | 1): number => {
  const currentIndex = MOVEMENT_SPEED_MULTIPLIERS.reduce((closest, value, index) => (
    Math.abs(value - current) < Math.abs(MOVEMENT_SPEED_MULTIPLIERS[closest] - current)
      ? index
      : closest
  ), 0)
  const nextIndex = THREE.MathUtils.clamp(
    currentIndex + direction,
    0,
    MOVEMENT_SPEED_MULTIPLIERS.length - 1,
  )
  return MOVEMENT_SPEED_MULTIPLIERS[nextIndex]
}

export const formatMovementSpeed = (value: number): string => `${value}x`

export const sceneClippingPlanes = (sceneScale: number, speedMultiplier: number): {
  near: number
  far: number
} => {
  const near = Math.max(0.000001, sceneScale * 0.00001 * speedMultiplier)
  const far = Math.max(sceneScale * 2, sceneScale * 100 * speedMultiplier, near * 1000)
  return { near, far }
}

export const wheelZoomDistance = (
  sceneScale: number,
  wheelDelta: number,
  speedMultiplier: number,
): number => -wheelDelta * sceneScale * 0.0009 * speedMultiplier

// Local quaternion の累積は通常 mouse look に roll を混入させるため、YXZ から毎回再構成する。
export const composeViewQuaternion = (
  yaw: number,
  pitch: number,
  roll: number,
  target = new THREE.Quaternion(),
): THREE.Quaternion => target.setFromEuler(new THREE.Euler(pitch, yaw, roll, 'YXZ'))

export const panViewPosition = (
  position: THREE.Vector3,
  quaternion: THREE.Quaternion,
  movementX: number,
  movementY: number,
  sceneScale: number,
): THREE.Vector3 => {
  const distance = sceneScale * 0.0012
  const screenRight = new THREE.Vector3(1, 0, 0).applyQuaternion(quaternion)
  const screenUp = new THREE.Vector3(0, 1, 0).applyQuaternion(quaternion)
  return position
    .addScaledVector(screenRight, -movementX * distance)
    .addScaledVector(screenUp, movementY * distance)
}

export const configureGridMaterial = (material: THREE.Material): void => {
  material.depthTest = true
  material.depthWrite = false
  material.polygonOffset = true
  material.polygonOffsetFactor = 1
  material.polygonOffsetUnits = 1
  material.side = THREE.DoubleSide
  material.needsUpdate = true
}
