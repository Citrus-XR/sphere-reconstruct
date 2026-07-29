import * as THREE from 'three'

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
