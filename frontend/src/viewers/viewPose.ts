import * as THREE from 'three'
import type { ViewerCameraPose } from '../api/client'

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
