import { expect, test } from '@playwright/test'
import * as THREE from 'three'
import {
  composeViewQuaternion,
  configureGridMaterial,
  formatMovementSpeed,
  panViewPosition,
  sceneClippingPlanes,
  stepMovementSpeed,
  wheelZoomDistance,
} from '../src/viewers/viewMath'
import {
  buildSelectedCameraViewGeometry,
  cameraViewKind,
  pickCameraGizmo,
} from '../src/viewers/cameraGizmo'
import { createCircularPointMaterial } from '../src/viewers/pointRendering'
import { applyViewPose, equalViewPose, fitInitialView, readViewPose } from '../src/viewers/viewPose'

test('mouse yaw and pitch do not introduce camera roll', () => {
  for (const [yaw, pitch] of [[0.8, 0.4], [-1.7, 0.9], [2.4, -0.7]]) {
    const quaternion = composeViewQuaternion(yaw, pitch, 0)
    const cameraRight = new THREE.Vector3(1, 0, 0).applyQuaternion(quaternion)
    expect(Math.abs(cameraRight.y)).toBeLessThan(1e-10)
  }
})

test('explicit Z/X roll remains independent from mouse orientation', () => {
  const upright = composeViewQuaternion(1.1, -0.5, 0)
  const rolled = composeViewQuaternion(1.1, -0.5, 0.4)
  const uprightRight = new THREE.Vector3(1, 0, 0).applyQuaternion(upright)
  const rolledRight = new THREE.Vector3(1, 0, 0).applyQuaternion(rolled)

  expect(Math.abs(uprightRight.y)).toBeLessThan(1e-10)
  expect(Math.abs(rolledRight.y)).toBeGreaterThan(0.1)
})

test('middle-button pan follows screen axes', () => {
  const position = new THREE.Vector3(0, 0, 5)
  panViewPosition(position, new THREE.Quaternion(), 100, 50, 10)

  expect(position.x).toBeCloseTo(-1.2)
  expect(position.y).toBeCloseTo(0.6)
  expect(position.z).toBeCloseTo(5)
})

test('middle-button pan remains in the screen plane after roll', () => {
  const quaternion = composeViewQuaternion(0.7, -0.4, 0.5)
  const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(quaternion)
  const before = new THREE.Vector3(2, 3, 4)
  const after = panViewPosition(before.clone(), quaternion, 80, -35, 12)
  const displacement = after.sub(before)

  expect(Math.abs(displacement.dot(forward))).toBeLessThan(1e-10)
})

test('scene grid keeps perspective depth without occluding point geometry', () => {
  const material = new THREE.ShaderMaterial({ transparent: true })

  configureGridMaterial(material)

  expect(material.depthTest).toBe(true)
  expect(material.depthWrite).toBe(false)
  expect(material.side).toBe(THREE.DoubleSide)
  expect(material.polygonOffset).toBe(true)
  expect(material.polygonOffsetFactor).toBe(1)
})

test('movement speed follows bounded half and double steps', () => {
  expect(stepMovementSpeed(1, -1)).toBe(0.5)
  expect(stepMovementSpeed(0.01, -1)).toBe(0.005)
  expect(stepMovementSpeed(0.001, -1)).toBe(0.001)
  expect(stepMovementSpeed(1, 1)).toBe(2)
  expect(stepMovementSpeed(16, 1)).toBe(16)
  expect(formatMovementSpeed(0.5)).toBe('0.5x')
  expect(formatMovementSpeed(0.001)).toBe('0.001x')
})

test('camera clipping planes follow scene scale and movement speed', () => {
  expect(sceneClippingPlanes(1000, 1)).toEqual({ near: 0.01, far: 100000 })
  expect(sceneClippingPlanes(1000, 0.001)).toEqual({ near: 0.00001, far: 2000 })
  expect(sceneClippingPlanes(1000, 16)).toEqual({ near: 0.16, far: 1600000 })
})

test('ordinary wheel zoom distance follows the movement-speed multiplier', () => {
  expect(wheelZoomDistance(1000, 100, 1)).toBeCloseTo(-90)
  expect(wheelZoomDistance(1000, 100, 0.5)).toBeCloseTo(-45)
  expect(wheelZoomDistance(1000, -100, 2)).toBeCloseTo(180)
})

test('camera model selects a projection-specific selected view', () => {
  expect(cameraViewKind('PINHOLE')).toBe('perspective')
  expect(cameraViewKind('OPENCV_FISHEYE')).toBe('fisheye')
  expect(cameraViewKind('THIN_PRISM_FISHEYE')).toBe('fisheye')
  expect(cameraViewKind('EQUIRECTANGULAR')).toBe('spherical')
})

test('fisheye selected view uses a circular boundary instead of a square frustum', () => {
  const depth = 10
  const geometry = buildSelectedCameraViewGeometry('fisheye', depth, 1)
  const positions = geometry.getAttribute('position') as THREE.BufferAttribute
  const ringRadii: number[] = []
  for (let index = 0; index < positions.count; index += 1) {
    const z = positions.getZ(index)
    if (Math.abs(z - depth * 0.35) < 1e-6)
      ringRadii.push(Math.hypot(positions.getX(index), positions.getY(index)))
  }

  expect(ringRadii.length).toBeGreaterThanOrEqual(64)
  for (const radius of ringRadii) expect(radius).toBeCloseTo(depth * 0.94, 5)
  geometry.dispose()
})

test('perspective selected view preserves the camera image aspect ratio', () => {
  const geometry = buildSelectedCameraViewGeometry('perspective', 10, 2)
  const positions = geometry.getAttribute('position') as THREE.BufferAttribute
  const far = Array.from({ length: positions.count }, (_, index) => [
    positions.getX(index), positions.getY(index), positions.getZ(index),
  ]).filter(position => position[2] === 10 && position[0] !== 0)
  const maximumX = Math.max(...far.map(position => Math.abs(position[0])))
  const maximumY = Math.max(...far.map(position => Math.abs(position[1])))

  expect(maximumX / maximumY).toBeCloseTo(2)
  geometry.dispose()
})

test('camera gizmo picking uses a fixed screen-pixel radius and ignores empty space', () => {
  const camera = new THREE.PerspectiveCamera(90, 1, 0.1, 100)
  camera.position.set(0, 0, 0)
  camera.lookAt(0, 0, -1)
  camera.updateMatrixWorld()
  camera.updateProjectionMatrix()
  const targets = [{ id: 1, position: [0, 0, -5] }]

  expect(pickCameraGizmo(targets, camera, 50, 50, 100, 100)).toBe(1)
  expect(pickCameraGizmo(targets, camera, 57.9, 50, 100, 100)).toBe(1)
  expect(pickCameraGizmo(targets, camera, 58.1, 50, 100, 100)).toBeNull()
  expect(pickCameraGizmo(targets, camera, 5, 5, 100, 100)).toBeNull()
})

test('overlapping camera gizmos select the closest visible camera', () => {
  const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 100)
  camera.updateMatrixWorld()
  camera.updateProjectionMatrix()

  expect(pickCameraGizmo([
    { id: 1, position: [0, 0, -8] },
    { id: 2, position: [0, 0, -3] },
    { id: 3, position: [0, 0, 3] },
  ], camera, 50, 50, 100, 100)).toBe(2)
})

test('empty scene starts above the ground with the center in view', () => {
  const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 100000)
  const center = new THREE.Vector3(3, 2, -8)
  fitInitialView(camera, center, 10)

  const projected = center.clone().project(camera)
  expect(projected.x).toBeCloseTo(0)
  expect(projected.y).toBeCloseTo(0)
  expect(camera.position.y).toBeGreaterThan(center.y)
  expect(camera.position.x).not.toBe(center.x)
  expect(camera.position.distanceTo(center)).toBeGreaterThan(9)
})

test('saved viewer pose roundtrips without sharing mutable camera arrays', () => {
  const camera = new THREE.PerspectiveCamera()
  camera.position.set(10, -3, 17)
  camera.quaternion.copy(composeViewQuaternion(0.7, -0.4, 0.2))
  const saved = readViewPose(camera)
  const restored = new THREE.PerspectiveCamera()
  applyViewPose(restored, saved)

  expect(equalViewPose(saved, readViewPose(restored))).toBe(true)
  restored.position.x += 1
  expect(equalViewPose(saved, readViewPose(restored))).toBe(false)
  expect(saved.position[0]).toBe(10)
})

test('point material keeps image colors unlit and uses circular MSAA coverage', () => {
  const material = createCircularPointMaterial(true)
  expect(material.vertexColors).toBe(true)
  expect(material.toneMapped).toBe(false)
  expect(material.sizeAttenuation).toBe(false)
  expect(material.depthWrite).toBe(true)
  expect(material.alphaToCoverage).toBe(true)
  expect(material.transparent).toBe(false)
  const shader = {
    fragmentShader: THREE.ShaderLib.points.fragmentShader,
    vertexShader: THREE.ShaderLib.points.vertexShader,
    uniforms: {},
  }
  material.onBeforeCompile(shader, {} as THREE.WebGLRenderer)
  expect(shader.fragmentShader).toContain('fwidth(circleDistance)')
  expect(shader.fragmentShader).toContain('if (diffuseColor.a <= 0.0) discard;')
  expect(shader.fragmentShader).toContain('#include <colorspace_fragment>')
  expect(material.map).toBeNull()
  material.dispose()
})

test('point material uses blended edge coverage when MSAA is unavailable', () => {
  const material = createCircularPointMaterial(false)
  expect(material.alphaToCoverage).toBe(false)
  expect(material.transparent).toBe(true)
  expect(material.depthTest).toBe(true)
  material.dispose()
})
