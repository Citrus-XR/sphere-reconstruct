import { expect, test } from '@playwright/test'
import * as THREE from 'three'
import {
  composeViewQuaternion,
  configureGridMaterial,
  formatMovementSpeed,
  panViewPosition,
  sceneClippingPlanes,
  stepMovementSpeed,
} from '../src/viewers/viewMath'

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
