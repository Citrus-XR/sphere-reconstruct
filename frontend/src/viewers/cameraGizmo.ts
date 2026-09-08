import * as THREE from 'three'

export type CameraViewKind = 'perspective' | 'fisheye' | 'spherical'

export interface CameraPickTarget {
  id: number
  position: number[]
}

export const cameraViewKind = (model: string): CameraViewKind => {
  if (model.includes('EQUIRECTANGULAR')) return 'spherical'
  if (model.includes('FISHEYE')) return 'fisheye'
  return 'perspective'
}

export const pickCameraGizmo = (
  targets: CameraPickTarget[],
  camera: THREE.Camera,
  pointerX: number,
  pointerY: number,
  viewportWidth: number,
  viewportHeight: number,
  radiusPx = 8,
): number | null => {
  const projected = new THREE.Vector3()
  let bestId: number | null = null
  let bestDistanceSq = radiusPx * radiusPx
  let bestDepth = Number.POSITIVE_INFINITY
  for (const target of targets) {
    projected.set(target.position[0], target.position[1], target.position[2]).project(camera)
    if (projected.z < -1 || projected.z > 1) continue
    const screenX = (projected.x + 1) * 0.5 * viewportWidth
    const screenY = (1 - projected.y) * 0.5 * viewportHeight
    const distanceSq = (screenX - pointerX) ** 2 + (screenY - pointerY) ** 2
    if (distanceSq < bestDistanceSq || (distanceSq === bestDistanceSq && projected.z < bestDepth)) {
      bestId = target.id
      bestDistanceSq = distanceSq
      bestDepth = projected.z
    }
  }
  return bestId
}

const lineGeometry = (segments: number[]): THREE.BufferGeometry => {
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(segments, 3))
  return geometry
}

const addSegment = (
  segments: number[],
  from: readonly [number, number, number],
  to: readonly [number, number, number],
) => segments.push(...from, ...to)

export const buildSelectedCameraViewGeometry = (
  kind: CameraViewKind,
  depth: number,
  aspect: number,
): THREE.BufferGeometry => {
  const segments: number[] = []
  const origin: [number, number, number] = [0, 0, 0]
  addSegment(segments, origin, [0, 0, depth])

  if (kind === 'perspective') {
    const halfHeight = depth * 0.32
    const halfWidth = halfHeight * Math.max(0.25, Math.min(4, aspect))
    const corners: Array<[number, number, number]> = [
      [halfWidth, halfHeight, depth],
      [-halfWidth, halfHeight, depth],
      [-halfWidth, -halfHeight, depth],
      [halfWidth, -halfHeight, depth],
    ]
    for (const corner of corners) addSegment(segments, origin, corner)
    for (let index = 0; index < corners.length; index += 1)
      addSegment(segments, corners[index], corners[(index + 1) % corners.length])
    return lineGeometry(segments)
  }

  const circle = (
    axisA: 0 | 1 | 2,
    axisB: 0 | 1 | 2,
    centerZ: number,
    radius: number,
    connectOrigin: boolean,
  ) => {
    const count = 32
    const points: Array<[number, number, number]> = []
    for (let index = 0; index < count; index += 1) {
      const point: [number, number, number] = [0, 0, centerZ]
      const angle = index / count * Math.PI * 2
      point[axisA] = Math.cos(angle) * radius
      point[axisB] = Math.sin(angle) * radius
      points.push(point)
    }
    for (let index = 0; index < count; index += 1)
      addSegment(segments, points[index], points[(index + 1) % count])
    if (connectOrigin)
      for (let index = 0; index < count; index += count / 8)
        addSegment(segments, origin, points[index])
  }

  if (kind === 'fisheye') {
    circle(0, 1, depth * 0.35, depth * 0.94, true)
  } else {
    const radius = depth * 0.55
    circle(0, 1, 0, radius, false)
    circle(0, 2, 0, radius, false)
    circle(1, 2, 0, radius, false)
  }
  return lineGeometry(segments)
}

export const buildCameraIconTexture = (): THREE.CanvasTexture => {
  const size = 64
  const canvas = document.createElement('canvas')
  canvas.width = canvas.height = size
  const context = canvas.getContext('2d')!
  context.strokeStyle = '#fff'
  context.fillStyle = 'rgba(255,255,255,.18)'
  context.lineWidth = 5
  context.lineJoin = 'round'
  context.beginPath()
  context.roundRect(8, 17, 35, 30, 5)
  context.fill()
  context.stroke()
  context.beginPath()
  context.moveTo(43, 25)
  context.lineTo(57, 18)
  context.lineTo(57, 46)
  context.lineTo(43, 39)
  context.closePath()
  context.fill()
  context.stroke()
  context.beginPath()
  context.arc(25.5, 32, 8, 0, Math.PI * 2)
  context.stroke()
  const texture = new THREE.CanvasTexture(canvas)
  texture.colorSpace = THREE.SRGBColorSpace
  texture.needsUpdate = true
  return texture
}
