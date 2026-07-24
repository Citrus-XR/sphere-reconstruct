import { useEffect, useMemo, useRef, useState } from 'react'
import { Canvas, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import { fetchPoints, type ParsedPoints, type ReconstructionData } from '../api/client'

// 点群を THREE.Points で描画する.
const PointCloud = ({ points, size }: { points: ParsedPoints; size: number }) => {
  const geom = useMemo(() => {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(points.positions, 3))
    g.setAttribute('color', new THREE.BufferAttribute(points.colors, 3))
    return g
  }, [points])
  return (
    <points geometry={geom}>
      <pointsMaterial size={size} vertexColors sizeAttenuation={false} />
    </points>
  )
}

// 各カメラ位置に小さなマーカーを置く. 登録済み画像は緑, その他は灰.
const CameraMarkers = ({ recon }: { recon: ReconstructionData }) => {
  const positions = useMemo(() => {
    const arr = new Float32Array(recon.images.length * 3)
    recon.images.forEach((img, i) => {
      arr[i * 3] = img.position[0]
      arr[i * 3 + 1] = img.position[1]
      arr[i * 3 + 2] = img.position[2]
    })
    return arr
  }, [recon])
  const geom = useMemo(() => {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    return g
  }, [positions])
  return (
    <points geometry={geom}>
      <pointsMaterial size={8} color="#3ad07a" sizeAttenuation={false} />
    </points>
  )
}

// 点群の重心へカメラを一度だけ寄せる.
const FitOnce = ({ points }: { points: ParsedPoints }) => {
  const { camera } = useThree()
  const done = useRef(false)
  useEffect(() => {
    if (done.current || points.count === 0) return
    done.current = true
    const box = new THREE.Box3()
    const v = new THREE.Vector3()
    for (let i = 0; i < points.count; i++) {
      v.set(points.positions[i * 3], points.positions[i * 3 + 1], points.positions[i * 3 + 2])
      box.expandByPoint(v)
    }
    const center = box.getCenter(new THREE.Vector3())
    const size = box.getSize(new THREE.Vector3()).length() || 10
    camera.position.set(center.x, center.y, center.z + size)
    camera.lookAt(center)
  }, [points, camera])
  return null
}

export const PointCloudViewer = ({
  projectId,
  recon,
}: {
  projectId: string
  recon: ReconstructionData
}) => {
  const [points, setPoints] = useState<ParsedPoints | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [pointSize, setPointSize] = useState(2)

  useEffect(() => {
    let alive = true
    fetchPoints(projectId)
      .then(p => alive && setPoints(p))
      .catch(e => alive && setError(String(e)))
    return () => {
      alive = false
    }
  }, [projectId])

  if (error) return <div className="error">{error}</div>
  if (!points) return <div>点群を読み込み中...</div>

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div className="mono">
        {points.count.toLocaleString()} points · {recon.images.length} cameras · mean err{' '}
        {recon.stats.mean_reprojection_error.toFixed(3)}px
        <label style={{ marginLeft: 16 }}>
          点サイズ{' '}
          <input
            type="range"
            min={1}
            max={6}
            step={0.5}
            value={pointSize}
            onChange={e => setPointSize(Number(e.target.value))}
          />
        </label>
      </div>
      <div style={{ height: 560, background: '#0b0b0b', borderRadius: 8 }}>
        <Canvas camera={{ position: [0, 0, 10], near: 0.01, far: 10000, fov: 50 }}>
          <ambientLight intensity={0.8} />
          <PointCloud points={points} size={pointSize} />
          <CameraMarkers recon={recon} />
          <FitOnce points={points} />
          <axesHelper args={[1]} />
          <OrbitControls makeDefault />
        </Canvas>
      </div>
    </div>
  )
}
