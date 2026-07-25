import { useEffect, useMemo, useRef, useState } from 'react'
import { Canvas, useThree, useFrame } from '@react-three/fiber'
import { GizmoHelper, GizmoViewport, Grid } from '@react-three/drei'
import * as THREE from 'three'
import { fetchPoints, type ParsedPoints, type ReconstructionData } from '../api/client'
import { useSettings } from '../ui/settings'

const useCircleTexture = () =>
  useMemo(() => {
    const s = 64
    const c = document.createElement('canvas'); c.width = c.height = s
    const ctx = c.getContext('2d')!
    ctx.beginPath(); ctx.arc(s / 2, s / 2, s / 2 - 2, 0, Math.PI * 2); ctx.fillStyle = '#fff'; ctx.fill()
    const tex = new THREE.CanvasTexture(c); tex.needsUpdate = true; return tex
  }, [])

const PointCloud = ({ points, size, tex }: { points: ParsedPoints; size: number; tex: THREE.Texture }) => {
  const geom = useMemo(() => {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(points.positions, 3))
    g.setAttribute('color', new THREE.BufferAttribute(points.colors, 3))
    return g
  }, [points])
  return (
    <points geometry={geom}>
      <pointsMaterial size={size} vertexColors map={tex} alphaTest={0.5} transparent sizeAttenuation={false} />
    </points>
  )
}

const CameraFrustums = ({ recon, scale, selectedId, onPick }: {
  recon: ReconstructionData; scale: number; selectedId: number | null; onPick: (id: number) => void
}) => {
  const geom = useMemo(() => {
    const s = scale, d = scale * 1.6
    const c = [[s, s, d], [-s, s, d], [-s, -s, d], [s, -s, d]]
    const segs: number[] = []
    for (const p of c) segs.push(0, 0, 0, ...p)
    for (let i = 0; i < 4; i++) segs.push(...c[i], ...c[(i + 1) % 4])
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.Float32BufferAttribute(segs, 3))
    return g
  }, [scale])
  return (
    <>
      {recon.images.map(img => {
        const q = new THREE.Quaternion(img.qvec[1], img.qvec[2], img.qvec[3], img.qvec[0]).invert()
        const sel = img.id === selectedId
        return (
          <group key={img.id} position={[img.position[0], img.position[1], img.position[2]]} quaternion={q}>
            <lineSegments geometry={geom}>
              <lineBasicMaterial color={sel ? '#ffcc33' : '#3ad07a'} />
            </lineSegments>
            <mesh onClick={e => { e.stopPropagation(); onPick(img.id) }}>
              <sphereGeometry args={[scale * 1.1, 8, 8]} />
              <meshBasicMaterial transparent opacity={sel ? 0.25 : 0} color="#ffcc33" depthWrite={false} />
            </mesh>
          </group>
        )
      })}
    </>
  )
}

const typing = () => {
  const el = document.activeElement
  return !!el && ['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName)
}

// Unity シーンビュー風の操作:
//   右ドラッグ=視点回転 (マウス掴み), WASD/矢印=移動, E/Q=上下, Z/X=傾き(ロール),
//   F=フォーカス (選択カメラ or シーン中心), ホイール=前後. 慣性なし.
// 回転はカメラ自身のローカル軸まわりの四元数増分で行う (ロールで傾けた後も, 右ドラッグ回転が
// 世界空間ではなく現在の傾いた視点基準になる).
const AXIS_X = new THREE.Vector3(1, 0, 0)
const AXIS_Y = new THREE.Vector3(0, 1, 0)
const AXIS_Z = new THREE.Vector3(0, 0, 1)

const FlyControls = ({ scale, center, selectedPos }: {
  scale: number; center: THREE.Vector3; selectedPos: THREE.Vector3 | null
}) => {
  const { camera, gl } = useThree()
  const look = useRef(false)
  const keys = useRef<Record<string, boolean>>({})

  const fitTo = (tgt: THREE.Vector3, dist: number) => {
    camera.position.set(tgt.x, tgt.y, tgt.z + dist)
    camera.up.set(0, 1, 0)
    camera.lookAt(tgt) // ロールを 0 に戻して正立させる.
  }
  useEffect(() => { fitTo(center, scale) }, [center, scale]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const dom = gl.domElement
    const onDown = (e: PointerEvent) => {
      if (e.button !== 2) return
      look.current = true; dom.requestPointerLock?.()
    }
    const onUp = (e: PointerEvent) => {
      if (e.button !== 2) return
      look.current = false
      if (document.pointerLockElement) document.exitPointerLock()
    }
    const onMove = (e: PointerEvent) => {
      if (!look.current) return
      const s = 0.0022
      // ローカル軸まわりに右から掛ける (現在の姿勢基準で yaw/pitch).
      camera.quaternion
        .multiply(new THREE.Quaternion().setFromAxisAngle(AXIS_Y, -e.movementX * s))
        .multiply(new THREE.Quaternion().setFromAxisAngle(AXIS_X, -e.movementY * s))
    }
    const onCtx = (e: Event) => e.preventDefault()
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion)
      camera.position.addScaledVector(fwd, -e.deltaY * scale * 0.0009)
    }
    const kd = (e: KeyboardEvent) => {
      if (typing()) return
      if (e.code === 'KeyF') { fitTo(selectedPos ?? center, selectedPos ? scale * 0.2 : scale); return }
      if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.code)) e.preventDefault()
      keys.current[e.code] = true
    }
    const ku = (e: KeyboardEvent) => { keys.current[e.code] = false }
    dom.addEventListener('pointerdown', onDown); window.addEventListener('pointerup', onUp)
    window.addEventListener('pointermove', onMove); dom.addEventListener('contextmenu', onCtx)
    dom.addEventListener('wheel', onWheel, { passive: false })
    window.addEventListener('keydown', kd); window.addEventListener('keyup', ku)
    return () => {
      dom.removeEventListener('pointerdown', onDown); window.removeEventListener('pointerup', onUp)
      window.removeEventListener('pointermove', onMove); dom.removeEventListener('contextmenu', onCtx)
      dom.removeEventListener('wheel', onWheel)
      window.removeEventListener('keydown', kd); window.removeEventListener('keyup', ku)
    }
  }, [camera, gl, scale, center, selectedPos])

  useFrame((_, dt) => {
    const k = keys.current
    // ロールはローカル前方軸 (Z) まわり.
    if (k.KeyZ) camera.quaternion.multiply(new THREE.Quaternion().setFromAxisAngle(AXIS_Z, 1.3 * dt))
    if (k.KeyX) camera.quaternion.multiply(new THREE.Quaternion().setFromAxisAngle(AXIS_Z, -1.3 * dt))
    const v = new THREE.Vector3(
      (k.KeyD || k.ArrowRight ? 1 : 0) - (k.KeyA || k.ArrowLeft ? 1 : 0),
      (k.KeyE ? 1 : 0) - (k.KeyQ ? 1 : 0),
      (k.KeyS || k.ArrowDown ? 1 : 0) - (k.KeyW || k.ArrowUp ? 1 : 0),
    )
    if (v.lengthSq() === 0) return
    const speed = scale * 0.9 * dt * (k.ShiftLeft || k.ShiftRight ? 3 : 1)
    v.normalize().multiplyScalar(speed).applyQuaternion(camera.quaternion)
    camera.position.add(v)
  })
  return null
}

export const PointCloudViewer = ({
  projectId, recon, showPoints, showCams, selectedCameraId, onPickCamera,
}: {
  projectId: string; recon: ReconstructionData
  showPoints: boolean; showCams: boolean
  selectedCameraId: number | null; onPickCamera: (id: number) => void
}) => {
  const { t, theme } = useSettings()
  const tex = useCircleTexture()
  const [points, setPoints] = useState<ParsedPoints | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [pointSize, setPointSize] = useState(2.5)
  const [showGrid, setShowGrid] = useState(true)
  const [showCenter, setShowCenter] = useState(true)
  // 背景色: 明示指定が無ければテーマの --bg に追従する (テーマ切替で更新).
  const [bgOverride, setBgOverride] = useState<string | null>(null)
  const [themeBg, setThemeBg] = useState('#171717')
  useEffect(() => {
    const v = getComputedStyle(document.documentElement).getPropertyValue('--bg').trim()
    if (v) setThemeBg(v)
  }, [theme])
  const bg = bgOverride ?? themeBg

  useEffect(() => {
    let alive = true
    setPoints(null); setError(null)
    fetchPoints(projectId).then(p => alive && setPoints(p)).catch(e => alive && setError(String(e)))
    return () => { alive = false }
  }, [projectId])

  // bbox から scene の中心とスケールを 1 度算出.
  const scene = useMemo(() => {
    if (!points || points.count === 0) return { scale: 10, center: new THREE.Vector3() }
    const box = new THREE.Box3(), v = new THREE.Vector3()
    for (let i = 0; i < points.count; i++) {
      v.set(points.positions[i * 3], points.positions[i * 3 + 1], points.positions[i * 3 + 2])
      box.expandByPoint(v)
    }
    return { center: box.getCenter(new THREE.Vector3()), scale: box.getSize(new THREE.Vector3()).length() || 10 }
  }, [points])

  const selectedPos = useMemo(() => {
    if (selectedCameraId == null) return null
    const img = recon.images.find(i => i.id === selectedCameraId)
    return img ? new THREE.Vector3(img.position[0], img.position[1], img.position[2]) : null
  }, [selectedCameraId, recon])

  if (error) return <div style={overlayMsg}>{error}</div>
  if (!points) return <div style={overlayMsg}>...</div>

  return (
    <div style={{ position: 'absolute', inset: 0 }}>
      <div className="scene-toolbar">
        <label>{t('sc_bg')} <input type="color" value={bg} onChange={e => setBgOverride(e.target.value)} /></label>
        <label><input type="checkbox" checked={showGrid} onChange={e => setShowGrid(e.target.checked)} /> {t('sc_grid')}</label>
        <label><input type="checkbox" checked={showCenter} onChange={e => setShowCenter(e.target.checked)} /> {t('sc_center')}</label>
        <label>{t('sc_points')} <input type="range" min={1} max={8} step={0.5} value={pointSize}
          onChange={e => setPointSize(Number(e.target.value))} /></label>
        <span style={{ marginLeft: 'auto', opacity: 0.6 }}>右=視点 / WASD·矢印=移動 / Z X=傾き / F=フォーカス</span>
      </div>

      <Canvas camera={{ position: [0, 0, 10], near: 0.01, far: 100000, fov: 55 }}>
        <color attach="background" args={[bg]} />
        <ambientLight intensity={0.9} />
        {showPoints && <PointCloud points={points} size={pointSize} tex={tex} />}
        {showCams && <CameraFrustums recon={recon} scale={scene.scale * 0.012} selectedId={selectedCameraId} onPick={onPickCamera} />}
        {showGrid && (
          <Grid args={[scene.scale * 2, scene.scale * 2]}
            position={[scene.center.x, scene.center.y - scene.scale * 0.5, scene.center.z]}
            cellSize={scene.scale * 0.1} sectionSize={scene.scale * 0.5} infiniteGrid
            fadeDistance={scene.scale * 8} cellColor="#444" sectionColor="#666" />
        )}
        {showCenter && (
          <mesh position={[scene.center.x, scene.center.y, scene.center.z]}>
            <sphereGeometry args={[scene.scale * 0.01, 12, 12]} />
            <meshBasicMaterial color="#ff5577" />
          </mesh>
        )}
        <FlyControls scale={scene.scale} center={scene.center} selectedPos={selectedPos} />
        <GizmoHelper alignment="top-right" margin={[56, 92]}>
          <GizmoViewport axisColors={['#ff5566', '#4caf50', '#4488ff']} labelColor="#fff" />
        </GizmoHelper>
      </Canvas>
    </div>
  )
}

const overlayMsg: React.CSSProperties = {
  position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center',
  color: '#888', fontFamily: 'ui-monospace, monospace', fontSize: 12,
}
