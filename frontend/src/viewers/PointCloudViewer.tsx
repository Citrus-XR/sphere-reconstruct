import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { Canvas, useThree, useFrame } from '@react-three/fiber'
import { GizmoHelper, GizmoViewport, Grid } from '@react-three/drei'
import * as THREE from 'three'
import { fetchPoints, type ParsedPoints, type ReconstructionData } from '../api/client'
import { GlassSurface } from '../components/GlassSurface'
import { AppIcon } from '../components/AppIcon'
import {
  composeViewQuaternion,
  configureGridMaterial,
  formatMovementSpeed,
  panViewPosition,
  sceneClippingPlanes,
  stepMovementSpeed,
  wheelZoomDistance,
} from './viewMath'
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
      <pointsMaterial size={size} vertexColors map={tex} alphaTest={0.5}
        transparent={false} depthTest depthWrite sizeAttenuation={false} />
    </points>
  )
}

const StableGrid = ({ scale, center, groundY, light }: {
  scale: number
  center: THREE.Vector3
  groundY: number
  light: boolean
}) => {
  const ref = useRef<THREE.Mesh>(null)
  useLayoutEffect(() => {
    const material = ref.current?.material
    if (material && !Array.isArray(material)) configureGridMaterial(material)
  }, [])
  return (
    <Grid ref={ref} args={[scale * 4, scale * 4]}
      position={[center.x, groundY, center.z]}
      cellSize={scale * 0.05} sectionSize={scale * 0.5}
      infiniteGrid={false} followCamera={false} fadeDistance={scale * 2}
      fadeStrength={1.5} cellThickness={0.42} sectionThickness={0.82}
      cellColor={light ? '#b9c5d5' : '#43546c'} sectionColor={light ? '#7897bd' : '#7596c2'}
      side={THREE.DoubleSide} renderOrder={-100} />
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

const PITCH_LIMIT = Math.PI / 2 - 0.01

const FlyControls = ({ scale, center, selectedPos, onSpeedChange }: {
  scale: number
  center: THREE.Vector3
  selectedPos: THREE.Vector3 | null
  onSpeedChange: (speed: number) => void
}) => {
  const { camera, gl } = useThree()
  const interaction = useRef<'look' | 'pan' | null>(null)
  const capturedPointer = useRef<number | null>(null)
  const keys = useRef<Record<string, boolean>>({})
  const angles = useRef({ yaw: 0, pitch: 0, roll: 0 })
  const speedMultiplier = useRef(1)
  const wheelAccumulator = useRef(0)

  const applyOrientation = () => {
    const value = angles.current
    composeViewQuaternion(value.yaw, value.pitch, value.roll, camera.quaternion)
  }

  const readOrientation = () => {
    const value = new THREE.Euler().setFromQuaternion(camera.quaternion, 'YXZ')
    angles.current = { yaw: value.y, pitch: value.x, roll: value.z }
  }

  const applyClippingPlanes = () => {
    const clipping = sceneClippingPlanes(scale, speedMultiplier.current)
    camera.near = clipping.near
    camera.far = clipping.far
    camera.updateProjectionMatrix()
  }

  const fitTo = (tgt: THREE.Vector3, dist: number) => {
    camera.position.set(tgt.x, tgt.y, tgt.z + dist)
    camera.up.set(0, 1, 0)
    angles.current = { yaw: 0, pitch: 0, roll: 0 }
    applyOrientation()
  }
  useEffect(() => {
    fitTo(center, scale)
    applyClippingPlanes()
  }, [center, scale]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const dom = gl.domElement
    const stopInteraction = () => {
      if (interaction.current === 'look' && document.pointerLockElement)
        document.exitPointerLock()
      if (capturedPointer.current !== null && dom.hasPointerCapture?.(capturedPointer.current))
        dom.releasePointerCapture(capturedPointer.current)
      interaction.current = null
      capturedPointer.current = null
      dom.style.cursor = ''
    }
    const onDown = (e: PointerEvent) => {
      if (interaction.current !== null) return
      if (e.button === 2) {
        readOrientation()
        interaction.current = 'look'
        dom.requestPointerLock?.()
      } else if (e.button === 1) {
        e.preventDefault()
        interaction.current = 'pan'
        capturedPointer.current = e.pointerId
        dom.setPointerCapture?.(e.pointerId)
        dom.style.cursor = 'grabbing'
      }
    }
    const onUp = (e: PointerEvent) => {
      if ((e.button === 2 && interaction.current === 'look')
        || (e.button === 1 && interaction.current === 'pan')) stopInteraction()
    }
    const onMove = (e: PointerEvent) => {
      if (interaction.current === 'look') {
        const s = 0.0022
        angles.current.yaw -= e.movementX * s
        angles.current.pitch = THREE.MathUtils.clamp(
          angles.current.pitch - e.movementY * s,
          -PITCH_LIMIT,
          PITCH_LIMIT,
        )
        applyOrientation()
      } else if (interaction.current === 'pan') {
        panViewPosition(
          camera.position,
          camera.quaternion,
          e.movementX,
          e.movementY,
          scale * speedMultiplier.current,
        )
      }
    }
    const onCtx = (e: Event) => e.preventDefault()
    const onAux = (e: MouseEvent) => { if (e.button === 1) e.preventDefault() }
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const normalizedDelta = e.deltaY * (
        e.deltaMode === WheelEvent.DOM_DELTA_LINE
          ? 16
          : e.deltaMode === WheelEvent.DOM_DELTA_PAGE
            ? dom.clientHeight
            : 1
      )
      if (interaction.current !== 'look') {
        wheelAccumulator.current = 0
        const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion)
        camera.position.addScaledVector(
          forward,
          wheelZoomDistance(scale, normalizedDelta, speedMultiplier.current),
        )
        return
      }
      wheelAccumulator.current += normalizedDelta
      if (Math.abs(wheelAccumulator.current) < 50) return
      speedMultiplier.current = stepMovementSpeed(
        speedMultiplier.current,
        wheelAccumulator.current < 0 ? 1 : -1,
      )
      wheelAccumulator.current = 0
      applyClippingPlanes()
      onSpeedChange(speedMultiplier.current)
    }
    const kd = (e: KeyboardEvent) => {
      if (typing()) return
      if (e.code === 'KeyF') { fitTo(selectedPos ?? center, selectedPos ? scale * 0.2 : scale); return }
      if (!e.repeat && (e.code === 'KeyZ' || e.code === 'KeyX')) readOrientation()
      if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.code)) e.preventDefault()
      keys.current[e.code] = true
    }
    const ku = (e: KeyboardEvent) => { keys.current[e.code] = false }
    const onBlur = () => { keys.current = {}; stopInteraction() }
    const onPointerLockChange = () => {
      if (interaction.current === 'look' && document.pointerLockElement !== dom)
        interaction.current = null
    }
    dom.addEventListener('pointerdown', onDown); window.addEventListener('pointerup', onUp)
    window.addEventListener('pointercancel', stopInteraction)
    window.addEventListener('pointermove', onMove); dom.addEventListener('contextmenu', onCtx)
    dom.addEventListener('auxclick', onAux); window.addEventListener('blur', onBlur)
    document.addEventListener('pointerlockchange', onPointerLockChange)
    dom.addEventListener('wheel', onWheel, { passive: false })
    window.addEventListener('keydown', kd); window.addEventListener('keyup', ku)
    return () => {
      dom.removeEventListener('pointerdown', onDown); window.removeEventListener('pointerup', onUp)
      window.removeEventListener('pointercancel', stopInteraction)
      window.removeEventListener('pointermove', onMove); dom.removeEventListener('contextmenu', onCtx)
      dom.removeEventListener('auxclick', onAux); window.removeEventListener('blur', onBlur)
      document.removeEventListener('pointerlockchange', onPointerLockChange)
      dom.removeEventListener('wheel', onWheel)
      window.removeEventListener('keydown', kd); window.removeEventListener('keyup', ku)
      stopInteraction()
    }
  }, [camera, gl, scale, center, selectedPos, onSpeedChange])

  useFrame((_, dt) => {
    const k = keys.current
    const rollDirection = (k.KeyZ ? 1 : 0) - (k.KeyX ? 1 : 0)
    if (rollDirection !== 0) {
      angles.current.roll = THREE.MathUtils.euclideanModulo(
        angles.current.roll + rollDirection * 1.3 * dt + Math.PI,
        Math.PI * 2,
      ) - Math.PI
      applyOrientation()
    }
    const v = new THREE.Vector3(
      (k.KeyD || k.ArrowRight ? 1 : 0) - (k.KeyA || k.ArrowLeft ? 1 : 0),
      (k.KeyE ? 1 : 0) - (k.KeyQ ? 1 : 0),
      (k.KeyS || k.ArrowDown ? 1 : 0) - (k.KeyW || k.ArrowUp ? 1 : 0),
    )
    if (v.lengthSq() === 0) return
    const speed = scale * 0.9 * speedMultiplier.current * dt
      * (k.ShiftLeft || k.ShiftRight ? 3 : 1)
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
  const { t } = useSettings()
  const tex = useCircleTexture()
  const [points, setPoints] = useState<ParsedPoints | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [pointSize, setPointSize] = useState(2.5)
  const [showGrid, setShowGrid] = useState(true)
  const [showCenter, setShowCenter] = useState(true)
  const [speedNotice, setSpeedNotice] = useState<{ value: number; revision: number } | null>(null)
  const speedNoticeRevision = useRef(0)
  const speedNoticeTimer = useRef<number | null>(null)
  // 背景色: 明示指定が無ければテーマの --bg に追従する (テーマ切替で更新).
  const [bgOverride, setBgOverride] = useState<string | null>(null)
  const [themeBg, setThemeBg] = useState(() => (
    getComputedStyle(document.documentElement).getPropertyValue('--bg').trim() || '#171717'
  ))
  useLayoutEffect(() => {
    const root = document.documentElement
    const update = () => {
      const value = getComputedStyle(root).getPropertyValue('--bg').trim()
      if (value) setThemeBg(value)
    }
    const themeObserver = new MutationObserver(update)
    themeObserver.observe(root, { attributes: true, attributeFilter: ['data-theme'] })
    const colorScheme = window.matchMedia('(prefers-color-scheme: dark)')
    colorScheme.addEventListener('change', update)
    update()
    return () => {
      themeObserver.disconnect()
      colorScheme.removeEventListener('change', update)
    }
  }, [])
  const bg = bgOverride ?? themeBg

  const showSpeedNotice = useCallback((value: number) => {
    speedNoticeRevision.current += 1
    setSpeedNotice({ value, revision: speedNoticeRevision.current })
    if (speedNoticeTimer.current !== null) window.clearTimeout(speedNoticeTimer.current)
    speedNoticeTimer.current = window.setTimeout(() => {
      setSpeedNotice(null)
      speedNoticeTimer.current = null
    }, 900)
  }, [])

  useEffect(() => () => {
    if (speedNoticeTimer.current !== null) window.clearTimeout(speedNoticeTimer.current)
  }, [])

  useEffect(() => {
    let alive = true
    setPoints(null); setError(null)
    fetchPoints(projectId).then(p => alive && setPoints(p)).catch(e => alive && setError(String(e)))
    return () => { alive = false }
  }, [projectId, recon])

  // bbox から scene の中心とスケールを 1 度算出.
  const scene = useMemo(() => {
    if (!points || points.count === 0)
      return { scale: 10, center: new THREE.Vector3(), minimumY: 0 }
    const box = new THREE.Box3(), v = new THREE.Vector3()
    for (let i = 0; i < points.count; i++) {
      v.set(points.positions[i * 3], points.positions[i * 3 + 1], points.positions[i * 3 + 2])
      box.expandByPoint(v)
    }
    return {
      center: box.getCenter(new THREE.Vector3()),
      scale: box.getSize(new THREE.Vector3()).length() || 10,
      minimumY: box.min.y,
    }
  }, [points])

  const selectedPos = useMemo(() => {
    if (selectedCameraId == null) return null
    const img = recon.images.find(i => i.id === selectedCameraId)
    return img ? new THREE.Vector3(img.position[0], img.position[1], img.position[2]) : null
  }, [selectedCameraId, recon])

  if (error) return <div style={overlayMsg}>{error}</div>
  if (!points) return <div style={overlayMsg}>...</div>
  const gridGroundY = recon.ground_position?.applied ? 0 : scene.minimumY
  const backgroundColor = new THREE.Color(bg)
  const lightBackground = (
    backgroundColor.r * 0.2126 + backgroundColor.g * 0.7152 + backgroundColor.b * 0.0722
  ) > 0.55

  return (
    <div style={{ position: 'absolute', inset: 0 }}>
      <GlassSurface className="scene-toolbar-glass" cornerRadius={15} padding="0">
        <div className="scene-toolbar">
          <label><AppIcon name="color" size={15} /> {t('sc_bg')}
            <input type="color" value={bg} onChange={e => setBgOverride(e.target.value)} />
          </label>
          <label><AppIcon name="grid" size={15} />
            <input type="checkbox" checked={showGrid} onChange={e => setShowGrid(e.target.checked)} /> {t('sc_grid')}
          </label>
          <label><AppIcon name="target" size={15} />
            <input type="checkbox" checked={showCenter} onChange={e => setShowCenter(e.target.checked)} /> {t('sc_center')}
          </label>
          <label><AppIcon name="points" size={15} /> {t('sc_points')}
            <input type="range" min={1} max={8} step={0.5} value={pointSize}
            onChange={e => setPointSize(Number(e.target.value))} /></label>
        </div>
      </GlassSurface>

      {speedNotice && (
        <div key={speedNotice.revision} className="scene-speed-feedback" role="status" aria-live="polite">
          {formatMovementSpeed(speedNotice.value)}
        </div>
      )}

      <Canvas camera={{ position: [0, 0, 10], near: 0.01, far: 100000, fov: 55 }}>
        <color attach="background" args={[bg]} />
        <ambientLight intensity={0.9} />
        {showPoints && <PointCloud points={points} size={pointSize} tex={tex} />}
        {showCams && <CameraFrustums recon={recon} scale={scene.scale * 0.012} selectedId={selectedCameraId} onPick={onPickCamera} />}
        {showGrid && (
          <StableGrid scale={scene.scale} center={scene.center} groundY={gridGroundY}
            light={lightBackground} />
        )}
        {showCenter && (
          <mesh position={[scene.center.x, scene.center.y, scene.center.z]}>
            <sphereGeometry args={[scene.scale * 0.01, 12, 12]} />
            <meshBasicMaterial color="#ff5577" />
          </mesh>
        )}
        <FlyControls scale={scene.scale} center={scene.center} selectedPos={selectedPos}
          onSpeedChange={showSpeedNotice} />
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
