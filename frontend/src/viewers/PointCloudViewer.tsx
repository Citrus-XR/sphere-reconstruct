import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { Canvas, useThree, useFrame } from '@react-three/fiber'
import { GizmoHelper, GizmoViewport, Grid } from '@react-three/drei'
import * as THREE from 'three'
import {
  fetchPoints,
  type ParsedPoints,
  type ReconstructionData,
  type ViewerCameraPose,
  type ViewerPreferences,
} from '../api/client'
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
import {
  buildCameraIconTexture,
  buildSelectedCameraViewGeometry,
  cameraViewKind,
  pickCameraGizmo,
} from './cameraGizmo'
import { createCircularPointMaterial } from './pointRendering'
import { applyViewPose, equalViewPose, fitInitialView, readViewPose } from './viewPose'

const PointCloud = ({ points, size }: { points: ParsedPoints; size: number }) => {
  const { gl } = useThree()
  const material = useMemo(() => createCircularPointMaterial(
    gl.getContext().getContextAttributes()?.antialias === true,
  ), [gl])
  useLayoutEffect(() => { material.size = size }, [material, size])
  useEffect(() => () => material.dispose(), [material])
  const geom = useMemo(() => {
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(points.positions, 3))
    g.setAttribute('color', new THREE.BufferAttribute(points.colors, 3))
    return g
  }, [points])
  useEffect(() => () => geom.dispose(), [geom])
  return (
    <points geometry={geom} material={material} />
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

const CameraGizmos = ({ recon, scale, selectedId, onPick }: {
  recon: ReconstructionData
  scale: number
  selectedId: number | null
  onPick: (id: number | null) => void
}) => {
  const { camera, gl } = useThree()
  const texture = useMemo(buildCameraIconTexture, [])
  const picking = useRef({ images: recon.images, onPick })
  picking.current = { images: recon.images, onPick }
  useEffect(() => () => texture.dispose(), [texture])
  useEffect(() => {
    const canvas = gl.domElement
    const pick = (event: PointerEvent) => {
      const rect = canvas.getBoundingClientRect()
      return pickCameraGizmo(
        picking.current.images,
        camera,
        event.clientX - rect.left,
        event.clientY - rect.top,
        rect.width,
        rect.height,
      )
    }
    const onPointerDown = (event: PointerEvent) => {
      if (event.button === 0 && document.pointerLockElement !== canvas
        && canvas.style.cursor !== 'grabbing') picking.current.onPick(pick(event))
    }
    const onPointerMove = (event: PointerEvent) => {
      if (event.buttons !== 0 || document.pointerLockElement === canvas
        || canvas.style.cursor === 'grabbing') return
      canvas.style.cursor = pick(event) == null ? '' : 'pointer'
    }
    const clearHoverCursor = () => {
      // The hover layer must not release a cursor owned by drag controls.
      if (canvas.style.cursor === 'pointer') canvas.style.cursor = ''
    }
    canvas.addEventListener('pointerdown', onPointerDown)
    canvas.addEventListener('pointermove', onPointerMove)
    canvas.addEventListener('pointerleave', clearHoverCursor)
    return () => {
      canvas.removeEventListener('pointerdown', onPointerDown)
      canvas.removeEventListener('pointermove', onPointerMove)
      canvas.removeEventListener('pointerleave', clearHoverCursor)
      clearHoverCursor()
    }
  }, [camera, gl])
  const geometry = useMemo(() => {
    const positions = new Float32Array(recon.images.length * 3)
    recon.images.forEach((image, index) => {
      positions.set(image.position, index * 3)
    })
    const result = new THREE.BufferGeometry()
    result.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    return result
  }, [recon.images])
  useEffect(() => () => geometry.dispose(), [geometry])
  const selected = selectedId == null ? undefined : recon.images.find(image => image.id === selectedId)
  const selectedCamera = selected == null
    ? undefined
    : recon.cameras.find(camera => camera.id === selected.camera_id)
  const selectedView = useMemo(() => selectedCamera
    ? buildSelectedCameraViewGeometry(
        cameraViewKind(selectedCamera.model),
        scale * 0.006,
        selectedCamera.width / Math.max(1, selectedCamera.height),
      )
    : null, [selectedCamera, scale])
  useEffect(() => () => selectedView?.dispose(), [selectedView])
  const selectedIcon = useMemo(() => {
    if (!selected) return null
    const result = new THREE.BufferGeometry()
    result.setAttribute('position', new THREE.Float32BufferAttribute(selected.position, 3))
    return result
  }, [selected])
  useEffect(() => () => selectedIcon?.dispose(), [selectedIcon])
  const selectedQuaternion = selected == null
    ? null
    : new THREE.Quaternion(
        selected.qvec[1], selected.qvec[2], selected.qvec[3], selected.qvec[0],
      ).invert()
  return (
    <>
      <points geometry={geometry} renderOrder={10} raycast={() => null}>
        <pointsMaterial map={texture} size={9} sizeAttenuation={false} color="#3ad07a" transparent
          alphaTest={0.12} depthTest={false} depthWrite={false} />
      </points>
      {selectedIcon && (
        <points geometry={selectedIcon} renderOrder={13}>
          <pointsMaterial map={texture} size={11} sizeAttenuation={false} color="#ffcc33" transparent
            alphaTest={0.12} depthTest={false} depthWrite={false} />
        </points>
      )}
      {selected && selectedView && selectedQuaternion && (
        <group position={[selected.position[0], selected.position[1], selected.position[2]]}
          quaternion={selectedQuaternion}>
          <lineSegments geometry={selectedView} renderOrder={12}>
            <lineBasicMaterial color="#ffcc33" transparent opacity={0.92}
              depthTest depthWrite={false} />
          </lineSegments>
        </group>
      )}
    </>
  )
}

const typing = () => {
  const el = document.activeElement
  return !!el && (['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName)
    || el instanceof HTMLElement && el.isContentEditable)
}

const PITCH_LIMIT = Math.PI / 2 - 0.01

const FlyControls = ({ scale, center, selectedPos, onSpeedChange, active, hasData,
  cameraPose, onCameraPoseChange }: {
  scale: number
  center: THREE.Vector3
  selectedPos: THREE.Vector3 | null
  onSpeedChange: (speed: number) => void
  active: boolean
  hasData: boolean
  cameraPose: ViewerCameraPose | null
  onCameraPoseChange: (pose: ViewerCameraPose) => void
}) => {
  const { camera, gl } = useThree()
  const interaction = useRef<'look' | 'pan' | null>(null)
  const capturedPointer = useRef<number | null>(null)
  const keys = useRef<Record<string, boolean>>({})
  const angles = useRef({ yaw: 0, pitch: 0, roll: 0 })
  const speedMultiplier = useRef(1)
  const wheelAccumulator = useRef(0)
  const current = useRef({ scale, center, selectedPos, onSpeedChange, onCameraPoseChange })
  current.current = { scale, center, selectedPos, onSpeedChange, onCameraPoseChange }
  const poseTimer = useRef<number | null>(null)
  const lastEmit = useRef(0)
  const view = useRef({
    initialized: false,
    fitted: false,
    touched: false,
    disposed: false,
    saved: null as ViewerCameraPose | null,
    observed: null as ViewerCameraPose | null,
  })

  const persistPose = useCallback(() => {
    if (poseTimer.current !== null) window.clearTimeout(poseTimer.current)
    poseTimer.current = null
    if (!view.current.initialized || view.current.disposed) return
    const pose = readViewPose(camera)
    if (view.current.saved && equalViewPose(view.current.saved, pose)) return
    view.current.saved = pose
    lastEmit.current = performance.now()
    current.current.onCameraPoseChange(pose)
  }, [camera])

  useLayoutEffect(() => {
    view.current.disposed = false
    return () => {
      // Flush before the next project's layout effects reuse the same camera.
      persistPose()
      view.current.disposed = true
    }
  }, [persistPose])

  const applyOrientation = () => {
    const value = angles.current
    composeViewQuaternion(value.yaw, value.pitch, value.roll, camera.quaternion)
  }

  const readOrientation = () => {
    const value = new THREE.Euler().setFromQuaternion(camera.quaternion, 'YXZ')
    angles.current = { yaw: value.y, pitch: value.x, roll: value.z }
  }

  const applyClippingPlanes = () => {
    const clipping = sceneClippingPlanes(current.current.scale, speedMultiplier.current)
    camera.near = clipping.near
    camera.far = clipping.far
    camera.updateProjectionMatrix()
  }

  const fitTo = (tgt: THREE.Vector3, dist: number) => {
    fitInitialView(camera, tgt, dist)
    readOrientation()
  }

  useLayoutEffect(() => {
    const state = view.current
    if (!state.initialized) {
      if (cameraPose) applyViewPose(camera, cameraPose)
      else fitInitialView(camera, new THREE.Vector3(), 10)
      state.initialized = true
      state.fitted = cameraPose !== null
      state.saved = readViewPose(camera)
      state.observed = state.saved
      readOrientation()
    }
    // Server echoes and refreshed bounds must not replace the interactive view.
    if (hasData && !state.fitted && !state.touched) {
      fitTo(center, scale)
      state.fitted = true
      state.saved = readViewPose(camera)
      state.observed = state.saved
    }
    applyClippingPlanes()
  }, [camera, cameraPose, center, hasData, scale])

  useEffect(() => {
    if (!active) return
    const dom = gl.domElement
    const stopInteraction = () => {
      if (interaction.current === 'look' && document.pointerLockElement === dom)
        document.exitPointerLock()
      if (capturedPointer.current !== null && dom.hasPointerCapture?.(capturedPointer.current))
        dom.releasePointerCapture(capturedPointer.current)
      interaction.current = null
      capturedPointer.current = null
      dom.style.cursor = ''
      persistPose()
    }
    const onDown = (e: PointerEvent) => {
      if (interaction.current !== null) return
      view.current.touched = true
      dom.focus({ preventScroll: true })
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
          current.current.scale * speedMultiplier.current,
        )
      }
    }
    const onCtx = (e: Event) => e.preventDefault()
    const onAux = (e: MouseEvent) => { if (e.button === 1) e.preventDefault() }
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      view.current.touched = true
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
          wheelZoomDistance(current.current.scale, normalizedDelta, speedMultiplier.current),
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
      current.current.onSpeedChange(speedMultiplier.current)
    }
    const kd = (e: KeyboardEvent) => {
      if (typing() || dom.getClientRects().length === 0) return
      const { selectedPos, center, scale } = current.current
      if (e.code === 'KeyF') {
        view.current.touched = true
        fitTo(selectedPos ?? center, selectedPos ? scale * 0.2 : scale)
        persistPose()
        return
      }
      if (!e.repeat && (e.code === 'KeyZ' || e.code === 'KeyX')) readOrientation()
      if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.code)) e.preventDefault()
      keys.current[e.code] = true
    }
    const ku = (e: KeyboardEvent) => {
      keys.current[e.code] = false
      persistPose()
    }
    const onBlur = () => { keys.current = {}; stopInteraction() }
    const onVisibilityChange = () => { if (document.hidden) onBlur() }
    const onPointerLockChange = () => {
      if (interaction.current === 'look' && document.pointerLockElement !== dom)
        interaction.current = null
    }
    dom.addEventListener('pointerdown', onDown); window.addEventListener('pointerup', onUp)
    window.addEventListener('pointercancel', stopInteraction)
    window.addEventListener('pointermove', onMove); dom.addEventListener('contextmenu', onCtx)
    dom.addEventListener('auxclick', onAux); window.addEventListener('blur', onBlur)
    window.addEventListener('pagehide', onBlur)
    document.addEventListener('visibilitychange', onVisibilityChange)
    document.addEventListener('pointerlockchange', onPointerLockChange)
    dom.addEventListener('wheel', onWheel, { passive: false })
    window.addEventListener('keydown', kd); window.addEventListener('keyup', ku)
    return () => {
      dom.removeEventListener('pointerdown', onDown); window.removeEventListener('pointerup', onUp)
      window.removeEventListener('pointercancel', stopInteraction)
      window.removeEventListener('pointermove', onMove); dom.removeEventListener('contextmenu', onCtx)
      dom.removeEventListener('auxclick', onAux); window.removeEventListener('blur', onBlur)
      window.removeEventListener('pagehide', onBlur)
      document.removeEventListener('visibilitychange', onVisibilityChange)
      document.removeEventListener('pointerlockchange', onPointerLockChange)
      dom.removeEventListener('wheel', onWheel)
      window.removeEventListener('keydown', kd); window.removeEventListener('keyup', ku)
      keys.current = {}
      stopInteraction()
    }
  }, [active, camera, gl, persistPose])

  useFrame((_, dt) => {
    if (!active || gl.domElement.getClientRects().length === 0) return
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
    if (v.lengthSq() !== 0) {
      const speed = scale * 0.9 * speedMultiplier.current * Math.min(dt, 0.1)
        * (k.ShiftLeft || k.ShiftRight ? 3 : 1)
      v.normalize().multiplyScalar(speed).applyQuaternion(camera.quaternion)
      camera.position.add(v)
    }
    const pose = readViewPose(camera)
    if (view.current.observed && equalViewPose(view.current.observed, pose)) return
    view.current.touched = true
    view.current.observed = pose
    if (performance.now() - lastEmit.current >= 1000) persistPose()
    if (poseTimer.current !== null) window.clearTimeout(poseTimer.current)
    poseTimer.current = window.setTimeout(persistPose, 300)
  })
  return null
}

export const PointCloudViewer = ({
  projectId, recon, revision, active, preferences, onPreferencesChange, cameraPose, onCameraPoseChange,
  selectedCameraId, onPickCamera,
}: {
  projectId?: string | null
  recon?: ReconstructionData | null
  revision: number
  active: boolean
  preferences: ViewerPreferences
  onPreferencesChange: (patch: Partial<ViewerPreferences>) => void
  cameraPose: ViewerCameraPose | null
  onCameraPoseChange: (pose: ViewerCameraPose) => void
  selectedCameraId: number | null; onPickCamera: (id: number | null) => void
}) => {
  const { t } = useSettings()
  const { showPoints, showCams, pointSize, showGrid, showCenter } = preferences
  const [loaded, setLoaded] = useState<{ projectId: string; points: ParsedPoints } | null>(null)
  const points = loaded && loaded.projectId === projectId && recon ? loaded.points : null
  const [error, setError] = useState<string | null>(null)
  const [speedNotice, setSpeedNotice] = useState<{ value: number; revision: number } | null>(null)
  const speedNoticeRevision = useRef(0)
  const speedNoticeTimer = useRef<number | null>(null)
  // 背景色: 明示指定が無ければテーマの --bg に追従する (テーマ切替で更新).
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
  const bg = preferences.background ?? themeBg

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
    const controller = new AbortController()
    setError(null)
    if (!projectId || !recon) {
      setLoaded(null)
      return
    }
    fetchPoints(projectId, controller.signal)
      .then(points => {
        if (!controller.signal.aborted) setLoaded({ projectId, points })
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted) setError(String(error))
      })
    return () => controller.abort()
  }, [projectId, recon, revision])

  const scene = useMemo(() => {
    const box = new THREE.Box3(), v = new THREE.Vector3()
    if (points && points.count > 0) {
      for (let i = 0; i < points.count; i++) {
        v.fromArray(points.positions, i * 3)
        box.expandByPoint(v)
      }
    } else {
      for (const image of recon?.images ?? []) box.expandByPoint(v.fromArray(image.position))
    }
    if (box.isEmpty()) return { scale: 10, center: new THREE.Vector3(), minimumY: 0 }
    return {
      center: box.getCenter(new THREE.Vector3()),
      scale: box.getSize(new THREE.Vector3()).length() || 10,
      minimumY: box.min.y,
    }
  }, [points, recon])

  const selectedPos = useMemo(() => {
    if (selectedCameraId == null) return null
    const img = recon?.images.find(i => i.id === selectedCameraId)
    return img ? new THREE.Vector3(img.position[0], img.position[1], img.position[2]) : null
  }, [selectedCameraId, recon])

  const gridGroundY = recon?.scene_alignment?.ground?.applied ? 0 : scene.minimumY
  const backgroundColor = new THREE.Color(bg)
  const lightBackground = (
    backgroundColor.r * 0.2126 + backgroundColor.g * 0.7152 + backgroundColor.b * 0.0722
  ) > 0.55

  return (
    <div style={{ position: 'absolute', inset: 0 }} data-testid="point-cloud-viewer">
      <GlassSurface className="scene-toolbar-glass" cornerRadius={15} padding="0">
        <div className="scene-toolbar">
          <label><AppIcon name="color" size={15} /> {t('sc_bg')}
            <input type="color" value={bg}
              onChange={e => onPreferencesChange({ background: e.target.value })} />
          </label>
          <label><AppIcon name="grid" size={15} />
            <input type="checkbox" checked={showGrid}
              onChange={e => onPreferencesChange({ showGrid: e.target.checked })} /> {t('sc_grid')}
          </label>
          <label><AppIcon name="target" size={15} />
            <input type="checkbox" checked={showCenter}
              onChange={e => onPreferencesChange({ showCenter: e.target.checked })} /> {t('sc_center')}
          </label>
          <label><AppIcon name="points" size={15} /> {t('sc_points')}
            <input type="range" min={1} max={8} step={0.5} value={pointSize}
              onChange={e => onPreferencesChange({ pointSize: Number(e.target.value) })} /></label>
        </div>
      </GlassSurface>

      {speedNotice && (
        <div key={speedNotice.revision} className="scene-speed-feedback" role="status" aria-live="polite">
          {formatMovementSpeed(speedNotice.value)}
        </div>
      )}
      {error && <div style={overlayMsg} role="alert">{error}</div>}

      <Canvas flat gl={{ antialias: true }} dpr={[1, 2]} frameloop={active ? 'always' : 'demand'}
        camera={{ position: [5.5, 4, 8], near: 0.01, far: 100000, fov: 55 }}
        onCreated={({ gl }) => { gl.domElement.tabIndex = 0 }}>
        <color attach="background" args={[bg]} />
        {showPoints && points && <PointCloud points={points} size={pointSize} />}
        {showCams && recon && <CameraGizmos recon={recon} scale={scene.scale}
          selectedId={selectedCameraId} onPick={onPickCamera} />}
        {showGrid && (
          <StableGrid scale={scene.scale} center={scene.center} groundY={gridGroundY}
            light={lightBackground} />
        )}
        <axesHelper args={[scene.scale * 0.1]} />
        {showCenter && (
          <mesh position={[scene.center.x, scene.center.y, scene.center.z]}>
            <sphereGeometry args={[scene.scale * 0.01, 12, 12]} />
            <meshBasicMaterial color="#ff5577" />
          </mesh>
        )}
        <FlyControls key={projectId ?? 'empty'} scale={scene.scale} center={scene.center}
          selectedPos={selectedPos} active={active}
          hasData={(points !== null || error !== null) && (!!points?.count || !!recon?.images.length)}
          cameraPose={cameraPose} onCameraPoseChange={onCameraPoseChange}
          onSpeedChange={showSpeedNotice} />
        <GizmoHelper alignment="top-right" margin={[56, 92]}>
          <GizmoViewport axisColors={['#ff5566', '#4caf50', '#4488ff']} labelColor="#fff" />
        </GizmoHelper>
      </Canvas>
    </div>
  )
}

const overlayMsg: React.CSSProperties = {
  position: 'absolute', left: 12, right: 12, bottom: 12, zIndex: 2, pointerEvents: 'none',
  color: 'var(--danger)', fontFamily: 'ui-monospace, monospace', fontSize: 12,
}
