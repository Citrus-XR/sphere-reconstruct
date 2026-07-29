import LiquidGlass from 'liquid-glass-react'
import type { CSSProperties, ReactNode } from 'react'

export const GlassSurface = ({
  children,
  className = '',
  cornerRadius = 16,
  padding = '4px',
  style,
}: {
  children: ReactNode
  className?: string
  cornerRadius?: number
  padding?: string
  style?: CSSProperties
}) => (
  <div className={`glass-surface-host ${className}`.trim()} style={style}>
    <LiquidGlass
      className="liquid-glass-background"
      mode="standard"
      displacementScale={28}
      blurAmount={0.09}
      saturation={125}
      aberrationIntensity={0.8}
      elasticity={0}
      cornerRadius={cornerRadius}
      padding="0"
      style={{ position: 'absolute', top: '50%', left: '50%', width: '100%', height: '100%', pointerEvents: 'none' }}
    >
      <span aria-hidden="true" />
    </LiquidGlass>
    <div className="glass-surface-content" style={{ padding }}>{children}</div>
  </div>
)
