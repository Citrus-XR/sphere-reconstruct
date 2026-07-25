// 処理進捗の環形インジケータ. Step 一覧と Inspector で共用する.
export const ProgressRing = ({
  value, size = 16, stroke = 2.5, color = 'var(--accent)', showLabel = false,
}: {
  value: number
  size?: number
  stroke?: number
  color?: string
  showLabel?: boolean
}) => {
  const r = (size - stroke) / 2
  const circ = 2 * Math.PI * r
  const pct = Math.max(0, Math.min(1, value))
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
      <svg width={size} height={size} style={{ display: 'block', flex: 'none' }}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--border)" strokeWidth={stroke} />
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={color} strokeWidth={stroke}
          strokeDasharray={circ} strokeDashoffset={circ * (1 - pct)} strokeLinecap="round"
          transform={`rotate(-90 ${size / 2} ${size / 2})`} />
      </svg>
      {showLabel && <span className="mono" style={{ fontSize: 10, color: 'var(--fg-mute)' }}>{Math.round(pct * 100)}%</span>}
    </span>
  )
}
