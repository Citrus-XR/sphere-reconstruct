// 処理進捗の環形インジケータ. Step 一覧と Inspector で共用する.
export const ProgressRing = ({
  value, size = 16, stroke = 2.5, color = 'var(--accent)', showLabel = false, label = 'Progress',
}: {
  value: number | null
  size?: number
  stroke?: number
  color?: string
  showLabel?: boolean
  label?: string
}) => {
  const r = (size - stroke) / 2
  const circ = 2 * Math.PI * r
  const pct = value === null ? null : Math.max(0, Math.min(1, value))
  const arc = (
    <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={color} strokeWidth={stroke}
      strokeDasharray={pct === null ? `${circ * 0.25} ${circ * 0.75}` : circ}
      strokeDashoffset={pct === null ? 0 : circ * (1 - pct)} strokeLinecap="round"
      transform={`rotate(-90 ${size / 2} ${size / 2})`} />
  )
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }} role="progressbar"
      aria-label={label} aria-valuemin={0} aria-valuemax={100}
      aria-valuenow={pct === null ? undefined : Math.round(pct * 100)}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}
        style={{ display: 'block', flex: 'none' }}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--border)" strokeWidth={stroke} />
        {pct === null ? <g className="progress-ring-indeterminate">{arc}</g> : arc}
      </svg>
      {showLabel && <span className="mono" style={{ fontSize: 10, color: 'var(--fg-mute)' }}>
        {pct === null ? '…' : `${Math.round(pct * 100)}%`}
      </span>}
    </span>
  )
}
