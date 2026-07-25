import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'

// 使用率ゲージ群 (下部バー右側). 環形 (省スペース) で CPU / RAM / GPU util / VRAM を出す.
// 1.5s ポーリング. GPU 名はホバー時のみ表示する.
export const SystemStatsBar = () => {
  const { data } = useQuery({
    queryKey: ['system-stats'],
    queryFn: api.getSystemStats,
    refetchInterval: 1500,
    retry: false,
  })
  const cpu = data?.cpu_percent
  const ram = data?.ram
  return (
    <div className="stats-right">
      <Ring label="CPU" pct={cpu ?? null} text={cpu == null ? 'n/a' : `${cpu.toFixed(0)}%`} />
      <Ring label="RAM" pct={ram?.percent ?? null}
        text={ram ? `${(ram.used_mb / 1024).toFixed(1)}/${(ram.total_mb / 1024).toFixed(0)}G` : 'n/a'}
        title={ram ? `RAM ${ram.percent.toFixed(0)}%` : undefined} />
      {(data?.gpus ?? []).map((g, i) => (
        <span key={i} className="gpu-group" title={g.name}>
          <Ring label="GPU" pct={g.util_percent} text={g.util_percent == null ? 'n/a' : `${g.util_percent.toFixed(0)}%`} />
          <Ring label="VRAM" pct={g.mem_used_mb != null && g.mem_total_mb ? (g.mem_used_mb / g.mem_total_mb) * 100 : null}
            text={g.mem_used_mb != null && g.mem_total_mb
              ? `${(g.mem_used_mb / 1024).toFixed(1)}/${(g.mem_total_mb / 1024).toFixed(0)}G` : 'n/a'} />
        </span>
      ))}
      {!data?.gpus?.length && <span className="mono" style={{ color: 'var(--fg-mute)' }}>GPU n/a</span>}
    </div>
  )
}

const Ring = ({ label, pct, text, title }: { label: string; pct: number | null; text: string; title?: string }) => {
  const r = 8
  const circ = 2 * Math.PI * r
  const p = Math.max(0, Math.min(100, pct ?? 0))
  return (
    <span className="ring-gauge" title={title ?? `${label} ${text}`}>
      <svg width="22" height="22" viewBox="0 0 22 22">
        <circle cx="11" cy="11" r={r} fill="none" stroke="var(--border)" strokeWidth="2.5" />
        <circle cx="11" cy="11" r={r} fill="none" stroke={gaugeColor(pct)} strokeWidth="2.5"
          strokeDasharray={circ} strokeDashoffset={circ * (1 - p / 100)} strokeLinecap="round"
          transform="rotate(-90 11 11)" />
      </svg>
      <span className="ring-text"><b>{label}</b><span className="mono">{text}</span></span>
    </span>
  )
}

const gaugeColor = (pct: number | null) => {
  if (pct == null) return 'var(--border)'
  if (pct > 90) return 'var(--error)'
  if (pct > 60) return '#d69a2a'
  return 'var(--accent)'
}
