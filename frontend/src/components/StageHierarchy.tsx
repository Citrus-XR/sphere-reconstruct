// 左 hierarchy: 工程一覧をバッジ + (任意で) 有効チェックボックス付きで並べる. クリックで選択.
// items は App 側で翻訳・状態解決済みにして渡す (このコンポーネントは描画のみ).
import { ProgressRing } from './ProgressRing'
import { useSettings } from '../ui/settings'

export interface HierItem {
  key: string
  label: string
  badgeColor: string
  statusLabel: string
  toggleable: boolean
  enabled: boolean
  dim?: boolean
  running?: boolean
  progress?: number | null  // null = 作業中だが総量不明.
  detail?: string
}

export const StageHierarchy = ({
  items,
  selected,
  onSelect,
}: {
  items: HierItem[]
  selected: string | null
  onSelect: (key: string) => void
}) => {
  const { t } = useSettings()
  return <div style={{ height: '100%', overflowY: 'auto' }}>
    {items.map(it => {
      const off = it.toggleable && !it.enabled
      const trailingStatus = off ? t('skip') : it.running
        ? it.progress == null ? t('working') : `${Math.round(it.progress * 100)}%`
        : it.statusLabel
      return (
        <button type="button" key={it.key}
          className={`hier-item hier-row-button${selected === it.key ? ' sel' : ''}${off ? ' disabled' : ''}`}
          style={it.dim && !off ? { opacity: 0.4 } : undefined} onClick={() => onSelect(it.key)}
          aria-label={`${it.label} ${trailingStatus}`}
          aria-current={selected === it.key ? 'step' : undefined}>
          {it.running
            ? <ProgressRing value={it.progress ?? null} size={14} stroke={2.5} label={it.label} />
            : <span className="hier-badge" style={{ background: it.badgeColor }} />}
          <span className="hier-step-content">
            <span>{it.label}</span>
            {it.detail && <span className="hier-step-detail" title={it.detail}>{it.detail}</span>}
          </span>
          <span className="mono" style={{ fontSize: 10, color: 'var(--fg-mute)' }}>
            {trailingStatus}
          </span>
        </button>
      )
    })}
  </div>
}
