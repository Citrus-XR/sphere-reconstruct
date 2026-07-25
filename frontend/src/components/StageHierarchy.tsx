// 左 hierarchy: 工程一覧をバッジ + (任意で) 有効チェックボックス付きで並べる. クリックで選択.
// items は App 側で翻訳・状態解決済みにして渡す (このコンポーネントは描画のみ).
import { ProgressRing } from './ProgressRing'

export interface HierItem {
  key: string
  label: string
  badgeColor: string
  statusLabel: string
  toggleable: boolean
  enabled: boolean
  dim?: boolean
  running?: boolean
  progress?: number  // 実行中の進捗 0..1 (環形表示用)
}

export const StageHierarchy = ({
  items,
  selected,
  onSelect,
}: {
  items: HierItem[]
  selected: string | null
  onSelect: (key: string) => void
}) => (
  <div>
    {items.map(it => {
      const off = it.toggleable && !it.enabled
      return (
        <div key={it.key} className={`hier-item${selected === it.key ? ' sel' : ''}${off ? ' disabled' : ''}`}
          style={it.dim && !off ? { opacity: 0.4 } : undefined} onClick={() => onSelect(it.key)}>
          {it.running
            ? <ProgressRing value={it.progress ?? 0} size={14} stroke={2.5} />
            : <span className="hier-badge" style={{ background: it.badgeColor }} />}
          <span style={{ flex: 1 }}>{it.label}</span>
          <span className="mono" style={{ fontSize: 10, color: 'var(--fg-mute)' }}>
            {off ? 'skip' : it.running ? `${Math.round((it.progress ?? 0) * 100)}%` : it.statusLabel}
          </span>
        </div>
      )
    })}
  </div>
)
