import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'

export const SettingsPage = () => {
  const { data, error, isLoading } = useQuery({
    queryKey: ['settings'],
    queryFn: api.getSettings,
  })

  return (
    <div className="card">
      <h2>設定</h2>
      <p className="mono">
        Phase 1 段階では read-only. runtime/config.toml を編集して backend を再起動してください.
      </p>
      {isLoading && <div>読み込み中...</div>}
      {error && <div className="error">{String(error)}</div>}
      {data && (
        <pre className="mono" style={{ overflow: 'auto' }}>
          {JSON.stringify(data, null, 2)}
        </pre>
      )}
    </div>
  )
}
