import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api, openEventStream, type EventEnvelope, type SourceKind } from '../api/client'

// V1 では Source ページと Detail は同じで良い. Phase 3 以降で分離する.
export const ProjectDetailPage = () => {
  const { id = '' } = useParams<{ id: string }>()
  const qc = useQueryClient()

  const { data: project, error } = useQuery({
    queryKey: ['projects', id],
    queryFn: () => api.getProject(id),
    enabled: !!id,
    refetchInterval: 2000,
  })

  const [kind, setKind] = useState<SourceKind>('insv')
  const [path, setPath] = useState('')
  const setSource = useMutation({
    mutationFn: () => api.setSource(id, kind, path),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['projects', id] }),
  })
  const run = useMutation({
    mutationFn: () => api.runPipeline(id),
    onSuccess: r => setActiveJobId(r.job_id),
  })

  const [activeJobId, setActiveJobId] = useState<string | null>(null)
  const [events, setEvents] = useState<EventEnvelope[]>([])

  // 指定プロジェクトの event を live で受け取る. project 変更で購読しなおす.
  useEffect(() => {
    if (!id) return
    setEvents([])
    const ws = openEventStream({ projectId: id }, e => {
      setEvents(prev => [...prev.slice(-499), e])
    })
    return () => ws.close()
  }, [id])

  if (error) return <div className="error">{String(error)}</div>
  if (!project) return <div>読み込み中...</div>

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="card">
        <h2>{project.name}</h2>
        <div className="mono">id: {project.id}</div>
        <div className="mono">state: {project.state}</div>
        <div className="mono">source: {project.source_kind ?? '(未設定)'} {project.source_path ?? ''}</div>
        <div style={{ marginTop: 8, display: 'flex', gap: 16 }}>
          <Link to={`/projects/${project.id}/masks`}>マスク確認 →</Link>
          <Link to={`/projects/${project.id}/reconstruction`}>再構成ビューア →</Link>
        </div>
      </div>

      <div className="card">
        <h3>ソース設定</h3>
        <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
          <select
            className="input"
            style={{ maxWidth: 200 }}
            value={kind}
            onChange={e => setKind(e.target.value as SourceKind)}
          >
            <option value="insv">INSV (Insta360 X5)</option>
            <option value="erp_video">Equirectangular 動画</option>
            <option value="erp_images">Equirectangular 画像フォルダ</option>
          </select>
          <input
            className="input"
            placeholder="ソースの絶対パス (allowed_roots 内)"
            value={path}
            onChange={e => setPath(e.target.value)}
          />
          <button
            className="btn"
            disabled={!path.trim() || setSource.isPending}
            onClick={() => setSource.mutate()}
          >
            設定
          </button>
        </div>
        {setSource.error && <div className="error">{String(setSource.error)}</div>}
      </div>

      <div className="card">
        <h3>実行</h3>
        <button
          className="btn"
          disabled={!project.source_path || run.isPending}
          onClick={() => run.mutate()}
        >
          パイプライン実行
        </button>
        {run.error && <div className="error">{String(run.error)}</div>}
        {activeJobId && <div className="mono">job: {activeJobId}</div>}
      </div>

      <div className="card">
        <h3>ログ (live)</h3>
        <div
          className="mono"
          style={{ maxHeight: 400, overflow: 'auto', background: 'var(--bg)', padding: 8 }}
        >
          {events.length === 0 && <div>(未受信)</div>}
          {events.map(e => (
            <div key={e.id}>
              <span style={{ color: levelColor(e.level) }}>
                [{e.level}]
              </span>{' '}
              {e.stage ? `<${e.stage}> ` : ''}
              {e.message}
              {e.progress != null ? ` (${Math.round(e.progress * 100)}%)` : ''}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

const levelColor = (l: string) => {
  switch (l) {
    case 'error':
      return 'var(--error)'
    case 'warn':
      return '#d69a2a'
    case 'debug':
      return 'var(--fg-mute)'
    default:
      return 'var(--accent)'
  }
}
