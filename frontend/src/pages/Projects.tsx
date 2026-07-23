import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { api } from '../api/client'

export const ProjectsPage = () => {
  const qc = useQueryClient()
  const { data, isLoading, error } = useQuery({
    queryKey: ['projects'],
    queryFn: api.listProjects,
  })
  const [name, setName] = useState('')
  const create = useMutation({
    mutationFn: (n: string) => api.createProject(n),
    onSuccess: () => {
      setName('')
      qc.invalidateQueries({ queryKey: ['projects'] })
    },
  })

  return (
    <div className="card">
      <h2>プロジェクト一覧</h2>
      <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
        <input
          className="input"
          placeholder="新規プロジェクト名"
          value={name}
          onChange={e => setName(e.target.value)}
        />
        <button
          className="btn"
          disabled={!name.trim() || create.isPending}
          onClick={() => create.mutate(name.trim())}
        >
          作成
        </button>
      </div>
      {isLoading && <div>読み込み中...</div>}
      {error && <div className="error">{String(error)}</div>}
      {data && data.length === 0 && <div className="mono">まだプロジェクトはありません.</div>}
      {data?.map(p => (
        <div key={p.id} className="list-row">
          <div style={{ flex: 1 }}>
            <Link to={`/projects/${p.id}`}>{p.name}</Link>
            <div className="mono">{p.id}</div>
          </div>
          <div className="mono">state: {p.state}</div>
        </div>
      ))}
    </div>
  )
}
