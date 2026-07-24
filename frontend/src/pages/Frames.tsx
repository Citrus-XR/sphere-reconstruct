import { useState } from 'react'
import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { api, frameImageUrl } from '../api/client'

// 抽出された前後レンズ fisheye フレームの一覧とプレビュー.
export const FramesPage = () => {
  const { id = '' } = useParams<{ id: string }>()
  const { data, error, isLoading } = useQuery({
    queryKey: ['frames', id],
    queryFn: () => api.getFrames(id),
    enabled: !!id,
    retry: false,
  })
  const [sel, setSel] = useState(0)

  if (isLoading) return <div className="card">読み込み中...</div>
  if (error)
    return (
      <div className="card mono">
        抽出フレームがまだありません. extract_frames ステージまで実行してください.
      </div>
    )
  if (!data || data.frames.length === 0) return <div className="card">フレームが空です.</div>

  const isDual = data.kind === 'insv_dual'
  const current = data.frames[Math.min(sel, data.frames.length - 1)]

  return (
    <div className="card">
      <h2>抽出フレーム</h2>
      <div className="mono" style={{ marginBottom: 8 }}>
        {data.kind} · {data.count} frames · {data.width}x{data.height}
        {data.fps ? ` · ${data.fps}fps` : ''}
      </div>

      <div
        style={{
          display: 'flex',
          gap: 4,
          overflowX: 'auto',
          padding: 4,
          marginBottom: 12,
          borderBottom: '1px solid var(--border)',
        }}
      >
        {data.frames.map((f, i) => (
          <button
            key={f.index}
            className="mono"
            onClick={() => setSel(i)}
            style={{
              flex: '0 0 auto',
              padding: '4px 8px',
              border: '1px solid var(--border)',
              borderRadius: 4,
              background: i === sel ? 'var(--accent)' : 'transparent',
              color: i === sel ? 'white' : 'var(--fg-mute)',
              cursor: 'pointer',
            }}
          >
            {f.index}
          </button>
        ))}
      </div>

      <div className="mono" style={{ marginBottom: 8 }}>
        frame {current.index}
        {current.timestamp_sec != null ? ` · t=${current.timestamp_sec.toFixed(2)}s` : ''}
      </div>

      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
        <figure style={{ margin: 0 }}>
          <img
            src={frameImageUrl(id, current.index, 0)}
            style={{ maxWidth: 360, borderRadius: 8, background: '#111' }}
            alt="lens0"
          />
          <figcaption className="mono">{isDual ? 'lens0 (front)' : 'ERP'}</figcaption>
        </figure>
        {isDual && (
          <figure style={{ margin: 0 }}>
            <img
              src={frameImageUrl(id, current.index, 1)}
              style={{ maxWidth: 360, borderRadius: 8, background: '#111' }}
              alt="lens1"
            />
            <figcaption className="mono">lens1 (back)</figcaption>
          </figure>
        )}
      </div>
    </div>
  )
}
