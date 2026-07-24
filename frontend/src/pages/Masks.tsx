import { useMemo, useState } from 'react'
import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { api, maskUrl, pinholeUrl, type MaskViewRecord } from '../api/client'

type Mode = 'image' | 'mask' | 'overlay'

// pinhole 画像とその SAM3 mask を確認するページ.
// original / mask / overlay を切り替え, frame と view を選ぶ.
export const MasksPage = () => {
  const { id = '' } = useParams<{ id: string }>()
  const { data, error, isLoading } = useQuery({
    queryKey: ['masks', id],
    queryFn: () => api.getMasks(id),
    enabled: !!id,
    retry: false,
  })

  const [frameIdx, setFrameIdx] = useState(0)
  const [view, setView] = useState<{ view: string; lens: number } | null>(null)
  const [mode, setMode] = useState<Mode>('overlay')

  const frames = data?.frames ?? []
  const current = frames[Math.min(frameIdx, frames.length - 1)]
  const views = current?.views ?? []
  const sel: MaskViewRecord | undefined = useMemo(() => {
    if (!current) return undefined
    if (view) return current.views.find(v => v.view === view.view && v.lens === view.lens)
    return current.views[0]
  }, [current, view])

  if (isLoading) return <div className="card">読み込み中...</div>
  if (error)
    return (
      <div className="card mono">
        マスクがまだありません. generate_masks ステージまで実行してください.
      </div>
    )
  if (!current || !sel) return <div className="card">マスクデータが空です.</div>

  return (
    <div className="card">
      <h2>マスク確認</h2>
      <div className="mono" style={{ marginBottom: 8 }}>
        prompt: {data?.prompt.join(', ')}
      </div>

      <div style={{ display: 'flex', gap: 8, marginBottom: 8, flexWrap: 'wrap' }}>
        <button
          className="btn-secondary btn"
          disabled={frameIdx <= 0}
          onClick={() => setFrameIdx(i => Math.max(0, i - 1))}
        >
          ← 前フレーム
        </button>
        <span className="mono" style={{ alignSelf: 'center' }}>
          frame {current.index} ({frameIdx + 1}/{frames.length})
        </span>
        <button
          className="btn-secondary btn"
          disabled={frameIdx >= frames.length - 1}
          onClick={() => setFrameIdx(i => Math.min(frames.length - 1, i + 1))}
        >
          次フレーム →
        </button>
        <select
          className="input"
          style={{ maxWidth: 200 }}
          value={sel ? `${sel.view}:${sel.lens}` : ''}
          onChange={e => {
            const [v, l] = e.target.value.split(':')
            setView({ view: v, lens: Number(l) })
          }}
        >
          {views.map(v => (
            <option key={`${v.view}:${v.lens}`} value={`${v.view}:${v.lens}`}>
              {v.view} lens{v.lens} (cov {(v.coverage * 100).toFixed(1)}%)
            </option>
          ))}
        </select>
        {(['image', 'mask', 'overlay'] as Mode[]).map(m => (
          <button
            key={m}
            className="btn"
            style={{ opacity: mode === m ? 1 : 0.5 }}
            onClick={() => setMode(m)}
          >
            {m}
          </button>
        ))}
      </div>

      <div className="mono" style={{ marginBottom: 8 }}>
        coverage {(sel.coverage * 100).toFixed(1)}% · hits{' '}
        {Object.entries(sel.detections)
          .filter(([, n]) => n > 0)
          .map(([k, n]) => `${k}:${n}`)
          .join(', ') || '(none)'}
        {sel.coverage_warning && <span className="error"> ⚠ coverage high</span>}
      </div>

      <div
        style={{
          position: 'relative',
          maxWidth: 640,
          background: '#111',
          borderRadius: 8,
          overflow: 'hidden',
        }}
      >
        {(mode === 'image' || mode === 'overlay') && (
          <img
            src={pinholeUrl(id, current.index, sel.view, sel.lens)}
            style={{ width: '100%', display: 'block' }}
            alt="pinhole"
          />
        )}
        {(mode === 'mask' || mode === 'overlay') && (
          <img
            src={maskUrl(id, current.index, sel.view, sel.lens)}
            style={{
              width: '100%',
              display: 'block',
              position: mode === 'overlay' ? 'absolute' : 'static',
              top: 0,
              left: 0,
              opacity: mode === 'overlay' ? 0.5 : 1,
              mixBlendMode: mode === 'overlay' ? 'multiply' : 'normal',
            }}
            alt="mask"
          />
        )}
      </div>
    </div>
  )
}
