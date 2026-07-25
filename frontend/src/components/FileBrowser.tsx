import { useEffect, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api, type FsEntry } from '../api/client'
import { useSettings } from '../ui/settings'

// ソース選択 (モーダル). ドライブ選択 + フォルダを辿る. 行全体クリックで下階層へ.
export const FileBrowser = ({
  onPick,
  onClose,
}: {
  onPick: (path: string, isDir: boolean) => void
  onClose: () => void
}) => {
  const { t } = useSettings()
  const { data: drivesData } = useQuery({ queryKey: ['fs-drives'], queryFn: api.getFsDrives, retry: false })
  const [cwd, setCwd] = useState<string | null>(null)

  useEffect(() => {
    if (cwd === null && drivesData?.drives?.length) setCwd(drivesData.drives[0])
  }, [drivesData, cwd])

  const { data, error, isFetching } = useQuery({
    queryKey: ['fs-browse', cwd],
    queryFn: () => api.browseFs(cwd as string),
    enabled: !!cwd,
    retry: false,
  })

  return (
    <div className="modal-back" onClick={onClose}>
      <div className="modal" style={{ height: 520, width: 640 }} onClick={e => e.stopPropagation()}>
        <div className="modal-head">
          <strong style={{ flex: 'none' }}>{t('selectSource')}</strong>
          {drivesData?.drives.map(d => (
            <button key={d} className={`btn ${cwd?.startsWith(d) ? '' : 'btn-secondary'}`}
              style={{ padding: '4px 10px', fontSize: 12 }} onClick={() => setCwd(d)}>{d}</button>
          ))}
          <span className="mono" style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{cwd}</span>
          <button className="btn btn-secondary" onClick={onClose}>×</button>
        </div>
        <div className="modal-list">
          {error && <div className="error" style={{ padding: 8 }}>{String(error)}</div>}
          {isFetching && <div className="mono" style={{ padding: 8 }}>...</div>}
          {data?.parent && (
            <div className="fs-row" onClick={() => setCwd(data.parent)}>
              <span>📁</span> <span>..</span>
            </div>
          )}
          {data?.dirs.map(d => (
            <div key={d.path} className="fs-row" onClick={() => setCwd(d.path)}>
              <span>📁</span>
              <span style={{ flex: 1 }}>{d.name}</span>
              <button className="btn btn-secondary" style={{ fontSize: 10, padding: '2px 6px' }}
                onClick={e => { e.stopPropagation(); onPick(d.path, true) }}>ERP画像として選択</button>
            </div>
          ))}
          {data?.files.map((f: FsEntry) => (
            <div key={f.path} className="fs-row" onClick={() => onPick(f.path, false)}>
              <span>🎬</span>
              <span style={{ flex: 1 }}>{f.name}</span>
              <span className="mono">{f.size != null ? `${(f.size / 1e9).toFixed(2)} GB` : ''}</span>
            </div>
          ))}
          {data && !data.dirs.length && !data.files.length && (
            <div className="mono" style={{ color: 'var(--fg-mute)', padding: 10 }}>—</div>
          )}
        </div>
      </div>
    </div>
  )
}

