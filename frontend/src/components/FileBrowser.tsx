import { useEffect, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api, type FsEntry } from '../api/client'
import { useSettings } from '../ui/settings'
import { formatPathForDisplay, PathText } from './PathText'

// ソース選択 (モーダル). ドライブ選択 + フォルダを辿る. 行全体クリックで下階層へ.
export const FileBrowser = ({
  onPick,
  onClose,
  selectionError,
}: {
  onPick: (path: string, isDir: boolean) => void
  onClose: () => void
  selectionError?: unknown
}) => {
  const { t } = useSettings()
  const { data: drivesData } = useQuery({ queryKey: ['fs-drives'], queryFn: api.getFsDrives, retry: false })
  const [cwd, setCwd] = useState<string | null>(null)

  useEffect(() => {
    if (cwd === null && drivesData?.drives?.length) setCwd(drivesData.drives[0])
  }, [drivesData, cwd])
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [onClose])

  const { data, error, isFetching } = useQuery({
    queryKey: ['fs-browse', cwd],
    queryFn: () => api.browseFs(cwd as string),
    enabled: !!cwd,
    retry: false,
  })

  return (
    <div className="modal-back" onClick={onClose}>
      <div className="modal" style={{ height: 520, width: 640 }} onClick={e => e.stopPropagation()}
        role="dialog" aria-modal="true" aria-labelledby="file-browser-title">
        <div className="modal-head">
          <strong id="file-browser-title" style={{ flex: 'none' }}>{t('selectSource')}</strong>
          {drivesData?.drives.map(d => (
            <button key={d} className={`btn ${cwd?.startsWith(d) ? '' : 'btn-secondary'}`}
              style={{ padding: '4px 10px', fontSize: 12 }} onClick={() => setCwd(d)}>{formatPathForDisplay(d)}</button>
          ))}
          {cwd && <PathText path={cwd} compact className="file-browser-path" />}
          <button type="button" className="btn btn-secondary" onClick={onClose}
            aria-label={t('close')} autoFocus>×</button>
        </div>
        <div className="modal-list">
          {selectionError != null && (
            <div className="error" style={{ padding: 8 }}>{String(selectionError)}</div>
          )}
          {error && <div className="error" style={{ padding: 8 }}>{String(error)}</div>}
          {isFetching && <div className="mono" style={{ padding: 8 }}>...</div>}
          {data?.parent && (
            <button type="button" className="fs-row fs-row-button" onClick={() => setCwd(data.parent)}>
              <span>📁</span> <span>..</span>
            </button>
          )}
          {data?.dirs.map(d => (
            <div key={d.path} className="fs-row">
              <button type="button" className="fs-main" onClick={() => setCwd(d.path)}>
                <span>📁</span>
                <span style={{ flex: 1 }}>{d.name}</span>
              </button>
              <button className="btn btn-secondary" style={{ fontSize: 10, padding: '2px 6px' }}
                type="button" onClick={() => onPick(d.path, true)}>{t('selectErpImages')}</button>
            </div>
          ))}
          {data?.files.map((f: FsEntry) => (
            <button type="button" key={f.path} className="fs-row fs-row-button" onClick={() => onPick(f.path, false)}>
              <span>🎬</span>
              <span style={{ flex: 1 }}>{f.name}</span>
              <span className="mono">{f.size != null ? `${(f.size / 1e9).toFixed(2)} GB` : ''}</span>
            </button>
          ))}
          {data && !data.dirs.length && !data.files.length && (
            <div className="mono" style={{ color: 'var(--fg-mute)', padding: 10 }}>—</div>
          )}
        </div>
      </div>
    </div>
  )
}
