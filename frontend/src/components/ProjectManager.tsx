import { useMemo, useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { api, type Project } from '../api/client'
import { useSettings } from '../ui/settings'

// Unity Hub 風のプロジェクト管理ウィンドウ: 一覧 (更新時刻降順) / 新規作成 (名前入力) /
// 削除 (ディスクから完全削除, ソース動画は消さない, 確認あり).
export const ProjectManager = ({
  projects, currentId, onSelect, onClose,
}: {
  projects: Project[] | undefined
  currentId: string | null
  onSelect: (id: string) => void
  onClose: () => void
}) => {
  const { t } = useSettings()
  const qc = useQueryClient()
  const [name, setName] = useState('')
  const [confirmDel, setConfirmDel] = useState<string | null>(null)

  const sorted = useMemo(
    () => [...(projects ?? [])].sort((a, b) => (b.updated_at || '').localeCompare(a.updated_at || '')),
    [projects],
  )

  const create = useMutation({
    mutationFn: () => api.createProject(name.trim() || `project-${new Date().toISOString().slice(0, 16)}`),
    onSuccess: p => { qc.invalidateQueries({ queryKey: ['projects'] }); onSelect(p.id); setName(''); onClose() },
  })
  const del = useMutation({
    mutationFn: (id: string) => api.deleteProject(id),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['projects'] }); setConfirmDel(null) },
  })

  return (
    <div className="modal-back" onClick={onClose}>
      <div className="modal" style={{ height: 560, width: 680 }} onClick={e => e.stopPropagation()}>
        <div className="modal-head">
          <strong style={{ flex: 1 }}>{t('projects')}</strong>
          <button className="btn btn-secondary" onClick={onClose}>×</button>
        </div>
        <div style={{ display: 'flex', gap: 8, padding: '8px 12px', borderBottom: '1px solid var(--border)' }}>
          <input className="input" style={{ flex: 1, minWidth: 0, width: 'auto' }} placeholder={t('projectName')} value={name}
            onChange={e => setName(e.target.value)} onKeyDown={e => e.key === 'Enter' && create.mutate()} />
          <button className="btn" disabled={create.isPending} onClick={() => create.mutate()}>{t('createProject')}</button>
        </div>
        <div className="modal-list" style={{ flex: 1 }}>
          {sorted.length === 0 && <div className="mono" style={{ padding: 12 }}>—</div>}
          {sorted.map(p => (
            <div key={p.id} className={`fs-row${p.id === currentId ? ' sel-row' : ''}`}
              style={{ justifyContent: 'space-between', cursor: 'default' }}>
              <div style={{ flex: 1, minWidth: 0 }} onClick={() => { onSelect(p.id); onClose() }} role="button">
                <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.name}</div>
                <div className="mono" style={{ fontSize: 10 }}>{p.state} · {fmtTime(p.updated_at)}</div>
              </div>
              {confirmDel === p.id ? (
                <span style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                  <span className="mono" style={{ color: 'var(--error)', fontSize: 11 }}>{t('deleteConfirm')}</span>
                  <button className="btn" style={{ background: 'var(--error)', padding: '4px 10px' }}
                    disabled={del.isPending} onClick={() => del.mutate(p.id)}>{t('delete')}</button>
                  <button className="btn btn-secondary" style={{ padding: '4px 10px' }} onClick={() => setConfirmDel(null)}>{t('cancel')}</button>
                </span>
              ) : (
                <span style={{ display: 'flex', gap: 6 }}>
                  <button className="btn btn-secondary" style={{ padding: '4px 10px' }} onClick={() => { onSelect(p.id); onClose() }}>{t('open')}</button>
                  <button className="btn btn-secondary" style={{ padding: '4px 10px', color: 'var(--error)', borderColor: 'var(--error)' }}
                    onClick={() => setConfirmDel(p.id)}>{t('delete')}</button>
                </span>
              )}
            </div>
          ))}
        </div>
        {del.error && <div className="error" style={{ padding: 8 }}>{String(del.error)}</div>}
      </div>
    </div>
  )
}

const fmtTime = (iso: string) => {
  try { return new Date(iso).toLocaleString() } catch { return iso }
}
