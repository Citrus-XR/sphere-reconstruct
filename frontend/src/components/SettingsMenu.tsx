import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'
import { useSettings } from '../ui/settings'
import type { Lang } from '../ui/i18n'
import type { Theme } from '../ui/settings'

// 右上の設定メニュー: テーマ (自動/ライト/ダーク) + 言語. ブラウザに保存.
export const SettingsMenu = () => {
  const { theme, setTheme, lang, setLang, t } = useSettings()
  const [open, setOpen] = useState(false)
  const { data: doctor } = useQuery({
    queryKey: ['doctor'], queryFn: api.getDoctor, enabled: open, staleTime: 30_000,
  })
  return (
    <div style={{ position: 'relative' }}>
      <button className="btn btn-secondary" onClick={() => setOpen(o => !o)} title={t('settings')}>⚙</button>
      {open && (
        <>
          <div style={{ position: 'fixed', inset: 0, zIndex: 199 }} onClick={() => setOpen(false)} />
          <div className="pop">
            <div className="ctl">
              <label>{t('theme')}</label>
              <select className="input" value={theme} onChange={e => setTheme(e.target.value as Theme)}>
                <option value="auto">{t('themeAuto')}</option>
                <option value="light">{t('themeLight')}</option>
                <option value="dark">{t('themeDark')}</option>
              </select>
            </div>
            <div className="ctl" style={{ marginBottom: 0 }}>
              <label>{t('language')}</label>
              <select className="input" value={lang} onChange={e => setLang(e.target.value as Lang)}>
                <option value="ja">日本語</option>
                <option value="zh">中文</option>
                <option value="en">English</option>
              </select>
            </div>
            {doctor && <div className="ctl" style={{ marginTop: 10, marginBottom: 0, minWidth: 280 }}>
              <label>Environment {doctor.ready ? '✓' : '⚠'}</label>
              {Object.entries(doctor.checks).map(([name, check]) => (
                <div key={name} className="mono" style={{ fontSize: 10, color: check.ok ? '#4caf50' : check.optional ? '#d69a2a' : 'var(--error)' }}>
                  {check.ok ? '✓' : check.optional ? '○' : '✗'} {name}: {check.message}
                </div>
              ))}
            </div>}
          </div>
        </>
      )}
    </div>
  )
}
