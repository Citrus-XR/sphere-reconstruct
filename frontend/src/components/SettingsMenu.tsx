import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'
import { useSettings } from '../ui/settings'
import type { Lang } from '../ui/i18n'
import type { Theme } from '../ui/settings'
import { PathText } from './PathText'
import { AppIcon } from './AppIcon'

// 右上の設定メニュー: テーマ (自動/ライト/ダーク) + 言語. ブラウザに保存.
export const SettingsMenu = () => {
  const { theme, setTheme, lang, setLang, t } = useSettings()
  const [open, setOpen] = useState(false)
  const { data: doctor } = useQuery({
    queryKey: ['doctor'], queryFn: api.getDoctor, enabled: open, staleTime: 30_000,
  })
  return (
    <div style={{ position: 'relative' }}>
      <button className="btn btn-secondary icon-only" onClick={() => setOpen(o => !o)}
        title={t('settings')} aria-label={t('settings')}>
        <AppIcon name="settings" />
      </button>
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
            {doctor && <div className="ctl environment-check" style={{ marginTop: 10, marginBottom: 0 }}>
              <label>{t('environment')} {doctor.ready ? '✓' : '⚠'}</label>
              {Object.entries(doctor.checks).map(([name, check]) => (
                <div key={name} style={{ marginBottom: 4, color: check.ok ? '#4caf50' : check.optional ? '#d69a2a' : 'var(--error)' }}>
                  <div className="mono" style={{ fontSize: 10, color: 'inherit' }}>
                    {check.ok ? '✓' : check.optional ? '○' : '✗'} {name}: {check.message}
                  </div>
                  {check.path && <PathText path={check.path} compact />}
                </div>
              ))}
            </div>}
          </div>
        </>
      )}
    </div>
  )
}
