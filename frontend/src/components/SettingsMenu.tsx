import { useState } from 'react'
import { useSettings } from '../ui/settings'
import type { Lang } from '../ui/i18n'
import type { Theme } from '../ui/settings'

// 右上の設定メニュー: テーマ (自動/ライト/ダーク) + 言語. ブラウザに保存.
export const SettingsMenu = () => {
  const { theme, setTheme, lang, setLang, t } = useSettings()
  const [open, setOpen] = useState(false)
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
          </div>
        </>
      )}
    </div>
  )
}
