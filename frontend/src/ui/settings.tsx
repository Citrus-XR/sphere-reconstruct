import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'
import { detectLang, translate, type Lang } from './i18n'

export type Theme = 'auto' | 'light' | 'dark'

interface SettingsCtx {
  theme: Theme
  setTheme: (t: Theme) => void
  lang: Lang
  setLang: (l: Lang) => void
  t: (key: string) => string
}

const Ctx = createContext<SettingsCtx | null>(null)

const applyTheme = (theme: Theme) => {
  const el = document.documentElement
  if (theme === 'auto') el.removeAttribute('data-theme')
  else el.setAttribute('data-theme', theme)
}

// テーマ/言語をブラウザ (localStorage) に保存する. テーマ既定=auto (ブラウザ準拠), 言語=ブラウザ言語.
export const SettingsProvider = ({ children }: { children: ReactNode }) => {
  const [theme, setThemeState] = useState<Theme>(() => (localStorage.getItem('theme') as Theme) || 'auto')
  const [lang, setLangState] = useState<Lang>(() => (localStorage.getItem('lang') as Lang) || detectLang())

  useEffect(() => { applyTheme(theme); localStorage.setItem('theme', theme) }, [theme])
  useEffect(() => { localStorage.setItem('lang', lang) }, [lang])

  const value: SettingsCtx = {
    theme, setTheme: setThemeState,
    lang, setLang: setLangState,
    t: (key: string) => translate(lang, key),
  }
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export const useSettings = (): SettingsCtx => {
  const c = useContext(Ctx)
  if (!c) throw new Error('useSettings outside provider')
  return c
}
