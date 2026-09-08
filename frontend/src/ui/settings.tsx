import { createContext, useCallback, useContext, useLayoutEffect, useState, type ReactNode } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api, type WorkspacePreferences, type WorkspacePreferencesPatch } from '../api/client'
import { detectLang, translate, type Lang } from './i18n'
import { mergeUiPatch } from './persistence'
import { usePersistence } from './usePersistence'

export type Theme = WorkspacePreferences['theme']

interface SettingsCtx {
  theme: Theme
  setTheme: (t: Theme) => void
  lang: Lang
  setLang: (l: Lang) => void
  t: (key: string) => string
  preferences: WorkspacePreferences
  updatePreferences: (patch: WorkspacePreferencesPatch) => void
  flushPreferences: () => Promise<void>
}

const Ctx = createContext<SettingsCtx | null>(null)

const writePreferences = (patch: WorkspacePreferencesPatch) => api.patchPreferences(patch, true)

const LoadedSettings = ({ initial, children }: { initial: WorkspacePreferences; children: ReactNode }) => {
  const [preferences, setPreferences] = useState(initial)
  const { save, flush, error, saving } = usePersistence(writePreferences)
  const updatePreferences = useCallback((patch: WorkspacePreferencesPatch) => {
    setPreferences(current => mergeUiPatch(current, patch) as WorkspacePreferences)
    save(patch)
  }, [save])
  const theme = preferences.theme
  const lang = preferences.lang ?? detectLang()
  const t = (key: string) => translate(lang, key)
  useLayoutEffect(() => {
    if (theme === 'auto') document.documentElement.removeAttribute('data-theme')
    else document.documentElement.setAttribute('data-theme', theme)
  }, [theme])
  return <Ctx.Provider value={{
    preferences, updatePreferences, flushPreferences: flush,
    theme, setTheme: next => updatePreferences({ theme: next }),
    lang, setLang: next => updatePreferences({ lang: next }), t,
  }}>
    {children}
    {error && <div className="persistence-error" role="alert">
      {t('settingsSaveFailed')}: {error.message}
      <button className="btn" onClick={() => { void flush().catch(() => {}) }}>{t('retry')}</button>
    </div>}
    {saving && <div className="persistence-saving" role="status">{t('savingSettings')}</div>}
  </Ctx.Provider>
}

export const SettingsProvider = ({ children }: { children: ReactNode }) => {
  const query = useQuery({ queryKey: ['preferences'], queryFn: api.getPreferences, retry: false })
  const t = (key: string) => translate(detectLang(), key)
  if (query.isPending) return <div className="startup-status" role="status">{t('loadingSettings')}</div>
  if (query.isError) return <div className="startup-status" role="alert">
    {t('settingsLoadFailed')}: {query.error.message}
    <button className="btn" onClick={() => { void query.refetch() }}>{t('retry')}</button>
  </div>
  return <LoadedSettings initial={query.data}>{children}</LoadedSettings>
}

export const useSettings = (): SettingsCtx => {
  const c = useContext(Ctx)
  if (!c) throw new Error('useSettings outside provider')
  return c
}
