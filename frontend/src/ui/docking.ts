import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react'
import { Actions, Model, TabNode, type IJsonModel } from 'flexlayout-react'
import { useSettings } from './settings'

export const DEFAULT_LAYOUT: IJsonModel = {
  global: { tabEnableClose: false, tabEnableRename: false, tabSetEnableMaximize: true },
  borders: [],
  layout: {
    type: 'row', children: [
      { type: 'tabset', weight: 20, children: [{ type: 'tab', id: 'sceneHier', name: 'Hierarchy', component: 'sceneHier' }] },
      { type: 'tabset', weight: 58, children: [
        { type: 'tab', id: 'scene', name: 'Scene View', component: 'scene', enableRenderOnDemand: false },
        { type: 'tab', id: 'console', name: 'Console', component: 'console' },
        { type: 'tab', id: 'inspector', name: 'Inspector', component: 'inspector' },
      ] },
      { type: 'tabset', weight: 22, children: [{ type: 'tab', id: 'steps', name: 'Steps', component: 'steps' }] },
    ],
  },
}

const COMPACT_LAYOUT: IJsonModel = {
  ...DEFAULT_LAYOUT,
  layout: { type: 'row', children: [{ type: 'tabset', children: [
    { type: 'tab', id: 'scene', name: 'Scene View', component: 'scene', enableRenderOnDemand: false },
    { type: 'tab', id: 'console', name: 'Console', component: 'console' },
    { type: 'tab', id: 'inspector', name: 'Inspector', component: 'inspector' },
    { type: 'tab', id: 'steps', name: 'Steps', component: 'steps' },
    { type: 'tab', id: 'sceneHier', name: 'Hierarchy', component: 'sceneHier' },
  ] }] },
}
const compactMedia = window.matchMedia('(max-width: 700px)')

export const useDocking = () => {
  const { preferences, updatePreferences, lang, t } = useSettings()
  // Choose the dock model at startup; changing models remounts editors and loses drafts.
  const [compact] = useState(() => compactMedia.matches)
  const [desktopModel] = useState(() => Model.fromJson(preferences.layout ?? DEFAULT_LAYOUT))
  const [compactModel] = useState(() => Model.fromJson(preferences.compactLayout ?? COMPACT_LAYOUT))
  const model = compact ? compactModel : desktopModel
  const [, rerender] = useState(0)
  const [pings, setPings] = useState<Record<string, number>>({})
  const lastPing = useRef<Record<string, number>>({})
  const ping = useCallback((id: string) => {
    const now = performance.now()
    if (now - (lastPing.current[id] ?? -Infinity) < 1200) return
    lastPing.current[id] = now
    setPings(previous => ({ ...previous, [id]: (previous[id] ?? 0) + 1 }))
  }, [])
  const onModelChange = useCallback((next: Model) => {
    rerender(value => value + 1)
    updatePreferences(compact ? { compactLayout: next.toJson() } : { layout: next.toJson() })
  }, [compact, updatePreferences])
  const reveal = (id: string) => {
    model.doAction(Actions.selectTab(id))
    ping(id)
  }
  const titles = useMemo(() => ({
    sceneHier: 'tabHierarchy', scene: 'tabSceneView', console: 'tabConsole',
    steps: 'tabSteps', inspector: 'tabInspector',
  }), [])
  useEffect(() => {
    model.visitNodes(node => {
      if (!(node instanceof TabNode)) return
      const title = t(titles[node.getId() as keyof typeof titles])
      if (node.getName() !== title) model.doAction(Actions.renameTab(node.getId(), title))
    })
  }, [lang, model]) // eslint-disable-line react-hooks/exhaustive-deps
  return { model, pings, ping, reveal, titles, onModelChange }
}

export const useTabVisibility = (node: TabNode): boolean => {
  const subscribe = useCallback((listener: () => void) => {
    node.setEventListener('visibility', listener)
    return () => node.removeEventListener('visibility')
  }, [node])
  return useSyncExternalStore(subscribe, () => node.isVisible())
}
