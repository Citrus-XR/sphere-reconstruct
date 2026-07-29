import { Add24Regular } from '@fluentui/react-icons/svg/add'
import { Broom24Regular } from '@fluentui/react-icons/svg/broom'
import { Checkmark24Regular } from '@fluentui/react-icons/svg/checkmark'
import { Copy24Regular } from '@fluentui/react-icons/svg/copy'
import { Delete24Regular } from '@fluentui/react-icons/svg/delete'
import { Dismiss24Regular } from '@fluentui/react-icons/svg/dismiss'
import { Folder24Regular } from '@fluentui/react-icons/svg/folder'
import { FolderOpen24Regular } from '@fluentui/react-icons/svg/folder-open'
import { Navigation24Regular } from '@fluentui/react-icons/svg/navigation'
import { Play24Filled } from '@fluentui/react-icons/svg/play'
import { Search24Regular } from '@fluentui/react-icons/svg/search'
import { Settings24Regular } from '@fluentui/react-icons/svg/settings'
import { Stop24Filled } from '@fluentui/react-icons/svg/stop'
import { VideoClip24Regular } from '@fluentui/react-icons/svg/video-clip'
import { Warning24Regular } from '@fluentui/react-icons/svg/warning'

const icons = {
  add: Add24Regular,
  broom: Broom24Regular,
  checkmark: Checkmark24Regular,
  copy: Copy24Regular,
  delete: Delete24Regular,
  dismiss: Dismiss24Regular,
  folder: Folder24Regular,
  folderOpen: FolderOpen24Regular,
  navigation: Navigation24Regular,
  play: Play24Filled,
  search: Search24Regular,
  settings: Settings24Regular,
  stop: Stop24Filled,
  video: VideoClip24Regular,
  warning: Warning24Regular,
} as const

export type AppIconName = keyof typeof icons

export const AppIcon = ({ name, size = 18, className = '' }: {
  name: AppIconName
  size?: number
  className?: string
}) => {
  const Icon = icons[name]
  return <Icon fontSize={size} className={className} aria-hidden="true" />
}
