import { Add24Regular } from '@fluentui/react-icons/svg/add'
import { Broom24Regular } from '@fluentui/react-icons/svg/broom'
import { Camera24Regular } from '@fluentui/react-icons/svg/camera'
import { Checkmark24Regular } from '@fluentui/react-icons/svg/checkmark'
import { ChevronLeft24Regular } from '@fluentui/react-icons/svg/chevron-left'
import { ChevronRight24Regular } from '@fluentui/react-icons/svg/chevron-right'
import { Color24Regular } from '@fluentui/react-icons/svg/color'
import { Copy24Regular } from '@fluentui/react-icons/svg/copy'
import { DataScatter24Regular } from '@fluentui/react-icons/svg/data-scatter'
import { Delete24Regular } from '@fluentui/react-icons/svg/delete'
import { Dismiss24Regular } from '@fluentui/react-icons/svg/dismiss'
import { ErrorCircle24Regular } from '@fluentui/react-icons/svg/error-circle'
import { Folder24Regular } from '@fluentui/react-icons/svg/folder'
import { FolderOpen24Regular } from '@fluentui/react-icons/svg/folder-open'
import { Grid24Regular } from '@fluentui/react-icons/svg/grid'
import { Image24Regular } from '@fluentui/react-icons/svg/image'
import { ImageMultiple24Regular } from '@fluentui/react-icons/svg/image-multiple'
import { ImageProhibited24Regular } from '@fluentui/react-icons/svg/image-prohibited'
import { Info24Regular } from '@fluentui/react-icons/svg/info'
import { Layer24Regular } from '@fluentui/react-icons/svg/layer'
import { Navigation24Regular } from '@fluentui/react-icons/svg/navigation'
import { Play24Filled } from '@fluentui/react-icons/svg/play'
import { Road24Regular } from '@fluentui/react-icons/svg/road'
import { Search24Regular } from '@fluentui/react-icons/svg/search'
import { Settings24Regular } from '@fluentui/react-icons/svg/settings'
import { Stop24Filled } from '@fluentui/react-icons/svg/stop'
import { Target24Regular } from '@fluentui/react-icons/svg/target'
import { VideoClip24Regular } from '@fluentui/react-icons/svg/video-clip'
import { Warning24Regular } from '@fluentui/react-icons/svg/warning'
import { ZoomIn24Regular } from '@fluentui/react-icons/svg/zoom-in'

const icons = {
  add: Add24Regular,
  broom: Broom24Regular,
  camera: Camera24Regular,
  checkmark: Checkmark24Regular,
  chevronLeft: ChevronLeft24Regular,
  chevronRight: ChevronRight24Regular,
  color: Color24Regular,
  copy: Copy24Regular,
  error: ErrorCircle24Regular,
  delete: Delete24Regular,
  dismiss: Dismiss24Regular,
  folder: Folder24Regular,
  folderOpen: FolderOpen24Regular,
  grid: Grid24Regular,
  image: Image24Regular,
  info: Info24Regular,
  layer: Layer24Regular,
  mask: ImageProhibited24Regular,
  navigation: Navigation24Regular,
  play: Play24Filled,
  points: DataScatter24Regular,
  path: Road24Regular,
  search: Search24Regular,
  settings: Settings24Regular,
  stop: Stop24Filled,
  target: Target24Regular,
  overlay: ImageMultiple24Regular,
  video: VideoClip24Regular,
  warning: Warning24Regular,
  zoom: ZoomIn24Regular,
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
