import { useEffect, useRef, useState } from 'react'
import { useSettings } from '../ui/settings'
import { AppIcon } from './AppIcon'

export const formatPathForDisplay = (path: string): string => path.replaceAll('\\', '/')

export const PathText = ({
  path,
  className = '',
  compact = false,
}: {
  path: string
  className?: string
  compact?: boolean
}) => {
  const { t } = useSettings()
  const [copyState, setCopyState] = useState<'idle' | 'copied' | 'failed'>('idle')
  const resetTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const displayPath = formatPathForDisplay(path)

  useEffect(() => () => {
    if (resetTimer.current !== null) clearTimeout(resetTimer.current)
  }, [])

  const copy = async () => {
    try {
      if (!navigator.clipboard) {
        setCopyState('failed')
      } else {
        await navigator.clipboard.writeText(path)
        setCopyState('copied')
      }
    } catch {
      setCopyState('failed')
    }
    if (resetTimer.current !== null) clearTimeout(resetTimer.current)
    resetTimer.current = setTimeout(() => setCopyState('idle'), 1800)
  }

  const copyLabel = copyState === 'copied' ? t('pathCopied')
    : copyState === 'failed' ? t('pathCopyFailed') : t('copyPath')

  return (
    <span className={`path-text ${compact ? 'compact' : ''} ${className}`.trim()} data-copy-value={path}>
      <span className="path-value" dir="ltr" title={displayPath}>{displayPath}</span>
      <button className={`path-copy ${copyState}`} type="button" onClick={copy}
        aria-label={`${copyLabel}: ${displayPath}`} title={copyLabel}>
        <AppIcon name={copyState === 'copied' ? 'checkmark' : copyState === 'failed' ? 'warning' : 'copy'} size={14} />
      </button>
      <span className="sr-only" role="status" aria-live="polite">
        {copyState === 'idle' ? '' : copyLabel}
      </span>
    </span>
  )
}
