import { useEffect, useMemo, useState } from 'react'
import { PatchQueue } from './persistence'

export const usePersistence = <T extends object>(write: (patch: T) => Promise<unknown>) => {
  const [status, setStatus] = useState<{ error: Error | null; saving: boolean }>({ error: null, saving: false })
  const queue = useMemo(() => new PatchQueue(write, (error, saving) => setStatus({ error, saving })), [write])
  useEffect(() => {
    const flush = () => { void queue.flush().catch(() => {}) }
    const visibility = () => { if (document.visibilityState === 'hidden') flush() }
    window.addEventListener('pagehide', flush)
    document.addEventListener('visibilitychange', visibility)
    return () => {
      window.removeEventListener('pagehide', flush)
      document.removeEventListener('visibilitychange', visibility)
      flush()
    }
  }, [queue])
  return { save: queue.push, flush: queue.flush, ...status }
}
