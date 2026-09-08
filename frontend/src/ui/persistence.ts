export const mergeUiPatch = <T extends object>(current: T, patch: T): T => {
  const result = { ...current, ...patch } as Record<string, unknown>
  for (const key of ['params', 'viewer', 'console']) {
    if (key in patch) result[key] = {
      ...(current as Record<string, object>)[key],
      ...(patch as Record<string, object>)[key],
    }
  }
  return result as T
}

// Serialize writes and retain failed patches. Responses never replace edits
// made while a request was in flight; project switches flush their own queue.
export class PatchQueue<T extends object> {
  private pending: T | null = null
  private running: Promise<void> | null = null
  private timer: ReturnType<typeof setTimeout> | undefined

  constructor(
    private readonly write: (patch: T) => Promise<unknown>,
    private readonly status: (error: Error | null, saving: boolean) => void,
  ) {}

  push = (patch: T): void => {
    this.pending = this.pending ? mergeUiPatch(this.pending, patch) : patch
    this.status(null, true)
    clearTimeout(this.timer)
    this.timer = setTimeout(() => { void this.flush().catch(() => {}) }, 200)
  }

  flush = (): Promise<void> => {
    clearTimeout(this.timer)
    if (this.running) return this.running.then(this.flush)
    if (!this.pending) return Promise.resolve()
    const patch = this.pending
    this.pending = null
    this.running = this.write(patch).then(() => {
      this.status(null, this.pending !== null)
    }).catch((error: unknown) => {
      this.pending = this.pending ? mergeUiPatch(patch, this.pending) : patch
      this.status(error instanceof Error ? error : new Error(String(error)), false)
      throw error
    }).finally(() => { this.running = null })
    return this.running.then(this.flush)
  }
}
