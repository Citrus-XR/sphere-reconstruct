import { expect, test } from '@playwright/test'
import { PatchQueue, mergeUiPatch } from '../src/ui/persistence'
import { parsePoints } from '../src/api/client'

test('partial panel patches preserve independent settings and replace layouts', () => {
  expect(mergeUiPatch({ params: { fps: 1.5, sharpness: 240 }, layout: { previous: true } },
    { params: { fps: 3 }, layout: { next: true } })).toEqual({
    params: { fps: 3, sharpness: 240 }, layout: { next: true },
  })
})

test('queued saves serialize, coalesce and retain edits after failure', async () => {
  const writes: object[] = []
  let unblock!: () => void
  const blocked = new Promise<void>(resolve => { unblock = resolve })
  const statuses: Array<Error | null> = []
  let fail = true
  const queue = new PatchQueue(async (patch: object) => {
    writes.push(patch)
    if (writes.length === 1) await blocked
    if (fail) throw new Error('network unavailable')
  }, error => statuses.push(error))
  queue.push({ params: { sharpness: 240 } })
  const first = queue.flush()
  queue.push({ params: { fps: 3 } })
  unblock()
  await expect(first).rejects.toThrow('network unavailable')
  expect(statuses.at(-1)?.message).toBe('network unavailable')
  fail = false
  await queue.flush()
  expect(writes).toEqual([{ params: { sharpness: 240 } }, { params: { sharpness: 240, fps: 3 } }])
  expect(statuses.at(-1)).toBeNull()
})

test('point payload checks its full shape before allocation and uses linear vertex colors', () => {
  expect(() => parsePoints(new ArrayBuffer(0))).toThrow('header')
  const bytes = new ArrayBuffer(28)
  const view = new DataView(bytes)
  view.setUint32(0, 1, true)
  view.setUint32(4, 20, true)
  view.setFloat32(8, 1, true)
  view.setUint8(20, 128)
  view.setUint8(21, 255)
  const parsed = parsePoints(bytes)
  expect(parsed.count).toBe(1)
  expect(parsed.colors[0]).toBeCloseTo(0.21586, 4)
  expect(parsed.colors[1]).toBe(1)
  expect(parsed.colors[2]).toBe(0)
  view.setUint32(0, 0xffffffff, true)
  expect(() => parsePoints(bytes)).toThrow('length or stride')
  view.setUint32(0, 1, true)
  view.setFloat32(8, NaN, true)
  expect(() => parsePoints(bytes)).toThrow('position')
})
