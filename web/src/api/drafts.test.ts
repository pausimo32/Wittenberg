/**
 * `docs/05-webapp-spec.md` §9, requirement 1: the bytes that arrive are the bytes that are
 * stored. The server half is asserted by `test_drafts.py::test_stored_bytes_are_the_uploaded_bytes`;
 * this is the browser half, and it is the half that can corrupt `data/raw/` invisibly — a
 * client-side resize or re-encode produces a file that still opens, still looks right, and
 * carries JPEG artefacts that survive Canny as spurious contours.
 */

import { afterEach, describe, expect, it, vi } from 'vitest'
import { uploadFrame } from './drafts'

afterEach(() => vi.unstubAllGlobals())

function captureFetch(): { body: () => FormData } {
  let sent: FormData | undefined
  vi.stubGlobal(
    'fetch',
    vi.fn(async (_url: string, init: RequestInit) => {
      sent = init.body as FormData
      return new Response(JSON.stringify({ name: 'IMG_0001.jpg' }), { status: 200 })
    }),
  )
  return { body: () => sent as FormData }
}

/**
 * jsdom's `Blob` predates `arrayBuffer()` and `text()` — its prototype carries `slice`, `size`
 * and `type` and nothing else — so the bytes of a form part have to come back through
 * `FileReader`, which jsdom does implement. The assertion is the same one either way: what came
 * out of the `FormData` is what went into the `File`.
 */
function bytesOf(blob: Blob): Promise<Uint8Array> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onerror = () => reject(reader.error ?? new Error('could not read the form part'))
    reader.onload = () => resolve(new Uint8Array(reader.result as ArrayBuffer))
    reader.readAsArrayBuffer(blob)
  })
}

describe('uploadFrame', () => {
  it("sends the file's bytes unmodified", async () => {
    // The requirement this test exists for: a browser form can recode what it uploads, and a
    // recoded JPEG survives Canny as spurious contours that nobody can see in the file
    // afterwards. If someone adds a client-side resize, this is what fails.
    const bytes = new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, 0x49, 0x46])
    const file = new File([bytes], 'IMG_4711.JPG', { type: 'image/jpeg' })
    const captured = captureFetch()

    await uploadFrame('20260830-01', 440, file)

    const part = captured.body().get('file') as File
    expect(await bytesOf(part)).toEqual(bytes)
    expect(part.name).toBe('IMG_4711.JPG')
  })

  it('puts the class in the query, not in the body', async () => {
    const captured = captureFetch()

    await uploadFrame('20260830-01', 440, new File([new Uint8Array([1])], 'a.jpg'))

    expect(vi.mocked(fetch).mock.calls[0]?.[0]).toBe('/api/drafts/20260830-01/frames?freq_hz=440')
    expect(captured.body().get('freq_hz')).toBeNull()
  })
})
