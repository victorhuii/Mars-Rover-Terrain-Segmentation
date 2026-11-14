import { readdir, stat, readFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { join } from 'node:path'

const limit = 25 * 1024 * 1024
async function walk(dir) {
  for (const item of await readdir(dir, { withFileTypes: true })) {
    const path = join(dir, item.name)
    if (item.isDirectory()) await walk(path)
    else if ((await stat(path)).size > limit) throw new Error(`Asset exceeds Cloudflare Pages 25 MiB limit: ${path}`)
  }
}
await walk('dist')
const model = await readFile('dist/models/marsterrain-v1-be2513a68f69.onnx')
const metadata = JSON.parse(await readFile('dist/models/marsterrain-v1-be2513a68f69.json', 'utf8'))
if (createHash('sha256').update(model).digest('hex') !== metadata.onnx_sha256) throw new Error('Bundled model hash mismatch')
await stat('dist/ort/ort-wasm-simd-threaded.mjs')
await stat('dist/ort/ort-wasm-simd-threaded.wasm')
console.log('Production assets fit the Pages limit and model hash matches metadata.')
