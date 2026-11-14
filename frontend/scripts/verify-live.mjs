import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { readFile, writeFile } from 'node:fs/promises'

const baseUrl = process.env.PLAYWRIGHT_BASE_URL
if (!baseUrl || new URL(baseUrl).protocol !== 'https:') throw new Error('Set PLAYWRIGHT_BASE_URL to the public HTTPS site.')
const hash = bytes => createHash('sha256').update(bytes).digest('hex')
const home = await fetch(baseUrl)
assert.equal(home.status, 200)
const html = await home.text()
const policy = home.headers.get('content-security-policy') || ''
assert.ok(policy.includes("default-src 'self'"))
assert.ok(policy.includes("'wasm-unsafe-eval'"))
assert.equal(home.headers.get('x-content-type-options'), 'nosniff')
const paths = [...html.matchAll(/(?:src|href)="([^"]+\.(?:js|css))"/g)].map(match => match[1])
assert.ok(paths.length >= 2)
const assets = []
for (const path of paths) {
  const response = await fetch(new URL(path, baseUrl))
  assert.equal(response.status, 200)
  const bytes = new Uint8Array(await response.arrayBuffer())
  const local = await readFile(`dist/${path.replace(/^\//, '')}`)
  assert.equal(hash(bytes), hash(local), `Hosted asset differs from local build: ${path}`)
  assets.push({ path, sha256: hash(bytes), bytes: bytes.length })
}
const modelPath = '/models/marsterrain-v1-be2513a68f69.onnx'
const response = await fetch(new URL(modelPath, baseUrl))
assert.equal(response.status, 200)
const model = new Uint8Array(await response.arrayBuffer())
const metadata = JSON.parse(await readFile(`public${modelPath.replace('.onnx', '.json')}`, 'utf8'))
assert.equal(hash(model), metadata.onnx_sha256)
assert.ok(response.headers.get('cache-control')?.includes('immutable'))
const runtime = await fetch(new URL('/ort/ort-wasm-simd-threaded.wasm', baseUrl))
assert.equal(runtime.status, 200)
assert.ok(runtime.headers.get('content-type')?.includes('application/wasm'))
const report = {
  checked_at: new Date().toISOString(), url: baseUrl, assets,
  model: { path: modelPath, sha256: hash(model), bytes: model.length, cache_control: response.headers.get('cache-control') },
  content_security_policy: policy, wasm_content_type: runtime.headers.get('content-type'),
  scope: 'Hosted assets match the local production build; model matches frozen metadata; security and cache headers verified. Browser flow is verified separately by the production E2E suite.'
}
await writeFile('../results/public-deployment.json', JSON.stringify(report, null, 2) + '\n')
console.log('Public assets, frozen model, WASM MIME type, security policy and model caching verified.')
