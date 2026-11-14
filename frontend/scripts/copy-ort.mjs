import { copyFile, mkdir, rm } from 'node:fs/promises'
import { resolve } from 'node:path'

const target = resolve('public/ort')
await rm(target, { recursive: true, force: true })
await mkdir(target, { recursive: true })
for (const name of ['ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm']) {
  await copyFile(resolve('node_modules/onnxruntime-web/dist', name), resolve(target, name))
}
