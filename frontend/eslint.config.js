import js from '@eslint/js'
import tseslint from 'typescript-eslint'

export default tseslint.config(
  { ignores: ['dist/**', 'public/ort/**', 'node_modules/**', 'playwright-report/**', 'test-results/**'] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  { files: ['**/*.{js,mjs,cjs}'], languageOptions: { globals: { process: 'readonly', URL: 'readonly', console: 'readonly', fetch: 'readonly', document: 'readonly', navigator: 'readonly' } } },
  { files: ['**/*.{ts,tsx}'], languageOptions: { globals: { process: 'readonly', Buffer: 'readonly', window: 'readonly', document: 'readonly', Image: 'readonly', ImageData: 'readonly', HTMLCanvasElement: 'readonly', HTMLImageElement: 'readonly', File: 'readonly', Blob: 'readonly', URL: 'readonly', performance: 'readonly', fetch: 'readonly', console: 'readonly', setTimeout: 'readonly', clearTimeout: 'readonly', Uint8Array: 'readonly', Uint8ClampedArray: 'readonly', Float32Array: 'readonly', DOMException: 'readonly' } } }
)
