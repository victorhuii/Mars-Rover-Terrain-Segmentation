import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const policy = "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; connect-src 'self'; font-src 'self'; worker-src 'self' blob:; object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
export default defineConfig({ plugins: [react()], preview: { headers: { 'Content-Security-Policy': policy, 'X-Content-Type-Options': 'nosniff' } } })
