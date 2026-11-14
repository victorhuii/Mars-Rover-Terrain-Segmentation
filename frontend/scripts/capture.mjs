import { spawn } from 'node:child_process'
import { writeFile } from 'node:fs/promises'
import { Buffer } from 'node:buffer'
import { setTimeout as delay } from 'node:timers/promises'
import { chromium } from 'playwright'
import { resolve } from 'node:path'

const server = spawn('npm', ['run', 'preview', '--', '--port', '4174', '--strictPort'], { stdio: 'ignore' })
let browser
try {
  let ready = false
  for (let i = 0; i < 40; i++) {
    try { if ((await fetch('http://127.0.0.1:4174/')).ok) { ready = true; break } } catch { await delay(250); continue }
    await delay(250)
  }
  if (!ready) throw new Error('Preview server did not start')
  browser = await chromium.launch({ headless: true, ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {}) })
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 })
  await page.goto('http://127.0.0.1:4174/')
  await page.screenshot({ path: resolve('../docs/assets/terrain-lab-desktop.png') })
  await page.getByRole('button', { name: /Open plain/i }).click()
  await page.getByRole('button', { name: /Analyze terrain/i }).click()
  await page.getByRole('heading', { name: 'Results', exact: true }).waitFor({ timeout: 120000 })
  await page.waitForFunction(() => { const canvas = document.querySelector('canvas[aria-label^="mask of"]'); return canvas && canvas.getContext('2d').getImageData(0, 0, 1, 1).data[3] > 0 })
  const maskData = await page.getByRole('img', { name: /mask of/i }).evaluate(canvas => canvas.toDataURL('image/png'))
  await writeFile(resolve('../data/processed/browser-mask.png'), Buffer.from(maskData.split(',')[1], 'base64'))
  await page.locator('.result-area').screenshot({ path: resolve('../docs/assets/terrain-result.png') })
  const initialLoadText = await page.locator('.result-foot span').last().innerText()
  const times = []
  for (let i = 0; i < 5; i++) {
    await page.getByRole('button', { name: /Analyze terrain/i }).click()
    await page.waitForFunction(() => { const button = document.querySelector('.analyze-button'); return button && !button.disabled })
    times.push(Number((await page.locator('.timing strong').innerText()).replace(/[^0-9.]/g, '')))
  }
  if (times.some(value => !Number.isFinite(value) || value <= 0)) throw new Error('Invalid browser inference timing')
  const sorted = [...times].sort((a, b) => a - b)
  await writeFile(resolve('../results/browser-latency-local.json'), JSON.stringify({
    image_id: 'NLB_443478972EDR_F0250886NCAM00258M1.JPG', runtime: 'ONNX Runtime Web WASM, one thread',
    user_agent: await page.evaluate(() => navigator.userAgent), warmup: 1, iterations: times.length,
    measured_ms: times, median_ms: sorted[Math.floor(sorted.length / 2)], initial_load_display: initialLoadText,
    scope: 'session.run only; excludes image decode, resizing, model loading and display; local production preview'
  }, null, 2) + '\n')
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('http://127.0.0.1:4174/')
  await page.screenshot({ path: resolve('../docs/assets/terrain-lab-mobile.png') })
  console.log('Saved desktop, result, and mobile screenshots.')
} finally {
  if (browser) await browser.close()
  server.kill('SIGTERM')
}
