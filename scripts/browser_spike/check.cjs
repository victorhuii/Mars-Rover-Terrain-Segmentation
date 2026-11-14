// Start the server and browser in one process, with an OS-assigned port and
// bounded timeouts. This is a feasibility check, not a trained-model test.
const { chromium } = require('playwright');
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const root = __dirname;
const types = { '.html': 'text/html', '.mjs': 'text/javascript', '.wasm': 'application/wasm', '.onnx': 'application/octet-stream' };
const server = http.createServer((req, res) => {
  const name = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
  const file = path.resolve(root, `.${name === '/' ? '/index.html' : name}`);
  if (!file.startsWith(root + path.sep)) { res.writeHead(403).end(); return; }
  fs.readFile(file, (error, bytes) => {
    if (error) { res.writeHead(404).end(); return; }
    res.writeHead(200, { 'Content-Type': types[path.extname(file)] || 'application/octet-stream' });
    res.end(bytes);
  });
});
(async () => {
  await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
  const options = { headless: true, timeout: 15000 };
  if (process.env.CHROME_PATH) options.executablePath = process.env.CHROME_PATH;
  const browser = await chromium.launch(options);
  try {
    const page = await browser.newPage();
    await page.goto(`http://127.0.0.1:${server.address().port}/${process.env.MODE === 'parity' ? 'parity.html' : ''}`,  { timeout: 15000 });
    await page.waitForFunction(() => document.querySelector('#result').textContent !== 'pending', null, { timeout: 15000 });
    const output = await page.locator('#result').innerText();
    const result = JSON.parse(output);
    if (JSON.stringify(result.shape) !== '[1,4,256,256]' || result.finite !== true) throw new Error(output);
    if (process.env.MODE === 'parity' && (!(result.maxError <= 1e-4) || !(result.agreement >= 0.999))) throw new Error(output);
    console.log(output);
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; }).finally(() => server.close());
