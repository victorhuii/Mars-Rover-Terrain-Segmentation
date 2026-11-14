# Browser feasibility spike

This checks a single-update experimental model, not a trained model. Install the Python dependencies described in the log, run `python3 scripts/feasibility_spike.py` from the repository root, and then:

```sh
cd scripts/browser_spike
npm ci
cp ../../data/processed/spike/untrained_spike.onnx model.onnx
CHROME_PATH='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' node check.cjs
```

Alternatively install the Playwright browser (`npx playwright install chromium`) and omit `CHROME_PATH`. The check starts and stops its own local server on an OS-assigned port and uses bounded browser timeouts. It checks finite WASM output and shape on a zero tensor. Production image preprocessing and numerical browser parity need separate checks. Do not deploy this model.
