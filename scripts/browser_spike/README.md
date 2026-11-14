# Browser feasibility spike

This checks a single-update experimental model, not a trained model. Install the Python dependencies described in the log, run `python3 scripts/feasibility_spike.py` from the repository root, and then:

```sh
cd scripts/browser_spike
npm ci
cp ../../data/processed/spike/untrained_spike.onnx model.onnx
CHROME_PATH='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' node check.cjs
```

Alternatively install the Playwright browser (`npx playwright install chromium`) and omit `CHROME_PATH`. The check starts and stops its own local server on an OS-assigned port and uses bounded browser timeouts. It checks finite WASM output and shape on a zero tensor. Production image preprocessing and numerical browser parity need separate checks. Do not deploy this model.

To compare an exported checkpoint on a real validation tensor, first run `marsterrain.export_onnx`, then from the repository root run:

```sh
python3 scripts/browser_spike/prepare_parity.py --checkpoint checkpoints/baseline-v1/best.pt --onnx data/processed/exports/baseline-v1.onnx --metadata data/processed/exports/baseline-v1.json
MODE=parity CHROME_PATH='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' node scripts/browser_spike/check.cjs
```

The generated model and tensor binaries are ignored. This checks browser WASM logits and argmax against PyTorch on the same already-prepared input; it does not test browser image decoding, resizing, or product UI behavior.
