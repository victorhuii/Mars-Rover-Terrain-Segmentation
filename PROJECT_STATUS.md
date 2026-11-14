# Project status

Updated: 2026-11-14

## Completed work

- Read both specifications, inspected the existing repository and toolchain, created the milestone plan, ignore rules, smoke test and CI skeleton.
- Committed and pushed milestone 0 as `b9778ed` to the configured origin/main.
- Pinned AI4Mars mirror revision `93dbb79de6cd234799750b6129fbfb3b197582fa`; downloaded all 13 train shards and three expert variants locally. The image-only full audit is complete; expert annotations were read only after the protocol was frozen.

## Verification and findings

- `python3 -m unittest discover -s tests -v`: 1 smoke test passed. GitHub CI passed for milestone 0: https://github.com/victorhuii/Mars-Rover-Terrain-Segmentation/actions/runs/43597529495.
- MPS is available outside the sandbox; inside the sandbox it reports unavailable. Python 3.14.2, Node 24.14.1, npm 11.11.0, PyTorch 2.15.0; 62 GiB disk free at inspection.
- Full audit: all 322 unique expert images also occur in train by filename and content. Conservative grouping marks 6,837 of 18,130 train rows as expert-related; 9,230 labeled rows are assigned to train and 1,563 to validation before mask validation. Three non-image files are excluded. Split identity, scene, hash and near-duplicate checks passed. Expert labels were untouched during split construction and model selection.
- Mirror lists CC0; official Zenodo record lists CC BY 4.0. NASA catalogue leaves license unspecified. Attribution and redistribution terms require care.
- Local dataset shards and audit cache are ignored and are not in Git; hashes and source details are in IMPLEMENTATION_LOG.md. The audit found a non-image `current.txt`; it is excluded and completed shard scans are checkpointed.
- The data pipeline has 13 passing local Python tests and a passing clean-install Chrome smoke; GitHub CI passed for milestone 1: https://github.com/victorhuii/Mars-Rover-Terrain-Segmentation/actions/runs/43600221766. Preparation yielded 9,154 train and 1,557 validation samples, excluding 82 further unusable annotations. Counts and manifests are tracked under `data/manifests`; raw/prepared imagery remains ignored.
- A single-update real-image CPU spike exported ONNX with max logit error 6.71e-8 on an audited clean training record. Chrome WASM loaded that experimental model and produced finite output of the expected shape on a zero tensor using a temporary server. This is not a trained model or production parity test. The self-contained, dynamically ported browser smoke check now passes in Chrome.

## Remaining

Public deployment and live incognito/mobile verification remain. README, model/data cards, local screenshots and portfolio notes are prepared, but the live link and final report cannot be completed until deployment is verified. No public application exists yet.

## Exact next step

The application was pushed as `194cd9e` and the browser CI fix as `8b3a1cf`; both Python and frontend jobs passed in [GitHub Actions](https://github.com/victorhuii/Mars-Rover-Terrain-Segmentation/actions/runs/43644643743). Cloudflare CLI is not authenticated. At the deployment checkpoint, connect the existing GitHub repository to Cloudflare Pages on a free plan (root `frontend`, build `npm run build`, output `dist`, production branch `main`) or authenticate the CLI, then verify the public URL and real uploads in a fresh browser. Do not retrain or reevaluate the expert test based on its results.

## Baseline result

The trained U-Net baseline achieved validation mIoU 0.1663 on 1,557 images, with IoU 0.0000 / 0.6653 / 0.0000 / 0.0000 for soil / bedrock / sand / big rock. It predicts almost entirely bedrock and needs improvement. The local checkpoint and reproducible experiment details are recorded in IMPLEMENTATION_LOG.md and docs/experiments.md; expert labels were untouched during baseline selection.

The baseline milestone was pushed as `b24f2aa`; its [GitHub Actions run](https://github.com/victorhuii/Mars-Rover-Terrain-Segmentation/actions/runs/43600757960) passed. The pretrained compact model achieved validation mIoU 0.6978 across the full 1,557-image validation split, with big-rock IoU 0.2896; a scratch-initialized alternative was selected after 512-, 2,048- and 4,096-image experiments. TorchVision warns that pretrained model weights may have separate terms; redistribution is not yet verified.

## Frozen model

Scratch MobileNetV3-small continuation v3, SHA-256 `d8c8ceb7dc2a535aeeb0b6246fc22d4f6bab306dff32fa1664476af9c821ca8f`, selected on clean validation: mIoU 0.6312; per-class IoU 0.7111 / 0.8808 / 0.6935 / 0.2395 for soil / bedrock / sand / big rock. The pretrained comparison scored 0.6978 but its weight redistribution terms are unresolved. The protocol and lineage were committed in `experiments/frozen-model.json` before the single expert evaluation; [CI passed](https://github.com/victorhuii/Mars-Rover-Terrain-Segmentation/actions/runs/43602866225) for that freeze commit. Expert mIoU is 0.6581 across 287 images with valid labels; 35 were all-ignore. Per-class IoU is 0.8463 / 0.8104 / 0.7692 / 0.2066; big-rock support is 4,852 pixels. Full results are in `results/expert-test-min3.json`.

## Web model artifact

GitHub CI passed for the expert-result commit: https://github.com/victorhuii/Mars-Rover-Terrain-Segmentation/actions/runs/43603057395. The frozen model is exported at `frontend/public/models/marsterrain-v1-be2513a68f69.onnx`, SHA-256 `be2513a68f691802b01a6b88f4328295fba6aa88addacfdc343383fb1a3ec483`, 4,385,772 bytes. CPU ONNX/PyTorch and Chrome WASM/PyTorch logits passed on real prepared validation tensors. Browser image preprocessing and public deployment remain unverified.

## Local application verification

The real trained model runs in the React/TypeScript app on attributed examples and uploaded images. Local checks: 27 Python tests, six frontend unit tests and two production Chrome E2E tests passed, including accessibility, mobile/EXIF, filters, comparison, download and PyTorch-reference mask parity. `npm audit` reported zero vulnerabilities. Five warm browser inference calls had a 43 ms median in local headless Chrome, excluding load and preprocessing. Screenshots are in `docs/assets/`; no public deployment has been verified.
