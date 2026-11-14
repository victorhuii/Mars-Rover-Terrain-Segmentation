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

## Release status

The application is live at [https://mars-rover-terrain-segmentation.pages.dev/](https://mars-rover-terrain-segmentation.pages.dev/). Training, frozen expert evaluation, ONNX export, local and public browser inference, CI and portfolio documentation are complete. Fresh public Chrome contexts passed sample-reference parity, uploaded-image prediction, mobile viewport, EXIF, input validation, filters, comparison, download and accessibility checks. Hosted assets and model hash match the tested build; CSP, WASM MIME type and model caching passed. Details: `results/public-deployment.json`.

## Exact next step

No implementation milestone remains. For future changes, run the relevant local checks and GitHub CI, let Cloudflare deploy `main`, and rerun the public checks documented in README. Do not retrain or repeat the frozen expert evaluation to improve the reported score. Earlier entries below record historical checkpoints; the release status above supersedes their deployment-pending notes.

## Baseline result

The trained U-Net baseline achieved validation mIoU 0.1663 on 1,557 images, with IoU 0.0000 / 0.6653 / 0.0000 / 0.0000 for soil / bedrock / sand / big rock. It predicts almost entirely bedrock and needs improvement. The local checkpoint and reproducible experiment details are recorded in IMPLEMENTATION_LOG.md and docs/experiments.md; expert labels were untouched during baseline selection.

The baseline milestone was pushed as `b24f2aa`; its [GitHub Actions run](https://github.com/victorhuii/Mars-Rover-Terrain-Segmentation/actions/runs/43600757960) passed. The pretrained compact model achieved validation mIoU 0.6978 across the full 1,557-image validation split, with big-rock IoU 0.2896; a scratch-initialized alternative was selected after 512-, 2,048- and 4,096-image experiments. TorchVision warns that pretrained model weights may have separate terms; redistribution is not yet verified.

## Frozen model

Scratch MobileNetV3-small continuation v3, SHA-256 `d8c8ceb7dc2a535aeeb0b6246fc22d4f6bab306dff32fa1664476af9c821ca8f`, selected on clean validation: mIoU 0.6312; per-class IoU 0.7111 / 0.8808 / 0.6935 / 0.2395 for soil / bedrock / sand / big rock. The pretrained comparison scored 0.6978 but its weight redistribution terms are unresolved. The protocol and lineage were committed in `experiments/frozen-model.json` before the single expert evaluation; [CI passed](https://github.com/victorhuii/Mars-Rover-Terrain-Segmentation/actions/runs/43602866225) for that freeze commit. Expert mIoU is 0.6581 across 287 images with valid labels; 35 were all-ignore. Per-class IoU is 0.8463 / 0.8104 / 0.7692 / 0.2066; big-rock support is 4,852 pixels. Full results are in `results/expert-test-min3.json`.

## Web model artifact

GitHub CI passed for the expert-result commit: https://github.com/victorhuii/Mars-Rover-Terrain-Segmentation/actions/runs/43603057395. The frozen model is exported at `frontend/public/models/marsterrain-v1-be2513a68f69.onnx`, SHA-256 `be2513a68f691802b01a6b88f4328295fba6aa88addacfdc343383fb1a3ec483`, 4,385,772 bytes. CPU ONNX/PyTorch and Chrome WASM/PyTorch logits passed on real prepared validation tensors. Browser image preprocessing and public deployment remain unverified.

## Local application verification

The real trained model runs in the React/TypeScript app on attributed examples and uploaded images. Local checks: 27 Python tests, six frontend unit tests and two production Chrome E2E tests passed, including accessibility, mobile/EXIF, filters, comparison, download and PyTorch-reference mask parity. `npm audit` reported zero vulnerabilities. Five warm browser inference calls had a 43 ms median in local headless Chrome, excluding load and preprocessing. Screenshots are in `docs/assets/`; no public deployment has been verified.

The interface copy and layout were simplified after a usability-guidance review: the title is “Mars Rover Terrain Segmentation,” controls use direct labels, and redundant decorative text was removed. Local frontend checks and both Chrome E2E tests passed; public deployment is still pending Cloudflare account connection.

## 2026-11-14 — code readability cleanup

- Removed explanatory comments and Python docstrings from project-owned source, scripts and tests. Retained the functional Vite type reference, documentation, attribution and model artifacts.
- Used clearer resize helper, channel and output-dimension names. Replaced the package smoke test's docstring assertion with an import-path assertion, and kept screenshot server retry handling explicit without an empty catch block.
- Verification: 27 Python tests, six frontend unit tests, TypeScript, ESLint, production build and both Chrome E2E tests passed. Compared Python syntax trees against the preceding commit: 30 changed Python files retain identical executable syntax after docstring removal; the smoke test assertion is the intentional exception. Browser tests required permission to bind the preview server port after the sandbox denied it. No training or expert evaluation was repeated.
- Public deployment remains pending Cloudflare account connection. Next step: connect the repository using the documented Pages settings and verify the supplied public URL. The user's VS Code settings were excluded from this change.

## 2026-11-14 — public deployment and release verification

- Public URL: https://mars-rover-terrain-segmentation.pages.dev/. Cloudflare Pages' successful GitHub check confirms deployment from `2803fc9`; GitHub CI passed: https://github.com/victorhuii/Mars-Rover-Terrain-Segmentation/actions/runs/43653680324.
- Added `PLAYWRIGHT_BASE_URL` support to reuse the production E2E suite against hosting without starting a local server. Both public Chrome tests passed (16.7 s), including actual sample/reference-mask parity, uploaded-image inference, mobile viewport/EXIF, invalid inputs, controls, PNG download and automated accessibility. The sample flow observed no page errors or failed HTTP responses.
- Added a reproducible hosted-asset check: public JS/CSS match the local build; frozen ONNX hash/size, CSP, nosniff, WASM MIME and immutable model cache headers passed. Evidence: `results/public-deployment.json`. No expert labels or training were revisited.
- Updated README, architecture, model card, portfolio, implementation plan and final report with verified hosting and accurate limits. Refreshed desktop, real-result and mobile screenshots from the public site. Final release commit CI/deployment checks are the remaining handoff gate.
