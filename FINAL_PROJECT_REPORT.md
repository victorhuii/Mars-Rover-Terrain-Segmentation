# MarsTerrain project report — deployment pending

This is a checkpoint report, not a completion claim. The trained model, evaluation, application and local tests are complete; public hosting and live verification remain outstanding.

## Evidence

- **Repository:** [Mars-Rover-Terrain-Segmentation](https://github.com/victorhuii/Mars-Rover-Terrain-Segmentation), branch `main`. The application checkpoint is `194cd9e`; preceding model/data milestones include `b9778ed`, `ccd4e74`, `b24f2aa`, `f0ee18a`, `c870e43`, and `b5849c0`.
- **Public URL:** not yet deployed. Cloudflare CLI reported that it is not authenticated.
- **Data:** AI4Mars Curiosity mirror revision `93dbb79de6cd234799750b6129fbfb3b197582fa`; 9,154 clean training and 1,557 validation images after audit and mask validation. The selected scratch model trained in stages on up to 4,096 training images. All 322 unique expert images also appeared in the mirror's training rows and were excluded, together with conservatively linked groups.
- **Model:** scratch-initialized MobileNetV3-small encoder and compact decoder, selected on validation; frozen checkpoint SHA-256 `d8c8ceb7dc2a535aeeb0b6246fc22d4f6bab306dff32fa1664476af9c821ca8f`. The exact ONNX artifact is 4,385,772 bytes, SHA-256 `be2513a68f691802b01a6b88f4328295fba6aa88addacfdc343383fb1a3ec483`.
- **Expert test:** one frozen `test_min3` evaluation, 287 of 322 images with valid labels, 7,310,367 valid pixels. mIoU 0.6581; per-class IoU soil 0.8463, bedrock 0.8104, sand 0.7692, big rock 0.2066; mean Dice 0.7560; pixel accuracy 0.8986. The other 35 images were all-ignore. Big rock has only 4,852 reference pixels, making that estimate fragile. See `results/expert-test-min3.json`.
- **Parity and runtime:** CPU ONNX/PyTorch logits on three validation tensors differed by at most 1.32e-5, with identical argmax; Chrome WASM on one prepared validation tensor differed by at most 1.10e-5, also with identical argmax. A browser-decoded and resized sample matched the Python-reference class mask in local Chrome. Five warm browser `session.run` calls on one sample had median 43 ms locally, excluding decode, resize and loading. Full-validation MPS evaluation took 17.0 s including I/O. These are not public network or device-wide benchmarks.
- **Local quality checks:** 27 Python tests, six frontend unit tests and two production Chrome E2E tests passed. The E2E suite covers actual model inference, sample and upload paths, EXIF rotation, input limits, filters, compare, download, mobile and an automated accessibility scan. `npm audit` reported zero vulnerabilities. The expanded GitHub Actions run for `194cd9e` passed Python, frontend lint/typecheck/unit tests and build, but failed the browser installation step before browser tests. A workflow change to use the runner-provided Chrome is pending verification: https://github.com/victorhuii/Mars-Rover-Terrain-Segmentation/actions/runs/43644313226.
- **Attribution:** NASA/JPL-Caltech and Swan et al., AI4Mars; the official Zenodo record reports CC BY 4.0, while the mirror declares CC0. The dataset card and third-party notices document source, code and font licenses. This is an independent research project with no NASA affiliation or endorsement.

## Remaining release work

1. Confirm the expanded GitHub Actions run and fix any failure.
2. Connect this GitHub repository to Cloudflare Pages on the free plan. Set root `frontend`, build command `npm run build`, output directory `dist`, production branch `main` (the checked-in `.nvmrc` selects Node 24.14.1); or authenticate Wrangler with `npx wrangler@4.162.0 login` and provide appropriate account access.
3. Verify the public URL in a fresh browser and mobile viewport, run real sample and arbitrary-upload inference, inspect network/console errors, compare, filters and download, and confirm deployment is triggered by GitHub pushes.
4. Update the README, portfolio copy, model card, status and this report with the verified URL and CI outcome. Mark the project complete only after those checks pass. Do not retrain or rerun expert evaluation to improve the reported test score.

The local screenshots are in `docs/assets/`, the full experimental record is in `docs/experiments.md`, and truthful resume bullets are in `docs/portfolio.md`.
