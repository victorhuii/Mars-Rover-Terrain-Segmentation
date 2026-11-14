# MarsTerrain final project report

The trained model, frozen evaluation, browser application, automated tests and public deployment are verified. [Open the application](https://mars-rover-terrain-segmentation.pages.dev/).

## Evidence

- **Repository:** [Mars-Rover-Terrain-Segmentation](https://github.com/victorhuii/Mars-Rover-Terrain-Segmentation), branch `main`. The application checkpoint is `194cd9e` and the browser CI fix is `8b3a1cf`; preceding model/data milestones include `b9778ed`, `ccd4e74`, `b24f2aa`, `f0ee18a`, `c870e43`, and `b5849c0`.
- **Public URL:** [https://mars-rover-terrain-segmentation.pages.dev/](https://mars-rover-terrain-segmentation.pages.dev/), hosted on Cloudflare Pages through Git integration. Cloudflare reported a successful deployment of `2803fc9`; fresh public browser tests passed.
- **Data:** AI4Mars Curiosity mirror revision `93dbb79de6cd234799750b6129fbfb3b197582fa`; 9,154 clean training and 1,557 validation images after audit and mask validation. The selected scratch model trained in stages on up to 4,096 training images. All 322 unique expert images also appeared in the mirror's training rows and were excluded, together with conservatively linked groups.
- **Model:** scratch-initialized MobileNetV3-small encoder and compact decoder, selected on validation; frozen checkpoint SHA-256 `d8c8ceb7dc2a535aeeb0b6246fc22d4f6bab306dff32fa1664476af9c821ca8f`. The exact ONNX artifact is 4,385,772 bytes, SHA-256 `be2513a68f691802b01a6b88f4328295fba6aa88addacfdc343383fb1a3ec483`.
- **Expert test:** one frozen `test_min3` evaluation, 287 of 322 images with valid labels, 7,310,367 valid pixels. mIoU 0.6581; per-class IoU soil 0.8463, bedrock 0.8104, sand 0.7692, big rock 0.2066; mean Dice 0.7560; pixel accuracy 0.8986. The other 35 images were all-ignore. Big rock has only 4,852 reference pixels, making that estimate fragile. See `results/expert-test-min3.json`.
- **Parity and runtime:** CPU ONNX/PyTorch logits on three validation tensors differed by at most 1.32e-5, with identical argmax; Chrome WASM on one prepared validation tensor differed by at most 1.10e-5, also with identical argmax. A browser-decoded and resized sample matched the Python-reference class mask in local Chrome. Five warm browser `session.run` calls on one sample had median 43 ms locally, excluding decode, resize and loading. Full-validation MPS evaluation took 17.0 s including I/O. These are not public network or device-wide benchmarks.
- **Local quality checks:** 27 Python tests, six frontend unit tests and two production Chrome E2E tests passed. The E2E suite covers actual model inference, sample and upload paths, EXIF rotation, input limits, filters, compare, download, mobile and an automated accessibility scan. `npm audit` reported zero vulnerabilities. Both Python and frontend jobs, including real browser tests, passed for `8b3a1cf`: https://github.com/victorhuii/Mars-Rover-Terrain-Segmentation/actions/runs/43644643743.
- **Attribution:** NASA/JPL-Caltech and Swan et al., AI4Mars; the official Zenodo record reports CC BY 4.0, while the mirror declares CC0. The dataset card and third-party notices document source, code and font licenses. This is an independent research project with no NASA affiliation or endorsement.

## Production verification

- Both production E2E tests passed against the public HTTPS URL in fresh Chrome contexts. They exercised real sample predictions against a saved PyTorch reference mask, uploaded-image inference, EXIF orientation, invalid-file and image-size rejection, class filters, overlay opacity, comparison, PNG download, mobile viewport and an axe accessibility scan. The desktop flow reported no uncaught page errors or failed HTTP responses.
- Hosted JavaScript and CSS match the local tested build. The downloaded model matches the frozen SHA-256 and size. Security policy, `nosniff`, WASM MIME type and immutable model caching passed; machine-readable evidence is in `results/public-deployment.json`.
- GitHub Actions passed for `2803fc9`: https://github.com/victorhuii/Mars-Rover-Terrain-Segmentation/actions/runs/43653680324. Cloudflare's GitHub check confirmed successful deployment from that commit. The release-documentation commit must also pass CI and deployment checks before the final handoff.
- Screenshots were refreshed from the public site. README includes the live link and repeatable live verification commands. The status log and portfolio case study record completed deployment.

## Limits and maintenance

The model is a research demonstration with weak big-rock performance and limited class support. The checks cover Chrome desktop and a mobile viewport, not every browser or physical phone. The 43 ms latency measurement is local and excludes loading and preprocessing; no public latency benchmark is claimed. Future source changes require local checks, GitHub CI, a successful Cloudflare deployment and public smoke tests. No further training or expert evaluation was needed for this release.

Screenshots captured from the public application are in `docs/assets/`, the full experimental record is in `docs/experiments.md`, and truthful resume bullets are in `docs/portfolio.md`.
