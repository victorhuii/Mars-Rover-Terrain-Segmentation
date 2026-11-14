# MarsTerrain — Autonomous Mars Rover Terrain Segmentation
## End-to-End Agent Build Specification (November 2026)

> **Purpose:** Give an IDE coding agent everything necessary to turn an already-existing, mostly empty local GitHub repository into a real, tested, deployed, recruiter-ready computer vision project. The agent should read this entire file and execute it; this document is a specification, not evidence that any deliverables have been implemented.

## 0. PRIMARY AGENT OPERATING INSTRUCTIONS

You are an autonomous senior ML engineer, computer-vision researcher, full-stack software engineer, QA engineer, and deployment engineer. Work within the **current already-existing Git repository and its existing empty project folder**. Never initialize a second nested Git repository, overwrite unrelated files, remove the original `.git`, invent research results, or treat a generated mock prediction as a trained model.

**Your final objective is a publicly accessible, portfolio-ready Mars terrain semantic-segmentation application with real learned predictions on both supplied examples and arbitrary valid uploaded images; reproducible training and evaluation; verified held-out results; tests, CI, docs, screenshots, and functional deployment.** Do not declare complete based solely on scaffolding or a pretty dashboard.

1. Inspect current repo, branch, remote, files, available hardware, disk capacity, Python/Node versions and authentication. Use the existing repository root. Create `IMPLEMENTATION_PLAN.md` with a feature checklist and exit criteria; update it continuously. Do not ask routine clarification questions—make defensible decisions and document them.
2. Run a small feasibility spike first: real image/mask data loading, one-model training step, PyTorch→ONNX export, ONNX Runtime Web WASM browser inference, Cloudflare Pages model-file size. The deployment architecture must be based on **working evidence**.
3. Implement in thin vertical slices. After each slice: format/lint, run relevant tests, verify the feature manually or with browser automation, update the plan, **make a meaningful Git commit and push** if repository terminal and Git authentication are available. The user has explicitly authorized these milestone pushes on the existing `main` branch. Do not force-push or amend published history. Do not create superficial commits just to inflate the GitHub contribution graph.
4. After each passing milestone, use only these commands for Git when practical:
   ```bash
   git add .
   git commit -m "Describe the finished feature clearly"
   git push origin main
   ```
   Ensure `.gitignore` excludes large raw data, environment secrets, caches and intermediate checkpoints. Never commit credential tokens. Before pushing, verify the remote and branch match the intended repository. Verify `git config user.name` and `git config user.email` use the user's intended GitHub identity/verified email; GitHub contribution visibility depends on author attribution and the branch, not just the number of commits. If remote `main` moved, pull with a safe reconciliation strategy and resolve conflicts; never force-push.
5. **If terminal Git push is unavailable, blocked by credentials, or user approval is required:** stop only at the completed, tested milestone; state exactly what changed; show precisely the three commands above with an appropriate commit message; wait for the user to push, then continue from the next milestone. Do not pretend to have pushed. When the assistant running in a chat cannot access local files or a Git remote, it cannot push the user's local repository; the IDE agent may be able to because it has local terminal and authentication.
6. Do not stop after planning or a partial demo. Continue through model training, rigorous evaluation, upload inference, visual app, tests, CI, production deployment, accessibility, README, screenshots, and portfolio copy. Exception: stop only for required login/permission, license uncertainty, cost approval, remote access, or irrecoverable resource limits. Give the user the smallest possible actionable unblock instructions, then resume once they act. **Never initiate paid services, paid training, paid APIs, or GPU purchases without explicit approval.** Do not claim to work asynchronously after the session ends.
7. Maintain a truthful `IMPLEMENTATION_LOG.md`: commands, tests, training dataset counts, versions, experiments, selected model, true metrics, failures, optimizations, deployment URLs, and unresolved issues. Use deterministic random seeds and a frozen final test protocol.
8. At completion, produce `FINAL_PROJECT_REPORT.md` with pass/fail evidence, live URLs (verified), Git commits, tests, model metrics, architectural explanation, actual limitations and concise resume bullets. If anything remains blocked, label it explicitly—not “complete.”

## 1. PROJECT GOAL AND REAL-WORLD CONTEXT

**Working title:** MarsTerrain — Mars Rover Terrain Segmentation (choose a consistent project name; do not claim any NASA affiliation).

**Problem:** Rover cameras capture Martian surface images containing loose soil, sand, exposed bedrock and large rocks. These terrain types influence rover mobility, wheel interactions, drive planning, and possible navigation hazards. Human operators analyze terrain, but manual interpretation limits scalability. Computer vision can give pixel-level terrain context to inform future terrain-aware autonomy. This is **not** a verified rover safety controller, and a semantic label does not by itself prove an area is traversable.

**What to build:** Train a semantic segmentation network to predict one terrain class for each image pixel, ignoring unlabelled/invalid areas; measure performance using expert-labelled held-out data; build a polished website where visitors choose included examples or upload a valid Mars landscape image, run the **actually trained model**, and inspect a color-coded mask and adjustable overlay against the image. Present per-class estimates and calibrated descriptions of limitations. Provide reproducible experiments, a Python API for local use, and a low-cost deployable inference path.

**Class vocabulary for chosen AI4Mars NAV version (verify mask format on ingestion):**
- `0` Soil
- `1` Bedrock
- `2` Sand
- `3` Big rock
- `255` No label / ignore (not a fifth terrain class)

Use a consistent, accessible four-color legend and separate visual handling for ignored pixels. Do **not** claim “safe/unsafe path” or automatic route planning unless you have distinct ground truth and evaluation for traversability and routing.

## 2. VERIFIED RESEARCH AND DATA SOURCES

**Primary scientific work:** Swan et al., *AI4MARS: A Dataset for Terrain-Aware Autonomous Driving on Mars*, CVPR Workshops 2021. https://openaccess.thecvf.com/content/CVPR2021W/AI4Space/html/Swan_AI4MARS_A_Dataset_for_Terrain-Aware_Autonomous_Driving_on_Mars_CVPRW_2021_paper.html

**NASA public catalogue:** https://data.nasa.gov/dataset/ai4mars-a-dataset-for-terrain-aware-autonomous-driving-on-mars

**NASA catalogue’s dataset distribution / official archive:** https://zenodo.org/records/15995036

**Third-party, convenient Curiosity mirror:** https://huggingface.co/datasets/hassanjbara/AI4MARS ; source files at https://huggingface.co/datasets/hassanjbara/AI4MARS/tree/main/data ; description at https://huggingface.co/datasets/hassanjbara/AI4MARS/blob/main/README.md

**Technical reference explaining mask values and file naming:** https://github.com/UnDauphin/ai4mars_DL-v2/blob/main/info.md

**Important source distinctions:** NASA's original catalogue/archive covers multiple rovers and its aggregate totals differ by version. The Hugging Face mirror presently documents **18,130 train examples and three separate expert test variants containing 322 images each** (`test_min1`, `test_min2`, `test_min3`); it is **not** the complete official multi-rover release, and those expert test variants may depict the same images with different consensus masks. It is about **6.47 GB total**. Record the exact mirror commit or dataset revision, checksum/manifest, source, license, split sizes, and chosen test variant. Do not add together test variants as independent images. The mirror lists CC0; the original dataset catalogue may have different or unspecified licensing metadata. Check source licenses, model weights licenses, and attribution before redistributing images/masks/weights. Cite Swan et al. and NASA data sources clearly.

**Default data strategy:** Use the Curiosity AI4Mars mirror for a practical first implementation, in streaming mode or download bounded shards if laptop disk permits; choose a **documented representative training subset** for initial experimentation. The project should work with the full available training split if hardware/time permits. Do not imply a 6.47 GB full download if only a subset was used. If the mirror changes, preserve a versioned manifest and adapt the loader. Prefer open, permission-compatible images; do not commit the full original dataset.

**Dataset guardrails:**
- Inspect real samples and actual PIL image modes, mask pixels, shape, metadata, and `has_labels` fields before writing hardcoded transforms. The mirror describes `image`, `label_mask`, optional `rover_mask` and `range_mask`, and `has_masks`/`has_labels`. Confirm whether labels arrive as single-channel `0..3/255` or triplicated RGB; convert losslessly to integer class IDs, never approximate colors or rescale masks like ordinary photos.
- Use **nearest-neighbor** interpolation on masks; bilinear/bicubic is allowed on photos. Preserve `ignore_index=255` across transforms and loss/metrics.
- Apply masks for rover pixels, beyond valid range, and unknown/unlabelled pixels when indicated by the source; confirm boolean polarity from data samples and documentation, rather than guessing.
- Deterministic, image-group-aware split of TRAIN only into `train` and `val` (e.g. 85/15). Prevent overlapping source image IDs or repeated crops of the same scene crossing splits. If sol/location/camera metadata exists, prefer stronger grouped split and document it.
- Keep expert-labelled `test_min3` untouched until a model, augmentations, threshold/palette, and evaluation procedure are frozen. The exact test variant choice is a default, not a claim that its mask quality is best. Do not iterate on test scores.
- Use strong train/val/test file manifests containing source IDs, group keys, hashes/counts. Never train on the expert test split.

## 3. PRIMARY TECHNICAL ARCHITECTURE

**Prioritize truly deployable, low-cost, real image inference.**

Recommended primary deployment: a compact PyTorch model exported to ONNX and run **in visitors' browsers using `onnxruntime-web` (WASM CPU fallback)**. Host the React/TypeScript/Vite UI, ONNX model and WASM runtime assets on Cloudflare Pages. This avoids requiring a hosted GPU or PyTorch server for uploaded-image predictions, and works for personal portfolio traffic. Validate this with a real browser spike, not assumption. ONNX Runtime supports in-browser inference: https://onnxruntime.ai/docs/tutorials/web/build-web-app.html ; deploy guidance: https://onnxruntime.ai/docs/tutorials/web/deploy.html . Cloudflare Pages docs: https://developers.cloudflare.com/pages/framework-guides/deploy-a-react-site/ ; free Pages asset file size limit currently 25 MiB: https://developers.cloudflare.com/pages/platform/limits/

**Fallback if ONNX Web fails feature/runtime/performance tests:** Develop a CPU inference backend using Python/FastAPI + ONNX Runtime CPU (not full PyTorch) and deploy only after verifying cold-start memory and runtime on the chosen host. Render Free currently advertises **512 MB RAM** and idle spin-down; do not assume this is sufficient. Use Cloudflare Pages + a validated backend host and a `VITE_API_BASE_URL` setting, CORS allowlist and informative cold-start UX. **Do not present a precomputed overlay for arbitrary uploads as if it were real inference.** A sample-only precomputed mode can exist as a *clearly labelled fallback*, but the project does not meet the uploaded-image inference completion criterion until actual predictions work publicly. Hugging Face Docker/Gradio Spaces now requires an eligible paid account to create compute Spaces even though CPU Basic hardware itself has no hourly charge—do not presume it is free to set up. Reference: https://huggingface.co/docs/hub/spaces-overview

**Codebase desired components:**
- Python data ingestion and validation, PyTorch training/evaluation, ONNX export/parity, optional FastAPI CPU inference.
- React + TypeScript + Vite UI; accessibility and responsive design.
- ONNX Runtime Web with lazy model initialization and UI progress/error handling, WebAssembly path packaging tested in production preview.
- GitHub Actions checks and reproducible Makefile/scripts.
- No paid database/auth needed. Never ship API keys client-side. Uploaded files should remain in-browser if using WASM inference (make this clear to users).

**Expected layout (adapt based on repo contents):**
```text
repo-root/
  .github/workflows/ci.yml
  .gitignore
  .env.example
  README.md
  LICENSE                    # only if appropriate for original code
  IMPLEMENTATION_PLAN.md
  IMPLEMENTATION_LOG.md
  FINAL_PROJECT_REPORT.md
  pyproject.toml             # or pinned requirements; consistent tooling
  configs/
    baseline.yaml
    experiment.yaml
  data/                      # raw/processed ignored; sample manifests tracked
  docs/
    assets/                  # vetted project screenshots/architecture visuals
    model_card.md
    dataset_card.md
    experiments.md
  notebooks/                 # optional EDA; reusable source remains in src/
  src/marsterrain/
    __init__.py
    dataset.py
    transforms.py
    models.py
    losses.py
    metrics.py
    train.py
    evaluate.py
    predict.py
    export_onnx.py
    visualization.py
  api/
    main.py                 # local/optional deployed FastAPI
  frontend/
    package.json
    src/
      components/
      lib/inference.ts
      lib/imageProcessing.ts
      lib/palette.ts
      pages/
    public/
      models/                # compact versioned deployment ONNX, if allowed
      samples/               # small attributed examples only
      ort/                   # pinned matching ONNX Runtime WASM if needed
  tests/
    test_dataset.py
    test_masks.py
    test_metrics.py
    test_training.py
    test_inference.py
    test_export.py
    test_api.py
  e2e/
    segmentation.spec.ts    # Playwright browser E2E
  scripts/
    inspect_dataset.py
    prepare_data.py
    smoke_train.py
    compare_onnx.py
    benchmark_browser.py
```

## 4. ML IMPLEMENTATION PLAN

### A. Data exploration & reproducibility
1. Record exact dataset source/revision, class counts, valid pixel proportions, class imbalance, image resolution distribution, per-image examples, train/val IDs, and official test selection; save EDA plots and inspection manifest.
2. Implement lazy/streaming loading and local caching to avoid storing 6+ GB unnecessarily. Handle truncated/corrupt images, missing labels, mismatched dimensions, all-ignore masks, zero valid pixels, duplicate IDs, and different image channels robustly.
3. Convert grayscale inputs to 3-channel only as required by encoder; normalize images using measured/declared conventions. If using pretrained ImageNet encoders, ensure matching normalization. Keep mask 255 ignore, no normalization.
4. Resize/crop default to 256x256 (or 320x320 if fast enough), synchronized image/mask transformations; use appropriate augmentations (horizontal flip when physically/semantically safe, mild photometric variations, small geometric transforms) without corrupting labels. Provide deterministic validation transforms.

### B. Real models and training
1. Implement lightweight **U-Net convolutional baseline** first, actually train it, log training and validation losses, sample predictions and per-class metrics. This ensures a real-model fallback and reveals loader mistakes. Include class imbalance handling via training-set-derived weights or carefully defined CE + Dice loss; `ignore_index=255` must be honored throughout.
2. Attempt stronger but compact **pretrained MobileNetV3-small encoder + U-Net-style decoder** (or another measurably feasible lightweight segmentation backbone). Do not automatically use an enormous vision transformer. Select final model using **validation mIoU and per-class behavior**, not the test labels or visual appearance alone.
3. Use M1 Mac PyTorch `mps` backend if `torch.backends.mps.is_available()`; CPU fallback. Have realistic batch sizes (~4–8 at 256px), mixed precision only if supported and validated, gradient clipping if needed, early stopping/patience, scheduler, checkpoints, seed, data-loader settings suitable for macOS. Avoid OOM and uncontrolled computation. A small smoke run must pass before long training. Note: MPS support depends on installed PyTorch and operators (https://docs.pytorch.org/docs/stable/notes/mps.html).
4. Run an experiment ladder: dataset/sample sanity check → overfit 2–4 images (confirm loss decreases substantially) → small subset baseline → full feasible train subset → improved architecture → validation selection → **one frozen expert test evaluation**. Use fixed configs, checkpoints and experiment log.
5. Continue toward the strongest sensible result under available laptop resources. If resources are limiting, use a smaller documented subset/model rather than replacing training with fake outputs. Keep training efficient and reproducible. No unsupported promise of any specific F1/mIoU.

### C. Evaluation and error analysis
Implement a confusion matrix over **only valid labelled pixels**. Calculate:
- **Mean IoU / mIoU** across specified present classes, with explicit absent-class convention.
- **Per-class IoU** for soil, bedrock, sand and big rock.
- **Mean Dice/F1**, per-class Dice, pixel accuracy, valid pixel counts, class support, model inference latency, model size.
- Visual original / reference mask / predicted mask / overlay examples, including success and failure cases.
- Model comparison table (baseline vs chosen model) on validation; frozen model and configuration on expert test. State data subset, split, model size, hardware, training time and test variant.

Guard against pixel accuracy dominating conclusions when classes are imbalanced. Evaluate rare big-rock class rather than hiding it. Don't claim navigation hazard probability or “safe driving” from segmentation alone.

### D. Export and parity
- Export chosen trained model to ONNX with fixed supported input size (prefer 256x256) and versioned metadata JSON (classes, input shape, normalization, palette, dataset revision, checkpoint SHA, commit SHA).
- Validate ONNX model with `onnxruntime` CPU and compare PyTorch/ONNX raw logits on multiple held-out validation examples within documented numeric tolerance. Verify argmax pixel agreement and preprocessing consistency.
- Test actual `onnxruntime-web` WASM loading in Chrome and Safari (or at minimum Chrome plus cross-browser fallback). Serve required WASM assets from valid build paths, handle CORS/headers, lazy-load, report initial model download and inference timing. Track size in MB and ensure the single model file is <25 MiB for Cloudflare Pages, or use a verified external source with immutable version/checksum and correct CORS. Quantize only if quality and browser parity remain acceptable.
- Do **not** ship random initialized or reference-only model weights as if they were trained. Check trained checkpoint and deployed ONNX are the same version.

## 5. WEBSITE PRODUCT SPECIFICATION

Build a distinct, polished Mars exploration visual identity (not a clone of MissionGuard): dark charcoal / muted red sandstone / earth-orange accents, clean typography, high contrast, limited animation, professional scientific UI. Avoid generic AI dashboard fluff, giant gradients, and endless feature cards.

**Main route: Terrain Lab**
- Hero: “See Mars through a rover’s eyes” with honest short mission statement (“research demonstration of terrain classification on historical rover imagery”).
- Choose 4–8 **licensed and attributed** sample Mars images from data actually allowed to redistribute; each has source/citation and optional expert label. Image gallery with thumbnails and distinct visual conditions (rocky, sandy, mixed, challenging).
- Accessible drag-and-drop and choose-file upload (JPEG/PNG/WebP if supported). Explicit file size/pixel limits; provide friendly errors for invalid/corrupt/huge images. Fix EXIF orientation, preserve aspect ratio, resize/letterbox per model configuration, map pixel labels back accurately to original displayed dimensions; no leaks of object URLs.
- “Analyze terrain” starts actual trained inference on the supplied image; model loading progress, running state, elapsed inference time and errors. No simulated predictions.
- Side-by-side **Original / Predicted mask / Overlay**, draggable before/after compare, opacity slider, toggles by terrain category, downloadable PNG composite. A legend explains soil/bedrock/sand/big rock and unknown.
- Per-class predicted coverage computed from valid output pixels (clearly labelled estimated pixel coverage); optional click/hover pixel class information. Do not call raw softmax probabilities calibrated confidence unless tested.
- When an expert ground-truth sample is selected, show a “Reference label” toggle and per-image IoU only where a correct corresponding mask exists; for uploads with no labels, no “accuracy” metric.
- Explicit note: “Research prototype; semantic classification only, not a rover navigation/safety system.”

**Research / Model page or clear section**
- Real model name, architecture, dataset source/version, data splits, actual test mIoU/per-class IoU/pixel accuracy, confusion matrix, training curves, qualitative examples and failure modes. Exact test counts, ignored pixel fraction, hardware and timing.
- Explain why soil/sand/bedrock/big rock distinctions matter to terrain-aware autonomous navigation (without conflating labels with traversability).
- Model version/date; dataset and citation credits; architecture diagram and reproducibility link to code.

**Technical/UI requirements**
- Responsive desktop and mobile (mobile layout supports image compare via tabs if slider isn't usable), keyboard-accessible controls, visible focus, accessible legend contrast, meaningful alt text, `prefers-reduced-motion` honored.
- No input tracking, auth or unnecessary persistence. Uploaded photos are processed locally if browser inference architecture succeeds. Show privacy statement accurately.
- Loading skeletons, errors, empty state, slow-device fallback, download failure handling; “Try an example” always works even if user does not upload.
- No fake “API connected” badge unless an actual API health check succeeds; no unverified NASA endorsement logos.

## 6. OPTIONAL BACKEND / ENGINEERING API

Implement a local FastAPI service for validation and production fallback where justified:
- `GET /health` returns model availability, version, inference mode.
- `GET /model-info` returns classes, input size, trained artifact hash and measured metrics manifest (do not fabricate).
- `POST /segment` accepts a multipart uploaded image, validates MIME **and actual magic bytes**/Pillow decoding, dimensions, max 10 MB or configurable cap, EXIF normalization, and returns compact PNG masks/overlays or a typed response with classes, percentages, latency and version. Strip metadata where appropriate; no storing uploads by default.
- `GET /samples`, `GET /metrics` grounded in tracked metadata. Use a bounded concurrency policy; lazy-load model once, cache by model version, no public training endpoint.
- CORS allowlist from env, error handling with clear HTTP status codes; don't expose filesystem paths or stack traces. API docs generated automatically.

If production UI performs **browser inference**, document that the API is for local reproducibility / alternative server inference and do not market the public site as calling a deployed API. If production needs backend inference, verify the complete route and actual cloud cold-start performance, CORS and timeouts.

## 7. TESTING, QUALITY, CONTINUOUS INTEGRATION

**Python unit tests with small synthetic fixtures and real tiny examples where permitted (CI must not need the entire dataset):**
- correct image-mask pairing and source ID uniqueness; train/val/test disjointness; no expert test in training;
- mask 0/1/2/3/255 decode, ignored 255 preservation on nearest-neighbor resize, rover/range exclusion, all-ignore handling;
- training transforms synchronized; model input `[B,3,H,W]` and output `[B,4,H,W]`; one gradient/backward pass, loss finite and decreases on tiny overfit;
- CE/Dice ignore math, confusion matrix and IoU/Dice/pixel-accuracy on hand-calculated toy tensors incl absent classes and zero valid pixels;
- reproducible seeded split; config parse and checkpoint metadata; train/eval CLI smoke;
- trained ONNX export smoke, PyTorch/ONNX parity when small fixture artifact available, corrupt/missing model errors;
- API `/health`, valid upload, invalid extension/content, oversized image, corrupt bytes, wrong dimensions, model-not-loaded, errors, response schema.

**Frontend tests:** TypeScript typecheck, ESLint, unit tests of palette/mask/overlay and image resizing, invalid upload flows, loading states and API/mock inference response validation. Playwright E2E on production build for sample selection, upload → segmentation → correct visible mask, opacity slider, class toggle, download, mobile viewport and keyboard navigation. Use a **real trained compact test artifact** in at least one deployment smoke, not fake mock only.

**CI (`.github/workflows/ci.yml`):** run on push to `main` and pull request; parallel Python tests and JS lint/typecheck/tests/build. Optionally Playwright smoke with pinned browsers. CI must be hermetic; no full data or GPU dependence, no hidden local files, no secret credentials. Publish test failures accurately and fix until green. Don't delete difficult tests simply to make CI pass.

**Minimum quality gates:** no failing tests; build passes; model parity passes; dataset split validation passes; all user-facing actions function; real upload inference works in production; live site tested in incognito and mobile; README accurately lists limitations; reasonable response times measured, not guessed.

## 8. DEPLOYMENT, COST AND RELEASE ENGINEERING

**Preferred Cloudflare Pages:** Connect the existing GitHub repo. If React lives in `frontend/`, set root `frontend`, build `npm run build`, output `dist`, production branch `main`. Configure model assets and WebAssembly packaging and route handling, verify no asset exceeds Cloudflare Pages free asset cap (currently 25 MiB). Use cacheable model filename containing version/hash; verify download/cache upgrades. Ensure GitHub pushes trigger new production builds. For browser inference the public site does **not** need a Render backend or a paid cloud GPU.

**If backend hosting required:** Prefer ONNX Runtime CPU. Test on chosen instance. Render Free may spin down and has 512 MB RAM; measure memory footprint, model load, per-image latency, upload limit and cold-start UX. Set frontend `VITE_API_BASE_URL` and server `ALLOWED_ORIGINS` as distinct environment variables; never use a relative `/api` without an actual proxy in production. Do not configure or buy upgraded tiers without explicit user approval. If deployment requires connecting accounts or secrets, stop at that precise user action and then resume.

**Release checklist:** public URL loads in incognito; sample segmentation and arbitrary upload generate real outputs; correct mask aspect/legend; no console/network errors; mobile check; download; `/health` if API deployed; CI green; README live links; training/validation/test metadata; screenshot assets (original + overlay + result), demo GIF or short recording if possible, captions and alt text; no bogus claims; confirm custom domain optional, not prerequisite.

## 9. PROJECT WRITING / DELIVERABLES

**README must include, in this order:**
1. Short problem statement (why identifying Martian terrain matters), screenshot and LIVE DEMO + GitHub badges.
2. Dataset and four class explanation including mask 255; NASA / AI4Mars attribution; data version/size/splits/licensing caveat.
3. What was built + sample GIF or annotated screenshot.
4. Technical architecture diagram: NASA imagery → preprocessing → training/eval → ONNX export → client inference or API → React overlay.
5. Model design and why chosen, real validation/test metrics (mIoU per class, macro Dice, pixel accuracy, latency, model size); honest failed experiments.
6. UI and engineering features + API documentation if applicable.
7. Fully reproducible setup (M1/MPS and CPU, env variables, training, evaluation, ONNX export, frontend dev, build, tests, deploy).
8. Limitations, what predictions mean, possible future work, funding/affiliation disclaimer and scientific citation.

**Dedicated research documentation:** `docs/dataset_card.md`, `docs/model_card.md`, `docs/experiments.md`, `docs/architecture.md`; store generated metrics in CSV/JSON from actual runs. Include dataset version hashes and seed, data subsets, failures and exact test split.

**Portfolio website copy:**
- **Card (two sentences):** “An interactive Mars rover computer-vision project that identifies soil, sand, bedrock and large rocks in rover camera images. I trained and evaluated a pixel-level segmentation model using NASA's AI4Mars dataset and built a deployed application to explore its predictions on sample and uploaded images.” Only claim actual achieved functionality.
- **Full case study:** real-world problem → data & labels → model & experiments → meaningful results & limitations → deployment & technical system → screenshots & links. Don't say a segmentation mask guarantees safe navigation.
- **Resume bullets:** write two variants using **only measured results**. Example templates, replace bracketed fields after implementation:

  **Data Science / Computer Vision**
  - “Trained and evaluated a [architecture] semantic-segmentation network on [N] Mars rover images, achieving [measured test mIoU]% mean IoU across soil, bedrock, sand, and rock classes.”
  - “Built a reproducible PyTorch computer-vision pipeline with [validations/augmentations], class-aware evaluation, ONNX export, and error analysis on [held-out test size] expert-labeled images.”

  **Software Engineering / Full Stack**
  - “Built and deployed an interactive React/TypeScript Mars terrain analysis app with image uploads, pixel-level segmentation overlays, class filtering, and downloadable visualizations.”
  - “Implemented [actual inference architecture: ONNX Runtime Web/FastAPI], automated Python and frontend tests ([verified count]), GitHub Actions CI, and reproducible model builds/deployment.”

Do not insert unverified N or mIoU; omit metrics until measured. No fictitious performance gains or “NASA-grade accuracy.”

## 10. FEATURE MILESTONES + REQUIRED GIT CHECKPOINTS

Mark a milestone complete only when tested; make one meaningful commit after each. Use actual Git privileges when available; otherwise stop and ask the user to run the standard three commands, then resume.

| # | Milestone | Evidence to produce | Commit message example |
|---|---|---|---|
| 0 | Repo inspection, config, CI skeleton, plan | Baseline smoke and setup commands | `Initialize MarsTerrain architecture and development plan` |
| 1 | Versioned AI4Mars data loader, masks, sample visualizations | Correct class mapping, split checks, unit tests | `Implement validated AI4Mars data pipeline` |
| 2 | Trainable U-Net baseline | Real loss decrease, checkpoint, validation results | `Train and evaluate terrain segmentation baseline` |
| 3 | Improved compact model + frozen held-out evaluation | Per-class IoU, mIoU, confusion matrix, images, log | `Benchmark compact segmentation model on expert labels` |
| 4 | ONNX export and true inference parity | PyTorch vs ONNX checked, browser WASM spike | `Export verified segmentation model for web inference` |
| 5 | Polished interactive frontend | Upload/sample/overlay/legend/tests | `Build interactive Mars terrain segmentation interface` |
| 6 | FastAPI local service and/or browser integration | Real image end-to-end, negative tests, latency measured | `Integrate validated image inference and API` |
| 7 | CI, security, mobile, accessibility, browser E2E | Green checks and production bundle | `Add automated quality gates and browser tests` |
| 8 | Public deployment | Verified public URL; actual uploads work; errors handled | `Deploy and validate live Mars terrain explorer` |
| 9 | README/screenshots/model card/case study/resume bullets | Verifiable portfolio audit and final report | `Document MarsTerrain research and portfolio demo` |

## 11. DEFINITION OF DONE (DO NOT END EARLY)

- [ ] Repo is synced with remote `main`; meaningful milestone commits exist and are not fabricated.
- [ ] Real, licensed/attributed Mars rover data loaded; masks and ignore pixels valid; no image ID leakage.
- [ ] At least one **trained** segmentation model has a real checkpoint and evaluation; held-out metrics calculated without leakage.
- [ ] ONNX or alternative production inference matches training preprocessing and model outputs; uploaded **new** image actually produces segmentation.
- [ ] Responsive, accessible website supports examples and uploads, original/mask/overlay comparisons, opacity, legend, class toggles, output downloads, accurate explanations.
- [ ] Browser and/or API deployment works in a fresh incognito window on a public URL; no mock inference passed off as real.
- [ ] Python/unit/frontend/E2E tests pass and GitHub Actions is green.
- [ ] README + dataset/model cards + architecture diagram + screenshots + benchmark tables + setup/deployment instructions exist.
- [ ] Data licensing and scientific attribution are correct, and application limitations are transparent.
- [ ] Data Science and SWE resume bullets use only *observed and supported* results.
- [ ] `FINAL_PROJECT_REPORT.md` includes link, real model results, runtime/model size, test counts, commits, cost/hosting decisions and outstanding caveats.

If external authentication or a paid plan prevents completion, document the blocker and give a concrete manual action; resume after user intervention. Otherwise keep executing until every box is met. Never claim deployment, GitHub push, model performance or testing without successful verification.

## 12. FIRST ACTIONS (START IMMEDIATELY)

1. Print repo root/branch/remote and `git status`, inspect disk/RAM/MPS and existing files. Confirm intended repo without asking for information already visible.
2. Write plan and `.gitignore`; check Python/Node versions; scaffold minimal tests/CI.
3. Retrieve and inspect 5–10 **real** AI4Mars examples and mask values. Document mask class mapping, valid pixels, grouped split strategy and source revision.
4. Prove an end-to-end micro-run (load → transform → forward → loss → backward → render mask) and ONNX export/WASM feasibility before building the entire site.
5. Implement remaining milestones and keep going to production and final report, committing/pushing after each verified feature as specified above.
