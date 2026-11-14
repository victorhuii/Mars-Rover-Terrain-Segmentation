# Portfolio notes

[Live application](https://mars-rover-terrain-segmentation.pages.dev/) · [Source](https://github.com/victorhuii/Mars-Rover-Terrain-Segmentation). Public sample and upload inference passed fresh Chrome desktop and mobile viewport tests.

## Project card

I built an interactive computer vision project that identifies soil, bedrock, sand and large rocks in Mars rover images. I trained and evaluated a pixel-level model on AI4Mars data, then deployed a browser-based interface for exploring its predictions on examples and uploaded images.

## Case study

**Problem.** Understanding terrain is part of planning how a rover interacts with the Martian surface. I wanted to explore how pixel-level classification could make historical rover imagery easier to inspect, while being clear that a terrain label does not establish a safe route.

**Data.** I used the Curiosity portion of AI4Mars and audited all training and expert image shards before training. Every unique expert image also appeared in the mirror's train split, so I excluded exact matches and conservatively grouped related images before creating train and validation manifests. The final clean pool contains 9,154 training and 1,557 validation images; the selected model used up to 4,096 training images.

**Model and results.** I trained a compact segmentation model from scratch and selected it on validation data. After freezing the model and evaluation protocol, I evaluated it once on expert `test_min3` labels: 0.6581 mean IoU across four classes on 287 images with valid labels. Big rock achieved 0.2066 IoU and had only 4,852 labeled pixels, so its result needs particular caution. A pretrained comparison performed better on validation, but I did not select it for redistribution because its weight terms were not established.

**System.** I exported the selected checkpoint to ONNX and checked its logits against PyTorch and Chrome WASM. The React/TypeScript interface runs inference locally, supports uploads and examples, and lets visitors compare the original, mask and overlay. Automated tests cover the data pipeline, model artifact, browser flow, accessibility and invalid uploads. [Architecture](architecture.md) · [Experiments](experiments.md) · [Model card](model_card.md)

**Deployment.** Cloudflare Pages builds the React application from GitHub. The trained ONNX model runs in the visitor’s browser. Public tests verified sample predictions against a saved PyTorch mask, uploaded-image inference, class filters, comparison, PNG download and accessibility. The model hash and deployed assets match the tested production build. Desktop, result and mobile screenshots are in `docs/assets/`.

## Resume bullets

### Data science / computer vision

- Trained and evaluated a scratch-initialized semantic-segmentation model on up to 4,096 AI4Mars rover images, achieving 0.658 mean IoU on the frozen expert test across four terrain classes.
- Audited 18,130 training rows against expert imagery, removed exact and grouped leakage, and built a reproducible PyTorch-to-ONNX pipeline with class-level IoU, Dice and error analysis.

### Software engineering / full stack

- Built and deployed a React and TypeScript terrain exploration app with local image uploads, ONNX Runtime Web inference, pixel overlays, class filters and PNG downloads.
- Added 27 Python tests, six frontend unit tests and two browser tests, artifact hash checks and GitHub Actions workflows for a reproducible computer vision application.

Browser checks cover Chrome desktop and a mobile viewport; they do not establish compatibility with every mobile device or browser. Research limitations and measured results are documented in the model card.
