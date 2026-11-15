# Mars Rover Terrain Segmentation — personal website handoff

Use this document to add the completed project to Victor Hui's personal website. Match the website's existing layout, typography and project conventions. Keep the copy direct and concise.

## Project details

| Field | Value |
| --- | --- |
| Title | Mars Rover Terrain Segmentation |
| Short name | MarsTerrain |
| Status | Built and publicly deployed |
| Live demo | https://mars-rover-terrain-segmentation.pages.dev/ |
| GitHub | https://github.com/victorhuii/Mars-Rover-Terrain-Segmentation |
| Role | Data preparation, model training and evaluation, frontend, browser inference, testing and deployment |
| Tags | Computer Vision, PyTorch, React, TypeScript, ONNX |

## Project card copy

Trained a model to identify soil, bedrock, sand and large rocks in Mars rover images. Built a React app that runs predictions in the browser, with image uploads, terrain overlays and downloadable results.

Use **Live demo** and **GitHub** as the link labels. If the existing cards support a result line, use: **65.8% mean IoU on the frozen expert test.** Mean IoU is an overlap metric, not accuracy.

## Optional project detail copy

I built a terrain segmentation model using Curiosity rover images from AI4Mars, then deployed an interactive app for exploring its predictions. Visitors can choose an example or upload an image, compare the original with the predicted terrain mask, filter classes and download an overlay. Images are processed in the browser.

Before training, I audited all training and expert image shards and found that every unique expert image also appeared in the mirror's training data. I excluded matching images and conservatively grouped related scenes, then created reproducible training and validation manifests. I trained a compact model from scratch on my M1 MacBook Pro and selected it using validation results before evaluating the frozen model once on expert labels.

The model achieved 0.6581 mean IoU across four terrain classes on 287 expert-test images with valid labels. Large rocks were the weakest class, with 0.2066 IoU and limited labeled examples. This is an independent research demonstration; terrain predictions do not establish a safe rover route.

I exported the model to ONNX and checked its outputs against PyTorch. The React and TypeScript app uses ONNX Runtime Web for local inference and is hosted on Cloudflare Pages. Automated tests cover the data pipeline, model artifact and browser workflows, including actual uploaded-image predictions on the public site.

## Technical facts for a detail page

- Data: AI4Mars Curiosity images; 18,130 training rows audited against all expert image shards.
- Leakage handling: all 322 unique expert images appeared in the mirror's training rows; matching and conservatively linked training groups were excluded.
- Clean data pool: 9,154 training images and 1,557 validation images. The selected model trained in stages on up to 4,096 training images.
- Model: scratch-initialized MobileNetV3-small encoder with a compact segmentation decoder; 256 × 256 RGB input.
- Expert evaluation: `test_min3`, evaluated once after model selection; 287 of 322 images had valid labels. The other 35 were entirely ignored after exclusions.
- Results: mean IoU 0.6581, mean Dice 0.7560, pixel accuracy 0.8986. Per-class IoU: soil 0.8463, bedrock 0.8104, sand 0.7692, big rock 0.2066.
- Deployment: React, TypeScript, Vite, ONNX Runtime Web WASM and Cloudflare Pages. Uploaded images stay in the browser.
- Controls: original/mask/overlay views, terrain class filters, opacity adjustment, draggable comparison, predicted pixel coverage and PNG download.
- Verification at release: 27 Python tests, six frontend unit tests and two production browser tests passed. Public sample/reference-mask comparison and uploaded-image inference passed in fresh Chrome contexts, including a mobile viewport.
- The ONNX model is approximately 4.4 MB. Hosted model integrity, security headers and caching were verified.

## Screenshots

Download an actual screenshot into the personal website's own assets directory and use its image component and sizing conventions. Use the result screenshot as the preferred project image; the desktop screenshot is an alternative. Preserve the aspect ratio and preview the crop on mobile.

| Asset | URL | Suggested alt text |
| --- | --- | --- |
| Prediction result, preferred | https://raw.githubusercontent.com/victorhuii/Mars-Rover-Terrain-Segmentation/main/docs/assets/terrain-result.png | Mars rover image beside a predicted terrain mask and colored overlay. |
| Desktop interface | https://raw.githubusercontent.com/victorhuii/Mars-Rover-Terrain-Segmentation/main/docs/assets/terrain-lab-desktop.png | Mars Rover Terrain Segmentation web application with image selection controls. |
| Mobile interface | https://raw.githubusercontent.com/victorhuii/Mars-Rover-Terrain-Segmentation/main/docs/assets/terrain-lab-mobile.png | Mars Rover Terrain Segmentation application in a mobile layout. |

Image credit: **Rover imagery: NASA/JPL-Caltech; dataset: AI4Mars, Swan et al.** Retain attribution in the project detail or image caption as appropriate. This is a personal project with no NASA affiliation or endorsement.

## Supporting links

- Model card: https://github.com/victorhuii/Mars-Rover-Terrain-Segmentation/blob/main/docs/model_card.md
- Experiments: https://github.com/victorhuii/Mars-Rover-Terrain-Segmentation/blob/main/docs/experiments.md
- Architecture: https://github.com/victorhuii/Mars-Rover-Terrain-Segmentation/blob/main/docs/architecture.md
- Final report: https://github.com/victorhuii/Mars-Rover-Terrain-Segmentation/blob/main/FINAL_PROJECT_REPORT.md
- Verified release CI: https://github.com/victorhuii/Mars-Rover-Terrain-Segmentation/actions/runs/43655843633

## Accuracy and wording

- Use the project card copy for the main projects section. Add the longer explanation only if the website already supports project detail pages.
- Describe the result as 65.8% mean IoU, not 65.8% accuracy. Keep model results separate from predicted pixel coverage in the interface.
- Distinguish the clean training pool from the 4,096 images used by the selected model.
- Do not claim NASA employment, collaboration, operational rover use, safe-route planning or state-of-the-art performance.
- Do not advertise a universal inference speed. The recorded 43 ms median measured only five warm model calls in local headless Chrome, excluding preprocessing, loading and network time.
- Keep the title literal. Avoid promotional slogans, invented results and unnecessary paragraphs.

## Prompt for the personal website agent

Read the attached `PERSONAL_WEBSITE_HANDOFF.md` and add **Mars Rover Terrain Segmentation** to my personal website's projects section. Implement the change in the existing website repository and follow its local instructions.

Inspect how the current project cards, assets and optional detail pages work. Reuse those components and match the site's existing visual style. Use the supplied short project card copy, an actual screenshot, the technology tags, and working **Live demo** and **GitHub** links. Download the screenshot into the website's assets and provide descriptive alt text and the supplied image attribution. Include the 65.8% mean IoU result only if the card has room; label it correctly. If the site already has project detail pages, add a concise case study using the supplied data, model, deployment and limitations. Keep the copy plain and brief.

Use only facts in this handoff or its linked repository evidence. Verify both destination links, run the website's relevant checks and production build, and inspect the updated section at desktop and mobile widths. Follow the website repository's existing Git and deployment instructions. Finish with the changed files, checks performed and preview or public URL when available.
