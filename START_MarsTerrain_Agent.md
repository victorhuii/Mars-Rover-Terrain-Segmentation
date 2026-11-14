# Paste this message into your coding agent

Read the attached `Mars_Rover_Terrain_Segmentation_Agent_Context.md` completely and execute the entire specification in my **already initialized local GitHub repository** (existing project directory). I am giving you permission to make meaningful Git commits and `git push origin main` after each tested feature if your IDE terminal has Git credentials. Never force-push or create a nested repo.

Do not stop after planning, scaffolding, a partial UI, or a toy model. Build the real NASA AI4Mars image-segmentation pipeline, train and evaluate a genuine model without test leakage, validate PyTorch-to-ONNX parity, implement working image upload and per-pixel overlays, add automated Python/TypeScript/browser tests and CI, deploy the production app, verify the live website, and finish the README/model card/screenshot/case study and DS/SWE resume bullets. Treat the attached context as the acceptance criteria and work sequentially through its milestones. Explain each milestone and the actual code it implements succinctly.

My machine is an M1 MacBook Pro. Use `python3` for terminal commands. Prefer efficient PyTorch MPS training and a genuinely working browser-side ONNX Runtime deployment on Cloudflare Pages so the public upload demo does not need a paid backend. Check the current SDK docs and dataset license; don't invent results, paid services, or deployment claims. No fake model predictions. Preserve source attribution and explicitly report limitations. Use realistic, limited compute and fix build/test failures rather than ignoring them.

When Git credentials/permissions or hosting logins prevent you from completing a checkpoint, **pause only at that necessary checkpoint** and show me exactly:

git add .
git commit -m "Specific completed feature"
git push origin main

Tell me the concrete external UI step needed. After I perform it, continue from the next unfinished milestone. Do not ask routine questions that can be resolved by inspecting the repo.

Only mark the project finished after successful held-out evaluation, working real upload inference on the public site, green CI, portfolio documentation, and a final evidence-based report with verified URLs and metrics.
