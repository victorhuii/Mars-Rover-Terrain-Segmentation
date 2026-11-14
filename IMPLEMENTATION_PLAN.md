# MarsTerrain implementation plan

The project is complete only after all milestones and the specification's definition of done are verified. Record evidence and exact continuation steps in PROJECT_STATUS.md and IMPLEMENTATION_LOG.md. Never use the expert test set to select a model.

- [x] 0. Inspect repository and environment; establish development configuration, smoke check and CI. Exit: smoke check passes and checkpoint is pushed (b9778ed). GitHub CI passed.
- [x] Feasibility spike: inspect 5–10 real, version-pinned AI4Mars records and their licenses and mask semantics; run load, transform, forward/backward, export and real browser WASM inference; check deployment asset limits. Exit: evidence supports architecture and identifies blockers.
- [x] 1. Validated, reproducible data ingestion, split manifests, masks and visualizations. Full image-only leakage audit, prepared data validation and local tests passed; GitHub CI passed.
- [x] 2. Train and validate U-Net baseline. Real MPS training, local checkpoint, loss, full validation metrics and qualitative inspection recorded; performance is poor and needs improvement.
- [x] 3. Train and compare a compact improved model; freeze selection and evaluate expert test once. Held-out metrics, per-class supports and local qualitative error analysis recorded; committed; CI passed.
- [x] 4. Export selected trained model and verify ONNX numerical and browser parity, metadata and size. Browser preprocessing parity and production packaging also passed.
- [x] 5. Build accessible, responsive interface with real sample and upload inference, overlays and tests. Verified locally in production Chrome.
- [x] 6. Integrate and verify end-to-end browser inference and local Python API; actual model and PyTorch-reference parity tested locally.
- [x] 7. Verify CI, browser E2E, accessibility, mobile and production build. Local checks and GitHub Actions run 43644643743 passed.
- [x] 8. Deploy publicly and verify real sample and uploaded-image inference in a fresh browser.
- [x] 9. Complete evidence-based README, research cards, screenshots, portfolio copy and final report.

Each verified milestone requires updated status and log, targeted tests, a meaningful commit and push to the existing remote main. If a required external login or Git access prevents progress, stop at that checkpoint with an actionable unblock. No paid service without approval.
