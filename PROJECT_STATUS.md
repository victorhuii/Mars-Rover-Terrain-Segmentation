# Project status

Updated: 2026-11-14

## Completed work

- Read both specifications, inspected the existing repository and toolchain, created the milestone plan, ignore rules, smoke test and CI skeleton.
- Committed and pushed milestone 0 as `b9778ed` to the configured origin/main.
- Pinned AI4Mars mirror revision `93dbb79de6cd234799750b6129fbfb3b197582fa`; downloaded all 13 train shards and three expert variants locally. The image-only full audit is complete; expert annotations remain unread.

## Verification and findings

- `python3 -m unittest discover -s tests -v`: 1 smoke test passed. GitHub CI passed for milestone 0: https://github.com/victorhuii/Mars-Rover-Terrain-Segmentation/actions/runs/43597529495.
- MPS is available outside the sandbox; inside the sandbox it reports unavailable. Python 3.14.2, Node 24.14.1, npm 11.11.0, PyTorch 2.15.0; 62 GiB disk free at inspection.
- Full audit: all 322 unique expert images also occur in train by filename and content. Conservative grouping marks 6,837 of 18,130 train rows as expert-related; 9,230 labeled rows are assigned to train and 1,563 to validation before mask validation. Three non-image files are excluded. Split identity, scene, hash and near-duplicate checks passed. Expert labels remain unexamined.
- Mirror lists CC0; official Zenodo record lists CC BY 4.0. NASA catalogue leaves license unspecified. Attribution and redistribution terms require care.
- Local dataset shards and audit cache are ignored and are not in Git; hashes and source details are in IMPLEMENTATION_LOG.md. The audit found a non-image `current.txt`; it is excluded and completed shard scans are checkpointed.
- The data pipeline has 13 passing local Python tests and a passing clean-install Chrome smoke; the expanded workflow has not yet run on GitHub. Preparation yielded 9,154 train and 1,557 validation samples, excluding 82 further unusable annotations. Counts and manifests are tracked under `data/manifests`; raw/prepared imagery remains ignored.
- A single-update real-image CPU spike exported ONNX with max logit error 6.71e-8 on an audited clean training record. Chrome WASM loaded that experimental model and produced finite output of the expected shape on a zero tensor using a temporary server. This is not a trained model or production parity test. The self-contained, dynamically ported browser smoke check now passes in Chrome.

## Remaining

Training, evaluation, production inference, interface, expanded CI verification, deployment and portfolio deliverables remain. No model, measured performance or deployed application exists yet.

## Exact next step

Commit and push the verified data milestone, verify its GitHub Actions run, then implement and run the seeded baseline training ladder starting with a 2–4-image overfit test. Do not inspect expert labels before freezing model selection. Expand browser parity to actual image preprocessing and trained outputs. Verify licensing/attribution for redistributed assets. See IMPLEMENTATION_PLAN.md and IMPLEMENTATION_LOG.md for details.
