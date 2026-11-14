# AI4Mars data and split protocol (in progress)

The selected source is the Curiosity subset in the [AI4Mars Hugging Face mirror](https://huggingface.co/datasets/hassanjbara/AI4MARS), revision `93dbb79de6cd234799750b6129fbfb3b197582fa`. Cite Swan et al., [AI4Mars: A Dataset for Terrain-Aware Autonomous Driving on Mars](https://openaccess.thecvf.com/content/CVPR2021W/AI4Space/html/Swan_AI4MARS_A_Dataset_for_Terrain-Aware_Autonomous_Driving_on_Mars_CVPRW_2021_paper.html), and the [official Zenodo record](https://zenodo.org/records/15995036). The mirror identifies CC0, while the official record API reports CC BY 4.0 and NASA's catalogue does not specify a license. Retain attribution and verify the rights for any redistributed images and model artifacts.

Labels are soil 0, bedrock 1, sand 2, big rock 3 and ignore 255. Rover and over-30-m range masks indicate excluded pixels with 1; see the [dataset format description](https://github.com/UnDauphin/ai4mars_DL-v2/blob/main/info.md). We read expert *images only* to guard against leakage; expert labels are reserved for the frozen final evaluation.

`scripts/audit_dataset.py` scans all 13 training and all three expert image shards and generates a versioned image manifest and checksums. It links identical product names, byte hashes and decoded-pixel hashes, plus product filename `F` plus seven digits as a conservative scene proxy. It also links images whose 64-bit difference hashes differ in at most two bits; transitive components containing any expert image are excluded from train and validation. The remaining components are deterministically assigned by a seeded hash. This proxy and perceptual hashing can both over-group and miss related viewpoints; they do not prove geographic independence. The three test variants are alternate labels for the same evaluation population, not three independent test sets. Final evaluation is planned for `test_min3` only.

The preparation script reads training masks only after the audit, validates pairing and class values, applies both exclusion masks, resizes image bilinearly and mask by nearest neighbor, and rejects empty, corrupt or missing annotations. The resulting cached data are local, ignored by Git and should not be mistaken for a complete dataset release. The completed audit found all 322 unique expert images in the train shards by filename and content; 6,837 training rows are linked to expert components, including unlabeled rows. The initial assignments include 9,230 train and 1,563 validation records before annotation validation. After annotation validation, 9,154 train and 1,557 validation images remain; 80 rows had no valid labeled pixels, one lacked a required mask, and one had no valid pixels after resizing. Pixel counts at 256 × 256, including ignored pixels, are recorded in `data/manifests/prepared_summary.json`; see `docs/assets/class_distribution.png` for valid-pixel class proportions. The compressed prepared manifest records each usable source ID, component, split, paths and pixel counts.

To reproduce the audit (approximately 6.5 GB of source downloads plus local cache):

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install .
python3 scripts/download_data.py
python3 scripts/audit_dataset.py
python3 scripts/verify_manifest.py
python3 scripts/prepare_data.py
python3 scripts/render_training_examples.py
python3 -m pip install ".[eda]"  # optional plot dependency
python3 scripts/plot_data_distribution.py
python3 -m unittest discover -s tests -v
```

The audit saves image-only per-shard checkpoints under ignored `data/cache` and a compressed identity/split manifest in `data/manifests`. The summary includes SHA-256 checksums for every downloaded shard. The preparation output and training examples stay in ignored `data/processed`; they should not be redistributed until source and attribution requirements are reviewed. Rerunning preparation targets the same filenames. If source data, preparation parameters or code change, choose a fresh output directory to avoid stale artifacts.

## Redistributed demonstration images

The four small examples in `frontend/public/samples` are source images from audited training records, not the expert test. Their original product identifiers, SHA-256 hashes, credits and source paths are in `frontend/public/samples/samples.json`; `scripts/export_samples.py` verifies and recreates them. Images: NASA/JPL-Caltech; AI4Mars, Swan et al. The official Zenodo record reports CC BY 4.0. The processed website screenshots in `docs/assets/terrain-*.png` depict those examples and should retain this attribution.
