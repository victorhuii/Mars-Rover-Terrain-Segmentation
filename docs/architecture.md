# Architecture (work in progress)

```mermaid
flowchart LR
  A[Version-pinned AI4Mars images and labels] --> B[Image identity audit and grouped split]
  B --> C[Training masks and exclusion masks]
  C --> D[PyTorch training]
  D --> E[Validation selection]
  E --> F[Frozen one-time expert evaluation]
  E --> G[ONNX export and parity checks]
  G --> H[Planned browser WASM inference]
  H --> I[Planned image overlay and legend]
```

The audit, train/validation preparation, baseline and compact model experiments exist. A guarded expert evaluation and trained-baseline ONNX/Chrome tensor parity have been tested locally. The scratch model and protocol were frozen before a one-time expert evaluation. Upload preprocessing, application and public deployment remain unfinished. Local raw data, cached prepared images and intermediate checkpoints are ignored by Git; tracked manifests bind the source and splits to checksums.
