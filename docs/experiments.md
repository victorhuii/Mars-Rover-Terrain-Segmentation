# Experiments

Expert `test_min3` labels are held out and have not been inspected. Results below are training diagnostics or validation measurements only. Class order is soil, bedrock, sand, big rock. mIoU omits a class only when it is absent from both truth and prediction.

## Four-image diagnostic

The first 40-update attempt with 8 channels did not overfit: training loss decreased from 2.0487 to 1.8469 while learning-rate scheduling reduced the learning rate prematurely. The diagnostic was repeated after disabling that scheduler for overfit mode, with 16 channels, 128-pixel inputs and 160 updates. Training loss fell from 2.1987 to 0.5509. These four images were also used for diagnostic evaluation, so those scores are not validation results. Configuration and epoch history are in `experiments/overfit-v2/`.

## Baseline v1

A U-Net with 16 initial channels was trained on 512 seeded clean training images, 256 × 256 resolution, batch 4, AdamW learning rate 0.001, cross entropy plus soft Dice and simple image augmentation. Selection used a fixed 256-image validation subset; training stopped after five epochs by the configured early-stopping rule. The best checkpoint was epoch 1. The complete clean validation split was evaluated once for this baseline:

| Metric | Validation result |
| --- | ---: |
| Images | 1,557 |
| mIoU | 0.1663 |
| Soil / bedrock / sand / big-rock IoU | 0.0000 / 0.6653 / 0.0000 / 0.0000 |
| Mean Dice | 0.1998 |
| Pixel accuracy | 0.6653 |
| Valid labeled pixels | 57,597,788 |
| Checkpoint size | 1,997,341 bytes |
| Full-validation elapsed time including data loading, MPS | 14.42 s |

The checkpoint is severely biased toward bedrock and is not suitable as a final model. High pixel accuracy reflects class imbalance and is not evidence of good segmentation. Full machine-readable metrics, support and confusion matrix are in `results/baseline-v1-validation.json`; configuration, selected IDs and epoch history are in `experiments/baseline-v1/`. The local checkpoint is ignored by Git; its SHA-256 is recorded in the metrics. The runtime includes data loading and is not a browser latency benchmark.

Reproduce after preparing the data as described in the dataset card:

```sh
python3 -m pip install ".[training]"
python3 -m marsterrain.train --output checkpoints/overfit-v2 --overfit --limit-train 4 --epochs 160 --patience 160 --channels 16 --size 128 --batch-size 4 --lr 0.001 --device mps
python3 -m marsterrain.train --output checkpoints/baseline-v1 --limit-train 512 --limit-val 256 --epochs 10 --patience 4 --channels 16 --size 256 --batch-size 4 --lr 0.001 --device mps
python3 -m marsterrain.evaluate --checkpoint checkpoints/baseline-v1/best.pt --device mps --output results/baseline-v1-validation.json
```

Use `--device cpu` when MPS is unavailable. Seeded sampling and checkpoints help reproducibility, but exact numerical results can vary by hardware and library version. The next experiment should address class imbalance and compare the same validation subset, then reevaluate the selected checkpoint on the full validation split. Do not select on expert test labels.
