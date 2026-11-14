# Experiments

Expert `test_min3` labels were held out during training and model selection, and evaluated once after the protocol was frozen. Training diagnostics, validation measurements and the final expert result are identified below. Class order is soil, bedrock, sand, big rock. mIoU omits a class only when it is absent from both truth and prediction.

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

## Compact model comparisons

A MobileNetV3-small encoder and compact skip decoder was tested on the same 512 training and 256 validation records as baseline v1, with inverse-square-root class weights computed from those training records. The ImageNet-pretrained version used the official `IMAGENET1K_V1` weights (SHA-256 `047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`) and embedded the corresponding ImageNet normalization in the model. Best subset mIoU was 0.7179 at epoch 12; full-validation mIoU was 0.6978 with soil / bedrock / sand / big-rock IoU 0.7911 / 0.9161 / 0.7944 / 0.2896. These are validation measurements, not expert-test scores. Full metrics are in `results/mobilenet-v1-validation.json`. The upstream [TorchVision documentation](https://docs.pytorch.org/vision/main/models.html) cautions that pretrained models may have separate terms from the datasets used to train them; redistribution of these particular pretrained-derived weights has not been verified.

A scratch-initialized MobileNetV3-small experiment using the same subset reached best validation-subset mIoU 0.4826 at epoch 10. A continuation from its scratch-trained checkpoint on 2,048 clean training records was then evaluated. It is being evaluated as an alternative that does not depend on the ImageNet pretrained weights. The expert labels remained untouched through full validation and model selection.

Scratch v2 continued the scratch v1 checkpoint on 2,048 training images, with the same fixed 256-image validation subset and class weighting. Its best subset checkpoint (epoch 6) scored full-validation mIoU 0.5800, mean Dice 0.6968, and soil / bedrock / sand / big-rock IoU 0.6678 / 0.8358 / 0.6468 / 0.1697. The complete matrix and class supports are in `results/mobile-scratch-v2-validation.json`, with selected IDs and history in `experiments/mobile-scratch-v2/`. This remains substantially behind the pretrained candidate, particularly on big rocks. A bounded continuation on 4,096 training images at a lower learning rate followed to improve the scratch-trained alternative. Checkpoints now contain optimizer and random state for resumption.

Scratch v3 continued v2 on 4,096 clean training images at learning rate 0.00015 for six epochs (467.4 seconds total on MPS); its best checkpoint was epoch 5. On the entire 1,557-image clean validation split, mIoU was 0.6312, macro Dice 0.7433, pixel accuracy 0.8944, and soil / bedrock / sand / big-rock IoU was 0.7111 / 0.8808 / 0.6935 / 0.2395. It improved all four classes over scratch v2 and was selected as the best evaluated scratch candidate. Its validation mIoU is lower than the pretrained comparison; the latter was excluded from public-artifact selection because redistribution terms for its pretrained weights remain unresolved. This decision and the exact checkpoint/data hashes are frozen in `experiments/frozen-model.json` before expert evaluation. Big-rock performance remains a material weakness.

The selected checkpoint is 4,707,191 bytes. A 30-iteration single-image PyTorch forward benchmark after five warmups measured median 9.92 ms on MPS and 39.48 ms on CPU with one CPU thread; these timings exclude image decoding, model loading, and browser execution. Full results and hardware conditions are in `results/mobile-scratch-v3-{mps,cpu}-latency.json`. All 153 learned parameter tensors differ from the recorded v2 initialization checkpoint, as checked by `scripts/verify_training_update.py`.

A preliminary 4,385,772-byte ONNX export of this selected checkpoint matched PyTorch on three clean validation tensors within 1.32e-5 absolute logit error and identical argmax predictions. Chrome WASM on one such tensor matched within 1.10e-5 with identical argmax. Browser image decoding, resize/letterboxing and production packaging are not yet tested. The expert test was untouched until the frozen protocol was committed.

Scratch continuation commands after preparing data and training scratch v1:

```sh
python3 -m marsterrain.train --output checkpoints/mobile-scratch-v2 --architecture mobilenet_v3_small --init-checkpoint checkpoints/mobile-scratch-v1/best.pt --class-weights inverse-sqrt --limit-train 2048 --limit-val 256 --epochs 8 --patience 4 --batch-size 4 --lr 0.0003 --device mps
python3 -m marsterrain.train --output checkpoints/mobile-scratch-v3 --architecture mobilenet_v3_small --init-checkpoint checkpoints/mobile-scratch-v2/best.pt --class-weights inverse-sqrt --limit-train 4096 --limit-val 256 --epochs 6 --patience 4 --batch-size 4 --lr 0.00015 --device mps
python3 -m marsterrain.evaluate --checkpoint checkpoints/mobile-scratch-v3/best.pt --device mps --output results/mobile-scratch-v3-validation.json
```

## Frozen expert evaluation

After the selection and protocol were committed as `f0ee18a`, the frozen scratch v3 checkpoint was evaluated once on expert `test_min3`. Of 322 images, 287 had valid labeled pixels after rover/range exclusions; 35 were entirely ignored and contributed no pixels. Across 7,310,367 valid pixels, test mIoU was 0.6581, mean Dice 0.7560, pixel accuracy 0.8986, and soil / bedrock / sand / big-rock IoU was 0.8463 / 0.8104 / 0.7692 / 0.2066. Class supports were 3,754,206 / 1,636,933 / 1,914,376 / 4,852 pixels, respectively. Among pixels in the 287 evaluated images, 61.13% were ignored. The very small big-rock support makes its estimate especially fragile.

The full confusion matrix, per-image scores, ignored-pixel counts and skipped-image reasons are in `results/expert-test-min3.json`. Four locally generated examples, including low- and high-IoU cases, were inspected. Expert annotations cover only parts of many images; per-image IoU can be perfect over a small labeled area while the rest remains unevaluated. The checkpoint, preprocessing and model choice have not been changed after seeing the test results.
