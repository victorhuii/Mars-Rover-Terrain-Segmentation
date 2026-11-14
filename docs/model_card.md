# Model card (selection pending)

MarsTerrain is a research demonstration of semantic segmentation in historical Curiosity rover imagery. It is not a rover navigation or safety controller, and terrain class alone does not establish traversability.

The candidates predict soil, bedrock, sand, or big rock; 255 denotes an ignored reference pixel and is not a predicted terrain class. Models were trained using audited AI4Mars train data and compared against a disjoint validation split. Details, source revision, split limitations and licensing are in [the dataset card](dataset_card.md); configurations and actual validation results are in [the experiment record](experiments.md).

The selected candidate is a scratch-initialized MobileNetV3-small encoder with a compact skip decoder, trained in stages on up to 4,096 clean training images. The checkpoint SHA-256 is `d8c8ceb7dc2a535aeeb0b6246fc22d4f6bab306dff32fa1664476af9c821ca8f`. Full validation mIoU is 0.6312, with per-class IoU 0.7111 / 0.8808 / 0.6935 / 0.2395 in class order; big rocks remain difficult. The input is 256 × 256 RGB with values divided by 255; ImageNet mean/std normalization is part of the model even though its weights were initialized from scratch. The frozen protocol is in `experiments/frozen-model.json`. Expert-test labels have not yet been evaluated, and this model is not deployed.
