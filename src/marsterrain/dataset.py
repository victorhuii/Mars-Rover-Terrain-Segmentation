"""Lossless AI4Mars mask decoding and exclusion masks."""
from __future__ import annotations

import io
import numpy as np
from PIL import Image

IGNORE = 255
CLASSES = ('soil', 'bedrock', 'sand', 'big rock')


def decode_image(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        return image.copy()


def decode_labels(image: Image.Image) -> np.ndarray:
    labels = np.asarray(image)
    if labels.ndim == 3 and labels.shape[2] in (3, 4):
        if not np.array_equal(labels[:, :, 0], labels[:, :, 1]) or not np.array_equal(labels[:, :, 0], labels[:, :, 2]):
            raise ValueError('RGB label channels disagree')
        labels = labels[:, :, 0]
    if labels.ndim != 2 or not np.isin(labels, [0, 1, 2, 3, IGNORE]).all():
        raise ValueError('Unexpected label values or shape')
    return labels.astype(np.uint8, copy=True)


def prepare_pair(image: Image.Image, label: Image.Image, rover: Image.Image,
                 distance: Image.Image, size: int = 256):
    if size <= 0 or any(mask.size != image.size for mask in (label, rover, distance)):
        raise ValueError('Image and mask dimensions must match and size must be positive')
    labels = decode_labels(label)
    for mask in (rover, distance):
        values = np.asarray(mask)
        if values.ndim != 2 or not np.isin(values, [0, 1]).all():
            raise ValueError('Auxiliary mask must be binary and single-channel')
        labels[values == 1] = IGNORE
    if not np.any(labels != IGNORE):
        raise ValueError('No valid labelled pixels')
    image = image.convert('RGB').resize((size, size), Image.Resampling.BILINEAR)
    label = Image.fromarray(labels).resize((size, size), Image.Resampling.NEAREST)
    if not np.any(np.asarray(label) != IGNORE):
        raise ValueError('No valid pixels after resizing')
    return image, label
