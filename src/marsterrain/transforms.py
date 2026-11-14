"""Synchronized train augmentation; validation is deterministic."""
from PIL import Image, ImageEnhance


def transform_pair(image, mask, *, flip=False, brightness=1.0):
    if image.size != mask.size:
        raise ValueError('Image and mask dimensions must match')
    if brightness <= 0:
        raise ValueError('Brightness must be positive')
    if flip:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if brightness != 1.0:
        image = ImageEnhance.Brightness(image).enhance(brightness)
    return image, mask
