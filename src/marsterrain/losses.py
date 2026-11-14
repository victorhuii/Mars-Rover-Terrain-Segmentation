import torch
import torch.nn.functional as F


def segmentation_loss(logits, target, class_weights=None):
    valid = target != 255
    if not bool(valid.any()):
        raise ValueError('Cannot train on an entirely ignored batch')
    ce = F.cross_entropy(logits, target, weight=class_weights, ignore_index=255)
    labels = target.masked_fill(~valid, 0)
    onehot = F.one_hot(labels, num_classes=logits.shape[1]).permute(0, 3, 1, 2).to(logits.dtype)
    valid = valid.unsqueeze(1)
    probabilities = logits.softmax(dim=1) * valid
    onehot = onehot * valid
    intersection = (probabilities * onehot).sum(dim=(0, 2, 3))
    denominator = (probabilities + onehot).sum(dim=(0, 2, 3))
    present = onehot.sum(dim=(0, 2, 3)) > 0
    dice_loss = 1 - ((2 * intersection[present] + 1e-6) / (denominator[present] + 1e-6)).mean()
    return ce + dice_loss
