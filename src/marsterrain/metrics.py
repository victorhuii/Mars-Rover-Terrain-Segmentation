"""Segmentation metrics; classes absent in both truth and prediction are omitted."""
import numpy as np


def confusion_matrix(target, prediction, classes=4):
    target = np.asarray(target).reshape(-1)
    prediction = np.asarray(prediction).reshape(-1)
    if target.shape != prediction.shape:
        raise ValueError('Prediction and target dimensions differ')
    valid = target != 255
    target, prediction = target[valid], prediction[valid]
    if ((target < 0) | (target >= classes) | (prediction < 0) | (prediction >= classes)).any():
        raise ValueError('Out-of-range class ID')
    return np.bincount((target.astype(np.int64) * classes + prediction.astype(np.int64)), minlength=classes**2).reshape(classes, classes)


def summarize(matrix):
    matrix = np.asarray(matrix, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or np.any(matrix < 0):
        raise ValueError('Expected nonnegative square confusion matrix')
    true = matrix.sum(axis=1); predicted = matrix.sum(axis=0); correct = np.diag(matrix)
    union = true + predicted - correct
    iou = [float(correct[i] / union[i]) if union[i] else None for i in range(len(true))]
    dice = [float(2 * correct[i] / (true[i] + predicted[i])) if true[i] + predicted[i] else None for i in range(len(true))]
    mean = lambda x: float(np.mean([v for v in x if v is not None])) if any(v is not None for v in x) else None
    return {'confusion_matrix': matrix.tolist(), 'per_class_iou': iou, 'mean_iou': mean(iou),
            'per_class_dice': dice, 'mean_dice': mean(dice), 'pixel_accuracy': float(correct.sum() / matrix.sum()) if matrix.sum() else None,
            'support': true.tolist(), 'valid_pixels': int(matrix.sum()), 'absent_class_convention': 'omit classes absent from both truth and prediction'}
