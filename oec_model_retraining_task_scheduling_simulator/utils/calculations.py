import numpy as np
from scipy.spatial import distance

# IoU score is calculated as intersection / union
def calculate_iou(gt, pred):
    intersection = np.logical_and(gt, pred)
    union = np.logical_or(gt, pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

# Dice score is calculated as 2 * intersection / (sum of gt and pred)
def calculate_dice_loss(gt, pred):
    gt = gt.flatten()
    pred = pred.flatten()
    if np.sum(gt) == 0 and np.sum(pred) == 0:
        return 1
    return 1 - distance.dice(gt, pred)

# entropy is calculated as -sum(p * log(p))
def calculate_entropy(output):
    if len(output.shape) == 3:
        c, w, h = output.shape
        b = 1
    elif len(output.shape) == 4:
        b, c, w, h = output.shape
    elif len(output.shape) == 2:
        w, h = output.shape
        b = 1
        c = 1
    else:
        raise ValueError("Invalid shape")
    log_sum = np.sum(output * np.log(output + 1e-6))
    if log_sum == 0:
        return 0
    entropy = (-1 / (b * c * w * h)) * log_sum
    return entropy