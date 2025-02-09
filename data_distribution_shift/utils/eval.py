import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm
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


def eval_UNet(net, dataset, device='cpu'):
    """Evaluation without the densecrf with the dice coefficient"""
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    net = net.to(device)
    net.eval()
    dice_value_list = []
    entropy_value_list = []
    with torch.no_grad(), tqdm(total=len(dataloader)) as progress_bar:
        for batch_idx, (imgs, true_masks, im_name) in enumerate(dataloader):
            imgs = imgs.to(device=device)
            masks_pred_raw = net(imgs)[0]
            masks_pred = (masks_pred_raw > 0.5).float()
            # get data back to numpy
            masks_pred = masks_pred.cpu().numpy()
            # print("masks_pred max", np.max(masks_pred), "min", np.min(masks_pred), "unique", np.unique(masks_pred))
            true_masks = true_masks[0].numpy()
            # print("true mask shape", true_masks.shape, "unique", np.unique(true_masks), "max", np.max(true_masks), "min", np.min(true_masks))
            # print("masks_pred shape", masks_pred.shape, "unique", np.unique(masks_pred), "max", np.max(masks_pred), "min", np.min(masks_pred))
            # calculate dice score
            dice = calculate_dice_loss(masks_pred, true_masks)
            dice_value_list.append(dice)
            # calculate entropy
            entropy = calculate_entropy(masks_pred_raw.cpu().numpy())
            entropy_value_list.append(entropy)
            progress_bar.set_postfix(DICE=dice, ENTROPY=entropy)
            progress_bar.update(1)
    dice_value_list = np.array(dice_value_list)
    dice_mean = np.mean(dice_value_list)
    dice_std = np.std(dice_value_list)
    entropy_value_list = np.array(entropy_value_list)
    entropy_mean = np.mean(entropy_value_list)
    entropy_std = np.std(entropy_value_list)
    return dice_mean, dice_std, entropy_mean, entropy_std

if __name__ == "__main__":
    # make two random 0-1 masks
    mask = np.random.randint(0, 2, (1, 320, 320))
    im = np.random.randint(0, 2, (1, 320, 320))
    dice = calculate_dice(im, mask)
    print("dice score", dice)
    dice_same = calculate_dice(im, im)
    print("dice score for same image", dice_same)
    dice_diff = calculate_dice(im, 1 - im)
    print("dice score for different image", dice_diff)
