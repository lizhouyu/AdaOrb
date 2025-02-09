import sys
import os
import cv2
import json
import argparse
import numpy as np
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
import json
from eval import eval_net
from unet import UNet
from utils.dataset import UNetDataset

def infer(model: UNet, input: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        output = model(input)
    return output

def load_model(model_path: str) -> UNet:
    model = UNet(n_channels=3, n_classes=1, f_channels='model_channels.txt')
    model.load_state_dict(torch.load(model_path))
    return model

def eval_location_A_model_on_location_B(output_dir: str, location_A: str, location_B: str):
    # output_dir = 'eval_results'
    os.makedirs(output_dir, exist_ok=True)

    # location_A = '0357E-1223N'
    # location_B = '0331E-1257N'

    # location_A = '0331E-1257N'
    # location_B = '0357E-1223N'

    # location_A = '0331E-1257N'
    # location_B = '0331E-1257N'

    # location_A = 'mask64_AOI_3_Paris_Train'
    # location_B = 'mask64_AOI_5_Khartoum_Train'

    # location_A = 'mask64_AOI_5_Khartoum_Train'
    # location_B = 'mask64_AOI_3_Paris_Train'

    val_im_folder_path = f'/mnt/sda/nfs/rw/oec/dataset/spacenet/building2/{location_B}/val/images'
    val_mask_folder_path = f'/mnt/sda/nfs/rw/oec/dataset/spacenet/building2/{location_B}/val/masks'

    model_path = f'checkpoints/original/{location_A}/best.pt'

    val_dataset = UNetDataset(im_folder_path=val_im_folder_path, mask_folder_path=val_mask_folder_path, format='image')
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    model = load_model(model_path)
    if torch.cuda.is_available():
        model.cuda()
    
    val_dice = eval_net(model, val_dataloader, True)

    print("Validation Dice Coefficient: {}".format(val_dice))

    # Save the validation results
    results = {
        'location_A': location_A,
        'location_B': location_B,
        'val_dice': val_dice
    }
    with open(os.path.join(output_dir, f'results_{location_A}_on_{location_B}.json'), 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    location_list = ['mask64_AOI_2_Vegas_Train', 'mask64_AOI_3_Paris_Train', 'mask64_AOI_4_Shanghai_Train', 'mask64_AOI_5_Khartoum_Train', 'mask64_AOI_0_global_sampled_Train']
    output_dir = 'eval_results'
    for location_A in location_list:
        for location_B in location_list:
            if 'global' in location_B or os.path.exists(os.path.join(output_dir, f'results_{location_A}_on_{location_B}.json')):
                continue
            eval_location_A_model_on_location_B(output_dir, location_A, location_B)