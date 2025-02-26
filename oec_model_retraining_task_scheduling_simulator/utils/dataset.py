import os
import json
import torch
import logging
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

'''
This dataset is for training UNet.
Parameters:
    - im_folder_path (str): the path to the image folder
    - mask_folder_path (str): the path to the mask folder
    - im_size (list): the size of the image, default is [320, 320]
    - format (str): the format of the image, chosen from image or numpy, default is 'image'
'''
class UNetDataset(Dataset):
    def __init__(self, im_folder_path: str, mask_folder_path: str, im_filename_list: list = None, im_size: list = [320, 320], format: str = 'image'):
        super(UNetDataset, self).__init__()
        self.im_folder_path = im_folder_path
        self.mask_folder_path = mask_folder_path
        # if the image name list is not provided, get all the image names from the image folder
        if im_filename_list is None:
            self.im_filename_list = os.listdir(im_folder_path)
        else:
            self.im_filename_list = im_filename_list
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(im_size)
        ])
        self.format = format

    def __getitem__(self, index):
        # get image name
        im_name = self.im_filename_list[index]
        if self.format == 'image':
            # return the data and label
            im = Image.open(os.path.join(self.im_folder_path, im_name))
            mask = Image.open(os.path.join(self.mask_folder_path, im_name))
            # convert to numpy array
            im = np.array(im)
            mask = np.array(mask)
        elif self.format == 'numpy':
            # return the data and label
            im = np.load(os.path.join(self.im_folder_path, im_name))
            mask = np.load(os.path.join(self.mask_folder_path, im_name))
        else:
            raise ValueError('The format should be chosen from image or numpy.')
        # apply transform
        # get max, min, and mean value of the image
        # print("numpy image", np.max(im), np.min(im), np.mean(im))
        im = self.trans(im)
        # print("torch image", torch.max(im), torch.min(im), torch.mean(im))
        # print("numpy mask", np.max(mask), np.min(mask), np.unique(mask), np.mean(mask))
        # mask = torch.Tensor(mask)
        mask = mask > 0.5
        mask = self.trans(mask)
        mask = mask * 1.0
        # convert mask datatype to float32
        mask = mask.float()
        
        # print("torch mask", torch.max(mask), torch.min(mask), torch.unique(mask), torch.mean(mask))
        # convert data to float32
        im = im.float()
        mask = mask.float()
        # unsqueeze the mask
        # mask = mask.unsqueeze(0)
        # return the data and label
        return im, mask, im_name
        
    def __len__(self):
        return len(self.im_filename_list)

'''
This dataset is for testing UNet inference
Parameters:
    - im_folder_path (str): the path to the image folder
    - im_size (list): the size of the image, default is [320, 320]
'''
class UNetInferenceDataset(Dataset):
    def __init__(self, im_folder_path: str, im_size: list = [320, 320]):
        super(UNetInferenceDataset, self).__init__()
        self.im_folder_path = im_folder_path
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(im_size)
        ])
        self.im_filename_list = os.listdir(im_folder_path)

    def __getitem__(self, index):
        # get image name
        im_name = self.im_filename_list[index]
        im_path = os.path.join(self.im_folder_path, im_name)
        im = Image.open(im_path)
        im = np.array(im)
        # apply transform
        im = self.trans(im)
        # convert data to float32
        im = im.float()
        # return the data and label
        return im, im_name
        
    def __len__(self):
        return len(self.im_filename_list)

if __name__ == '__main__':
    torch.manual_seed(0)
    im_folder_path = 'files/tiler/0/T18SUD_20230505T154811'
    infer_dataset = UNetInferenceDataset(im_folder_path=im_folder_path)
    dataloader = torch.utils.data.DataLoader(infer_dataset, batch_size=4, shuffle=True)
    print(len(infer_dataset))
    test_im, test_im_name = next(iter(dataloader))
    print(test_im.shape)
    print(test_im_name)
    # mask_folder_path = '/mnt/data/zli85/github/OECModelRetraining/pilot_study/data_drift_existence/cloud_detection/unet/train_tao_unet_cloud/tao_experiments/data/320/train/masks'
    # dataset = UNetDataset(im_folder_path=im_folder_path, mask_folder_path=mask_folder_path, format='image')
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    # print(len(dataset))
    # test_im, test_mask = next(iter(dataloader))
    # print(test_im.shape, test_mask.shape)
    # # get the max and min value of the image
    # print(torch.max(test_im), torch.min(test_im))
    # # get the max and min value of the mask
    # print(torch.max(test_mask), torch.min(test_mask), torch.unique(test_mask), torch.mean(test_mask))