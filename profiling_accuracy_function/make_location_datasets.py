import os
import re
import json
import yaml
import shutil
import numpy as np
from tqdm import tqdm

# location_list = ['0331E-1257N', '0357E-1223N']

# train_val_ratio = 0.8
random_seed=0

dataset_root_dir = '/mnt/sda/nfs/rw/oec/dataset/spacenet/urban_dev'
output_root_dir = 'location_datasets'

location_month_image_path_dict_filepath = '/mnt/sda/nfs/rw/oec/dataset/spacenet/urban_dev/location_month_image_path_dict_64_mask.json'

def make_location_dataset(location):
    location_output_dir = os.path.join(output_root_dir, location)
    np.random.seed(random_seed)

    with open(location_month_image_path_dict_filepath, 'r') as f:
        location_month_image_path_dict = json.load(f)

    if location not in location_month_image_path_dict:
        raise ValueError('location not in location_month_image_path_dict')

    month_image_path_dict = location_month_image_path_dict[location]

    num_train_images = 0
    num_val_images = 0

    for month, image_path_info_dict_list in tqdm(month_image_path_dict.items()):
        for image_path_info_dict in tqdm(image_path_info_dict_list):
            image_path = os.path.join(dataset_root_dir, image_path_info_dict['image_path'])
            label_path = os.path.join(dataset_root_dir, image_path_info_dict['mask_path'])

            output_image_dir = os.path.join(location_output_dir, 'images')
            output_label_dir = os.path.join(location_output_dir, 'masks')

            os.makedirs(output_image_dir, exist_ok=True)
            os.makedirs(output_label_dir, exist_ok=True)

            # copy the image and label to the output directory
            shutil.copy(image_path, output_image_dir)
            shutil.copy(label_path, output_label_dir)


    print(f'location: {location}, num_train_images: {num_train_images}, num_val_images: {num_val_images}')


if __name__ == '__main__':

    with open(location_month_image_path_dict_filepath, 'r') as f:
        location_month_image_path_dict = json.load(f)
    location_list = list(location_month_image_path_dict.keys())
    for location in location_list:
        make_location_dataset(location)

