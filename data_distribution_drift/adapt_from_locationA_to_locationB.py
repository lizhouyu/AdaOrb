import os
import cv2
import sys
import json
import torch
import shutil
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from ultralytics import YOLO

from utils.dataset import UNetDataset
from utils.eval import eval_UNet
from unet import UNet
from retrain_models import retrain_cloud_detection_model

if __name__ == "__main__":
    # location_A = '0331E-1257N'
    # location_B = '0357E-1223N'
    # location_A = 'dummy_weights'
    location_A = 'building1'
    location_B = 'mask64_AOI_2_Vegas_Train'
    dataset_root_dir = '/mnt/sda/nfs/rw/oec/dataset/spacenet/building2'
    model_input_size = (64, 64)
    training_image_pool_folder = os.path.join(dataset_root_dir, location_B, 'train', 'images')
    traning_label_folder = os.path.join(dataset_root_dir, location_B, 'train', 'masks')
    training_cache_folder = os.path.join('training_cache', 'building', f'adapt_from_{location_A}_to_{location_B}')
    random_seed = 0
    np.random.seed(random_seed)

    val_image_folder = os.path.join(dataset_root_dir, location_B, 'val', 'images')
    val_label_folder = os.path.join(dataset_root_dir, location_B, 'val', 'masks')

    pool_val_dataset = UNetDataset(val_image_folder, val_label_folder, im_size=model_input_size)

    is_pretrain = True # if True, use the pretrain model, otherwise use the original model
    is_continuous_training = True # if True, retrain the model from the previous round, otherwise retrain the model from the initial model
    is_replay = True # if True, replay the previous round training data to the current round

    if is_pretrain:
        model_weight_path = os.path.join('Prune_U-Net', 'checkpoints', 'original', f'{location_A}', 'best.pt') #f"building/v8n_{location_A}/weights/best.pt"
        # model_weight_path = os.path.join('Prune_U-Net', 'checkpoints', 'pruned', 'building1', 'best.pt') #f"building/v8n_{location_A}/weights/best.pt"
        training_cache_folder = os.path.join(training_cache_folder, 'pretrain')
    else:
        model_weight_path = ''
        training_cache_folder = os.path.join(training_cache_folder, 'no_pretrain')

    model_channel_filepath = os.path.join('Prune_U-Net', 'checkpoints', 'original', f'{location_A}', 'best_channels.txt') #f"building/v8n_{location_A}/weights/best_channels.txt"
    # model_channel_filepath = os.path.join('Prune_U-Net', 'checkpoints', 'pruned', 'building1', 'best_channels.txt') #f"building/v8n_{location_A}/weights/best_channels.txt"

    if is_continuous_training:
        training_cache_folder = os.path.join(training_cache_folder, 'continuous')
    else:
        training_cache_folder = os.path.join(training_cache_folder, 'retrain_from_initial')

    if is_replay:
        training_cache_folder = os.path.join(training_cache_folder, 'replay')
    else:
        training_cache_folder = os.path.join(training_cache_folder, 'no_replay')

    algorithm_training_cache_folder = os.path.join(training_cache_folder, 'unet')

    number_of_round = 50
    samples_per_round = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(training_cache_folder, exist_ok=True) # create the training cache folder

    # initialize the result saving dictionary
    result_dict = {} # {sampling_algorithm: {round_idx: {'dice_mean': dice_mean, 'dice_std': dice_std, 'entropy_mean': entropy_mean, 'entropy_std': entropy_std}}}
    # initialize the image profile dictionary
    image_profile_dict = {} # {image_filename: {'typicality': typicality, 'entropy': entropy}}

    # initialize the result saving dictionary
    result_dict = {}

    # create a round 0 folder
    round_0_cache_folder = os.path.join(algorithm_training_cache_folder, '0')
    os.makedirs(round_0_cache_folder, exist_ok=True)
    # create a checkpoint folder for round 0
    round_0_checkpoint_folder = os.path.join(round_0_cache_folder, 'checkpoints')
    os.makedirs(round_0_checkpoint_folder, exist_ok=True)
    # copy the model weight file to the training cache folder
    if is_pretrain:
        shutil.copyfile(model_weight_path, os.path.join(round_0_checkpoint_folder, 'best.pt'))
    shutil.copyfile(model_channel_filepath, os.path.join(round_0_checkpoint_folder, 'best_channels.txt'))

    # verify the model on the overall image set (image_pool_folder)
    round_0_model_weight_path = os.path.join(round_0_checkpoint_folder, 'best.pt')

    # load the model
    model = UNet(n_channels=3, n_classes=1, f_channels=model_channel_filepath)
    if is_pretrain:
        model.load_state_dict(torch.load(model_weight_path))
    model.to(device)

    # Evaluate the model on the overall image set (image_pool_folder)
    round_dice_mean, round_dice_std, round_entropy_mean, round_entropy_std = eval_UNet(model, pool_val_dataset, device)

    eval_results = {
        "dice_mean": float(round_dice_mean),
        "dice_std": float(round_dice_std),
        "entropy_mean": float(round_entropy_mean),
        "entropy_std": float(round_entropy_std),
    }

    result_dict['0'] = eval_results

    # get the list of image paths
    image_pool_filename_list = os.listdir(training_image_pool_folder)
    image_pool_filepath_list = [os.path.join(training_image_pool_folder, image_pool_filename) for image_pool_filename in image_pool_filename_list] 

    for round_idx in range(1, number_of_round + 1):
        # initialize the result saving dictionary
        result_dict[round_idx] = {
            'dice_mean': None,
            'dice_std': None,
            'entropy_mean': None,
            'entropy_std': None
        }

        # get image profile with the latest model
        prev_round_idx = round_idx - 1

        # randomly sample images according to the profile
        sampled_image_path_list = np.random.choice(image_pool_filepath_list, samples_per_round, replace=False)
        # remove the sampled images from the image pool
        image_pool_filepath_list = [image_pool_filepath for image_pool_filepath in image_pool_filepath_list if image_pool_filepath not in sampled_image_path_list]

        # make the retraining dataset
        ## get the training cache folder
        round_training_cache_folder = os.path.join(algorithm_training_cache_folder, str(round_idx))
        if not os.path.exists(round_training_cache_folder):
            os.makedirs(round_training_cache_folder)
        ## make a retraining dataset folder with images and labels
        retraining_dataset_folder = os.path.join(round_training_cache_folder, 'retraining_dataset')
        os.makedirs(retraining_dataset_folder, exist_ok=True)
        retraining_train_dataset_folder = os.path.join(retraining_dataset_folder, 'train')
        os.makedirs(retraining_train_dataset_folder, exist_ok=True)
        retraining_val_dataset_folder = os.path.join(retraining_dataset_folder, 'val')
        os.makedirs(retraining_val_dataset_folder, exist_ok=True)
        retrain_train_image_folder = os.path.join(retraining_train_dataset_folder, 'images')
        retrain_train_label_folder = os.path.join(retraining_train_dataset_folder, 'labels')
        retrain_val_image_folder = os.path.join(retraining_val_dataset_folder, 'images')
        retrain_val_label_folder = os.path.join(retraining_val_dataset_folder, 'labels')
        os.makedirs(retrain_train_image_folder, exist_ok=True)
        os.makedirs(retrain_train_label_folder, exist_ok=True)
        os.makedirs(retrain_val_image_folder, exist_ok=True)
        os.makedirs(retrain_val_label_folder, exist_ok=True)
        ## copy the sampled images to the retraining dataset folder with 8:2 split for training and validation
        is_first_image = True
        for sampled_image_path in sampled_image_path_list:
            # decide the image for training or validation

            if np.random.rand() < 0.8 and not is_first_image:
                output_image_folder = retrain_train_image_folder
                output_label_folder = retrain_train_label_folder
            else:
                output_image_folder = retrain_val_image_folder
                output_label_folder = retrain_val_label_folder  
                is_first_image = False              

            sampled_image_filename = os.path.basename(sampled_image_path)
            sampled_image_cache_path = os.path.join(output_image_folder, sampled_image_filename)
            shutil.copyfile(sampled_image_path, sampled_image_cache_path)
            ### copy the label
            sample_label_filename = sampled_image_filename
            sampled_label_path = os.path.join(traning_label_folder, sample_label_filename)
            sampled_label_cache_path = os.path.join(output_label_folder, sample_label_filename)
            shutil.copyfile(sampled_label_path, sampled_label_cache_path)
       ## copy the last round retraining dataset to the current round
        if is_replay and prev_round_idx > 0:
            prev_round_cache_folder = os.path.join(algorithm_training_cache_folder, str(prev_round_idx))
            prev_retraining_dataset_folder = os.path.join(prev_round_cache_folder, 'retraining_dataset')
            prev_retraining_train_dataset_folder = os.path.join(prev_retraining_dataset_folder, 'train')
            prev_retraining_val_dataset_folder = os.path.join(prev_retraining_dataset_folder, 'val')
            prev_round_retraining_train_image_folder = os.path.join(prev_retraining_train_dataset_folder, 'images')
            prev_round_retraining_train_label_folder = os.path.join(prev_retraining_train_dataset_folder, 'labels')
            prev_round_retraining_val_image_folder = os.path.join(prev_retraining_val_dataset_folder, 'images')
            prev_round_retraining_val_label_folder = os.path.join(prev_retraining_val_dataset_folder, 'labels')
            for prev_round_retraining_train_image_filename in os.listdir(prev_round_retraining_train_image_folder):
                prev_round_retraining_image_path = os.path.join(prev_round_retraining_train_image_folder, prev_round_retraining_train_image_filename)
                prev_round_retraining_label_path = os.path.join(prev_round_retraining_train_label_folder, prev_round_retraining_train_image_filename)
                shutil.copyfile(prev_round_retraining_image_path, os.path.join(retrain_train_image_folder, prev_round_retraining_train_image_filename))
                shutil.copyfile(prev_round_retraining_label_path, os.path.join(retrain_train_label_folder, prev_round_retraining_train_image_filename))
            for prev_round_retraining_val_image_filename in os.listdir(prev_round_retraining_val_image_folder):
                prev_round_retraining_image_path = os.path.join(prev_round_retraining_val_image_folder, prev_round_retraining_val_image_filename)
                prev_round_retraining_label_path = os.path.join(prev_round_retraining_val_label_folder, prev_round_retraining_val_image_filename)
                shutil.copyfile(prev_round_retraining_image_path, os.path.join(retrain_val_image_folder, prev_round_retraining_val_image_filename))
                shutil.copyfile(prev_round_retraining_label_path, os.path.join(retrain_val_label_folder, prev_round_retraining_val_image_filename))
                
        # use the retraining dataset to retrain the model
        ## get the model checkpoint folder from the previous round
        prev_round_cache_folder = os.path.join(algorithm_training_cache_folder, str(prev_round_idx))
        prev_model_checkpoint_folder = os.path.join(prev_round_cache_folder, 'checkpoints')
        prev_model_weight_path = os.path.join(prev_model_checkpoint_folder, 'best.pt')
        ## create a new round folder
        round_checkpoint_folder = os.path.join(round_training_cache_folder, 'checkpoints')
        os.makedirs(round_checkpoint_folder, exist_ok=True)
        ## retrain the model
        print(f'Retraining the model for round {round_idx}')
        if is_continuous_training:
            retrain_cloud_detection_model(prev_model_weight_path, model_input_size, retraining_train_dataset_folder, retraining_val_dataset_folder, round_checkpoint_folder)
        else:
            retrain_cloud_detection_model(model_weight_path, model_input_size, retraining_train_dataset_folder, retraining_val_dataset_folder, round_checkpoint_folder)
        
        # get model weight path
        round_model_weight_path = os.path.join(round_checkpoint_folder, 'best.pt')
        model_channel_file_path = os.path.join(round_checkpoint_folder, 'best_channels.txt')
        # load the model
        model = UNet(n_channels=3, n_classes=1, f_channels=model_channel_file_path)
        model.load_state_dict(torch.load(round_model_weight_path))
        model.to(device)

        # # evaluate the model on the overall image set (image_pool_folder)
        round_dice_mean, round_dice_std, round_entropy_mean, round_entropy_std = eval_UNet(model, pool_val_dataset, device)

        eval_results = {
            "dice_mean": float(round_dice_mean),
            "dice_std": float(round_dice_std),
            "entropy_mean": float(round_entropy_mean),
            "entropy_std": float(round_entropy_std),
        }

        result_dict[round_idx] = eval_results


        # save the result to file
        result_filepath = os.path.join(algorithm_training_cache_folder, 'result.json')
        with open(result_filepath, 'w') as f:
            json.dump(result_dict, f, indent=4)

    print('Training completed')




