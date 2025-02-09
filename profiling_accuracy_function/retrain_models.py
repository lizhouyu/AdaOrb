import os
import cv2
import json
import numpy as np
import yaml
import shutil
from tqdm import tqdm

from ultralytics import YOLO

from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

from utils.dataset import UNetDataset
from utils.eval import eval_UNet
from unet import UNet
# from utils.dataset import UNetDataset
# from utils.eval import eval_UNet
# from unet import UNet

def retrain_cloud_detection_model(
    model_weights_path: str, 
    model_input_size: tuple,
    retraining_dataset_path: str,
    validation_dataset_path: str,
    checkpoint_folder: str,
    ) -> tuple:
    # preflight check part 1 for data and model availability
    # 1). check if the retraining dataset exists and has data
    if not os.path.exists(retraining_dataset_path):
        print("No data for retraining cloud detection model in the given retraining dataset path.")
        return False, f"No data for retraining cloud detection model in the given retraining dataset path: {retraining_dataset_path}."
    # 2). check if the validation dataset exists
    if not os.path.exists(validation_dataset_path):
        print("No data for validation cloud detection model in the given validation dataset path.")
        return False, f"No data for validation cloud detection model in the given validation dataset path: {validation_dataset_path}."
    # 3). check if the model weights path exists
    if not os.path.exists(model_weights_path):
        print("Model weights path does not exist.")
        return False, f"Model weights path does not exist in the given path: {model_weights_path}." 

    # set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build the training set
    retraining_dataset_image_folder_path = os.path.join(retraining_dataset_path, 'images')
    retraining_dataset_mask_folder_path = os.path.join(retraining_dataset_path, 'labels')
    # preflight check part 2: check if the retraining dataset has data and each image has a corresponding mask
    # 1). check if both images and labels folder exist
    if not os.path.exists(retraining_dataset_image_folder_path):
        print("No images folder in the retraining dataset.")
        return False, f"No images folder in the retraining dataset: {retraining_dataset_image_folder_path}."
    if not os.path.exists(retraining_dataset_mask_folder_path):
        print("No labels folder in the retraining dataset.")
        return False, f"No labels folder in the retraining: {retraining_dataset_mask_folder_path}."
    # 2). check if each image in the retraining dataset has a corresponding mask
    # move the images without masks to the unlabelled folder
    retraining_image_unlabelled_folder_path = os.path.join(retraining_dataset_path, 'unlabelled')
    os.makedirs(retraining_image_unlabelled_folder_path, exist_ok=True)
    retraining_dataset_image_files = os.listdir(retraining_dataset_image_folder_path)
    retraining_dataset_mask_files = os.listdir(retraining_dataset_mask_folder_path)
    for image_filename in retraining_dataset_image_files:
        mask_filename = image_filename
        if mask_filename not in retraining_dataset_mask_files:
            # move the image file to the unlabelled folder
            shutil.move(os.path.join(retraining_dataset_image_folder_path, image_filename), os.path.join(retraining_image_unlabelled_folder_path, image_filename))
    # double check if there are no labelled images in the retraining dataset
    if len(os.listdir(retraining_dataset_image_folder_path)) == 0:
        print("No labelled images in the retraining dataset.")
        return False, f"No labelled images in the retraining dataset: {retraining_dataset_image_folder_path}."
    # 3). check if the image and labels folder has data
    if len(os.listdir(retraining_dataset_image_folder_path)) == 0:
        print("No data in the image folder of the retraining dataset.")
        return False, "No data in the image folder of the retraining dataset."
    if len(os.listdir(retraining_dataset_mask_folder_path)) == 0:
        print("No data in the mask folder of the retraining dataset.")
        return False, "No data in the mask folder of the retraining dataset."
    # 4). make the checkpoint folder if it does not exist
    os.makedirs(checkpoint_folder, exist_ok=True)

    # build the training set
    retraining_dataset = UNetDataset(retraining_dataset_image_folder_path, retraining_dataset_mask_folder_path, im_size=model_input_size)
    # build the validation set
    validation_dataset_image_folder_path = os.path.join(validation_dataset_path, 'images')
    validation_dataset_mask_folder_path = os.path.join(validation_dataset_path, 'labels')
    # preflight check part 3: check if the validation dataset has data and each image has a corresponding mask
    # 1). check if both images and masks folder exist
    if not os.path.exists(validation_dataset_image_folder_path):
        print("No images folder in the validation dataset.")
        return False, f"No images folder in the validation dataset: {validation_dataset_image_folder_path}."
    if not os.path.exists(validation_dataset_mask_folder_path):
        print("No masks folder in the validation dataset.")
        return False, f"No masks folder in the validation dataset: {validation_dataset_mask_folder_path}."
    # 2). check if each image has a corresponding mask  
    # move the images without masks to the unlabelled folder
    validation_image_unlabelled_folder_path = os.path.join(validation_dataset_path, 'unlabelled')
    os.makedirs(validation_image_unlabelled_folder_path, exist_ok=True)
    validation_dataset_image_files = os.listdir(validation_dataset_image_folder_path)
    validation_dataset_mask_files = os.listdir(validation_dataset_mask_folder_path)
    for image_filename in validation_dataset_image_files:
        mask_filename = image_filename
        if mask_filename not in validation_dataset_mask_files:
            # move the image file to the unlabelled folder
            shutil.move(os.path.join(validation_dataset_image_folder_path, image_filename), os.path.join(validation_image_unlabelled_folder_path, image_filename))
    # double check if there are no labelled images in the validation dataset
    if len(os.listdir(validation_dataset_image_folder_path)) == 0:
        print("No labelled images in the validation dataset.")
        return False, f"No labelled images in the validation dataset: {validation_dataset_image_folder_path}."
    # 3). check if the image and masks folder has data
    if len(os.listdir(validation_dataset_image_folder_path)) == 0:
        print("No data in the image folder of the validation dataset.")
        return False, f"No data in the image folder of the validation dataset: {validation_dataset_image_folder_path}."
    if len(os.listdir(validation_dataset_mask_folder_path)) == 0:
        print("No data in the mask folder of the validation dataset.")
        return False, f"No data in the mask folder of the validation dataset: {validation_dataset_mask_folder_path}."
    # build the validation set
    validation_dataset = UNetDataset(validation_dataset_image_folder_path, validation_dataset_mask_folder_path, im_size=model_input_size)

    # get the data loader
    batch_size = min(1, len(retraining_dataset))
    train_dataloader = DataLoader(retraining_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # load the model and set it to training mode
    model_channel_file_path = model_weights_path.replace('.pt', '_channels.txt')
    model = UNet(n_channels=3, n_classes=1, f_channels=model_channel_file_path)
    model.load_state_dict(torch.load(model_weights_path))
    model = model.to(device)
    model.train()

    # set the optimizer
    # optimizer = optim.Adam(model.parameters(), lr=1e-5, eps=9.99999993923e-09, betas=(0.899999976158, 0.999000012875))
    optimizer = optim.Adam(model.parameters(), lr=1e-5, eps=9.99999993923e-09, betas=(0.899999976158, 0.999000012875))

    # set the loss function
    criterion = nn.BCELoss()
    l2_reg_func = nn.MSELoss()

    # number of fine-tuning epochs
    num_epochs = 20
    min_num_epochs = 5
    coverge_threshold = 0.01

    dice_max = 0
    best_model_state_dict = model.state_dict()
    last_epoch_loss = float('inf')
    for epoch in range(num_epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, num_epochs))
        epoch_loss = 0
        pbar = tqdm(train_dataloader)
        for batch_idx, (imgs, true_masks, im_name) in enumerate(pbar):
            imgs = imgs.to(device)
            true_masks = true_masks.to(device)

            masks_pred = model(imgs)
            # print(f'Images shape: {imgs.shape}, masks shape: {true_masks.shape}', f'Predicted masks shape: {masks_pred.shape}')
            # print(f"Image max: {torch.max(imgs)}, Image min: {torch.min(imgs)}, Image mean: {torch.mean(imgs)}")

            # # draw the image and mask side by side
            # fig, ax = plt.subplots(1, 3)
            # ax[0].imshow(imgs[0].permute(1, 2, 0).cpu().numpy())
            # ax[1].imshow(true_masks[0][0].cpu().numpy())
            # ax[2].imshow(masks_pred[0][0].cpu().detach().numpy())
            # plt.savefig(f'./cloud_detection_retrain_test_debug_{im_name[0]}_image_mask.png')
            # plt.close()

            masks_probs_flat = masks_pred.view(-1)

            true_masks_flat = true_masks.view(-1)

            optimizer.zero_grad()

            cse_loss = criterion(masks_probs_flat, true_masks_flat)
            l2_reg = l2_reg_func(masks_probs_flat, true_masks_flat)
            loss = cse_loss + l2_reg * 2e-5
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item() / true_masks.size(0))
            pbar.update(1)

            batch_loss = loss.item() / true_masks.size(0)

        epoch_dice_mean, epoch_dice_std, epoch_entropy_mean, epoch_entropy_std = eval_UNet(model, validation_dataset, device)
        print(f'Epoch {epoch + 1}/{num_epochs} loss: {epoch_loss / len(train_dataloader)} dice: {epoch_dice_mean} entropy: {epoch_entropy_mean}')
        model.train()
    
        if epoch_dice_mean > dice_max:
            print("New best model found.")
            dice_max = epoch_dice_mean
            best_model_state_dict = model.state_dict()
        # best_model_state_dict = model.state_dict()
        epoch_loss /= len(train_dataloader)
        if last_epoch_loss - epoch_loss < coverge_threshold:
            print("Converged.")
            if min_num_epochs < epoch:
                break
            else:
                print("Keep training due to minimum epoch requirement.")
        else:
            print(f"Last epoch loss: {last_epoch_loss}, Current epoch loss: {epoch_loss}, Difference: {last_epoch_loss - epoch_loss}, keeping training.")
        last_epoch_loss = epoch_loss

    # save the best model
    checkpoint_path = os.path.join(checkpoint_folder, 'best.pt')
    torch.save(best_model_state_dict, checkpoint_path)

    # zli85: add for retraining experiment: save the channel file
    shutil.copyfile(model_channel_file_path, os.path.join(checkpoint_folder, 'best_channels.txt'))
    # print("model channel file: ", model_channel_file_path)
    # print("saved path: ", os.path.join(checkpoint_folder, 'best_channels.txt'))

    return True, "Model retrained successfully."

def retrain_crop_monitoring_model(
    model_weights_path: str, 
    model_input_size: tuple,
    retraining_dataset_path: str,
    validation_dataset_path: str,
    checkpoint_folder: str,
    ) -> tuple:
    try:
        # preflight check part 1 for data and model availability
        print("Checking data and model availability...")
        # 1). check if the retraining dataset exists and has data
        if not os.path.exists(retraining_dataset_path):
            print("No data for retraining crop monitoring model in the given retraining dataset path.")
            return False, f"No data for retraining crop monitoring model in the given retraining dataset path: {retraining_dataset_path}."
        # 2). check if the validation dataset exists
        if not os.path.exists(validation_dataset_path):
            print("No data for validation crop monitoring model in the given validation dataset path.")
            return False, f"No data for validation crop monitoring model in the given validation dataset path: {validation_dataset_path}."
        # 3). check if the model weights path exists
        if not os.path.exists(model_weights_path):
            print("Model weights path does not exist.")
            return False, f"Model weights path does not exist in the given path: {model_weights_path}."
        # 4.1) check if the retraining dataset has data 
        retraining_dataset_image_folder_path = os.path.join(retraining_dataset_path, 'images')
        retraining_dataset_labels_folder_path = os.path.join(retraining_dataset_path, 'labels')
        if not os.path.exists(retraining_dataset_image_folder_path) or len(os.listdir(retraining_dataset_image_folder_path)) == 0:
            print("No images in the retraining dataset.")
            return False, f"No images in the retraining dataset: {retraining_dataset_image_folder_path}."
        if not os.path.exists(retraining_dataset_labels_folder_path) or len(os.listdir(retraining_dataset_labels_folder_path)) == 0:
            print("No labels in the retraining dataset.")
            return False, f"No labels in the retraining dataset: {retraining_dataset_labels_folder_path}."
        # 4.2) check if the validation dataset has data
        validation_dataset_image_folder_path = os.path.join(validation_dataset_path, 'images')
        validation_dataset_labels_folder_path = os.path.join(validation_dataset_path, 'labels')
        if not os.path.exists(validation_dataset_image_folder_path) or len(os.listdir(validation_dataset_image_folder_path)) == 0:
            print("No images in the validation dataset.")
            return False, f"No images in the validation dataset: {validation_dataset_image_folder_path}."
        if not os.path.exists(validation_dataset_labels_folder_path) or len(os.listdir(validation_dataset_labels_folder_path)) == 0:
            print("No labels in the validation dataset.")
            return False, f"No labels in the validation dataset: {validation_dataset_labels_folder_path}."
        # 5). deal with the case where there are images but no labels
        # move the images without labels to the unlabelled folder
        # 5.1) check for retraining images without labels
        retraining_image_unlabelled_folder_path = os.path.join(retraining_dataset_path, 'unlabelled')
        os.makedirs(retraining_image_unlabelled_folder_path, exist_ok=True)
        retraining_image_filename_list = os.listdir(retraining_dataset_image_folder_path)
        retraining_label_filename_list = os.listdir(retraining_dataset_labels_folder_path)
        for image_filename in retraining_image_filename_list:
            if '.jpg' in image_filename:
                label_filename = image_filename.replace(".jpg", ".txt")
            elif '.png' in image_filename:
                label_filename = image_filename.replace(".png", ".txt")
            else:
                print(f"Invalid image file format: {image_filename}")
                continue
            if label_filename not in retraining_label_filename_list:
                # move the image file to the unlabelled folder
                shutil.move(os.path.join(retraining_dataset_image_folder_path, image_filename), os.path.join(retraining_image_unlabelled_folder_path, image_filename))
        # double check if there are no labelled images in the retraining dataset
        if len(os.listdir(retraining_dataset_image_folder_path)) == 0:
            print("No labelled images in the retraining dataset.")
            return False, f"No labelled images in the retraining dataset: {retraining_dataset_image_folder_path}."
        # 5.2) check for validation images without labels
        validation_image_unlabelled_folder_path = os.path.join(validation_dataset_path, 'unlabelled')
        os.makedirs(validation_image_unlabelled_folder_path, exist_ok=True)
        validation_image_filename_list = os.listdir(validation_dataset_image_folder_path)
        validation_label_filename_list = os.listdir(validation_dataset_labels_folder_path)
        for image_filename in validation_image_filename_list:
            label_filename = image_filename.replace('.jpg', '.txt').replace('.png', '.txt')
            if label_filename not in validation_label_filename_list:
                # move the image file to the unlabelled folder
                shutil.move(os.path.join(validation_dataset_image_folder_path, image_filename), os.path.join(validation_image_unlabelled_folder_path, image_filename))
        # double check if there are no labelled images in the validation dataset
        if len(os.listdir(validation_dataset_image_folder_path)) == 0:
            print("No labelled images in the validation dataset.")
            return False, f"No labelled images in the validation dataset: {validation_dataset_image_folder_path}."
        # 6) make the checkpoint folder if it does not exist
        os.makedirs(checkpoint_folder, exist_ok=True)

        # fabricate the training yaml config
        print("Fabricating the training yaml config...")
        root_path = os.getcwd()
        train_path = os.path.join(retraining_dataset_path, 'images')
        val_path = os.path.join(validation_dataset_path, 'images')
        # training_config = {
        #     'path': root_path,
        #     'train': train_path,
        #     'val': val_path,
        #     'names': {
        #         0: "Background",
        #         1: "Meadow",
        #         2: "Soft winter wheat",
        #         3: "Corn",
        #         4: "Winter barley",
        #         5: "Winter rapeseed",
        #         6: "Spring barley",
        #         7: "Sunflower",
        #         8: "Grapevine",
        #         9: "Beet",
        #         10: "Winter triticale",
        #         11: "Winter durum wheat",
        #         12: "Fruits,  vegetables, flowers",
        #         13: "Potatoes",
        #         14: "Leguminous fodder",
        #         15: "Soybeans",
        #         16: "Orchard",
        #         17: "Mixed cereal",
        #         18: "Sorghum",
        #         19: "Void label"
        #     }
        # }
        training_config = {
            'path': root_path,
            'train': train_path,
            'val': val_path,
            'names': {
                0: 'water',
            }
        }
        # export the training yaml config to checkpoint folder
        retraining_yaml_path = os.path.join(checkpoint_folder, 'training_config.yaml')
        with open(retraining_yaml_path, 'w') as f:
            yaml.dump(training_config, f, allow_unicode=True, sort_keys=False)
        # retrain the model
        print("Instantiating the model...")
        model = YOLO(model_weights_path)
        print("Retraining the model...")
        model.train(data=retraining_yaml_path, epochs=10, batch=1, imgsz=model_input_size[0], project=checkpoint_folder, name='tmp', device=[0], plots=False)
        # move the best weights
        best_weights_path = os.path.join(checkpoint_folder, 'tmp', 'weights', 'best.pt')
        # best_weights_path = os.path.join(checkpoint_folder, 'tmp', 'weights', 'last.pt') # zli85: change to last.pt to always update the weights
        shutil.move(best_weights_path, os.path.join(checkpoint_folder, 'best.pt'))
        # remove all the tmp* folders
        for folder in os.listdir(checkpoint_folder):
            if folder.startswith('tmp'):
                shutil.rmtree(os.path.join(checkpoint_folder, folder))
        return True, "Model retrained successfully."
    except Exception as e:
        print(f'Error retraining crop monitoring model: {e}')
        return False, f"Error retraining crop monitoring model with message: {e}"

def retrain_object_detection_model(
    model_weights_path: str, 
    model_input_size: tuple,
    retraining_dataset_path: str,
    validation_dataset_path: str,
    checkpoint_folder: str,
    ) -> tuple:
    try:
        # preflight check part 1 for data and model availability
        print("Checking data and model availability...")
        # 1). check if the retraining dataset exists and has data
        if not os.path.exists(retraining_dataset_path):
            print("No data for retraining crop monitoring model in the given retraining dataset path.")
            return False, f"No data for retraining crop monitoring model in the given retraining dataset path: {retraining_dataset_path}."
        # 2). check if the validation dataset exists
        if not os.path.exists(validation_dataset_path):
            print("No data for validation crop monitoring model in the given validation dataset path.")
            return False, f"No data for validation crop monitoring model in the given validation dataset path: {validation_dataset_path}."
        # 3). check if the model weights path exists
        if not os.path.exists(model_weights_path):
            print("Model weights path does not exist.")
            return False, f"Model weights path does not exist in the given path: {model_weights_path}."
        # 4.1) check if the retraining dataset has data 
        retraining_dataset_image_folder_path = os.path.join(retraining_dataset_path, 'images')
        retraining_dataset_labels_folder_path = os.path.join(retraining_dataset_path, 'labels')
        if not os.path.exists(retraining_dataset_image_folder_path) or len(os.listdir(retraining_dataset_image_folder_path)) == 0:
            print("No images in the retraining dataset.")
            return False, f"No images in the retraining dataset: {retraining_dataset_image_folder_path}."
        if not os.path.exists(retraining_dataset_labels_folder_path) or len(os.listdir(retraining_dataset_labels_folder_path)) == 0:
            print("No labels in the retraining dataset.")
            return False, f"No labels in the retraining dataset: {retraining_dataset_labels_folder_path}."
        # 4.2) check if the validation dataset has data
        validation_dataset_image_folder_path = os.path.join(validation_dataset_path, 'images')
        validation_dataset_labels_folder_path = os.path.join(validation_dataset_path, 'labels')
        if not os.path.exists(validation_dataset_image_folder_path) or len(os.listdir(validation_dataset_image_folder_path)) == 0:
            print("No images in the validation dataset.")
            return False, f"No images in the validation dataset: {validation_dataset_image_folder_path}."
        if not os.path.exists(validation_dataset_labels_folder_path) or len(os.listdir(validation_dataset_labels_folder_path)) == 0:
            print("No labels in the validation dataset.")
            return False, f"No labels in the validation dataset: {validation_dataset_labels_folder_path}."
        # 5). deal with the case where there are images but no labels
        # move the images without labels to the unlabelled folder
        # 5.1) check for retraining images without labels
        retraining_image_unlabelled_folder_path = os.path.join(retraining_dataset_path, 'unlabelled')
        os.makedirs(retraining_image_unlabelled_folder_path, exist_ok=True)
        retraining_image_filename_list = os.listdir(retraining_dataset_image_folder_path)
        retraining_label_filename_list = os.listdir(retraining_dataset_labels_folder_path)
        for image_filename in retraining_image_filename_list:
            label_filename = image_filename.replace('.jpg', '.txt').replace('.png', '.txt')
            if label_filename not in retraining_label_filename_list:
                # move the image file to the unlabelled folder
                shutil.move(os.path.join(retraining_dataset_image_folder_path, image_filename), os.path.join(retraining_image_unlabelled_folder_path, image_filename))
        # double check if there are no labelled images in the retraining dataset
        if len(os.listdir(retraining_dataset_image_folder_path)) == 0:
            print("No labelled images in the retraining dataset.")
            return False, f"No labelled images in the retraining dataset: {retraining_dataset_image_folder_path}."
        # 5.2) check for validation images without labels
        validation_image_unlabelled_folder_path = os.path.join(validation_dataset_path, 'unlabelled')
        os.makedirs(validation_image_unlabelled_folder_path, exist_ok=True)
        validation_image_filename_list = os.listdir(validation_dataset_image_folder_path)
        validation_label_filename_list = os.listdir(validation_dataset_labels_folder_path)
        for image_filename in validation_image_filename_list:
            label_filename = image_filename.replace('.jpg', '.txt').replace('.png', '.txt')
            if label_filename not in validation_label_filename_list:
                # move the image file to the unlabelled folder
                shutil.move(os.path.join(validation_dataset_image_folder_path, image_filename), os.path.join(validation_image_unlabelled_folder_path, image_filename))
        # double check if there are no labelled images in the validation dataset
        if len(os.listdir(validation_dataset_image_folder_path)) == 0:
            print("No labelled images in the validation dataset.")
            return False, f"No labelled images in the validation dataset: {validation_dataset_image_folder_path}."
        # 6) check if the validation dataset has data
        validation_dataset_image_folder_path = os.path.join(validation_dataset_path, 'images')
        validation_dataset_labels_folder_path = os.path.join(validation_dataset_path, 'labels')
        if not os.path.exists(validation_dataset_image_folder_path) or len(os.listdir(validation_dataset_image_folder_path)) == 0:
            print("No images in the validation dataset.")
            return False, f"No images in the validation dataset: {validation_dataset_image_folder_path}."
        if not os.path.exists(validation_dataset_labels_folder_path) or len(os.listdir(validation_dataset_labels_folder_path)) == 0:
            print("No labels in the validation dataset.")
            return False, f"No labels in the validation dataset: {validation_dataset_labels_folder_path}."
        # deal with the case where there are images but no labels
        # move the images without labels to the unlabelled folder
        validation_image_unlabelled_folder_path = os.path.join(validation_dataset_path, 'unlabelled')
        os.makedirs(validation_image_unlabelled_folder_path, exist_ok=True)
        validation_image_filename_list = os.listdir(validation_dataset_image_folder_path)
        validation_label_filename_list = os.listdir(validation_dataset_labels_folder_path)
        for image_filename in validation_image_filename_list:
            label_filename = image_filename.replace('.jpg', '.txt').replace('.png', '.txt')
            if label_filename not in validation_label_filename_list:
                # move the image file to the unlabelled folder
                shutil.move(os.path.join(validation_dataset_image_folder_path, image_filename), os.path.join(validation_image_unlabelled_folder_path, image_filename))
        # double check if there are no labelled images in the validation dataset
        if len(os.listdir(validation_dataset_image_folder_path)) == 0:
            print("No labelled images in the validation dataset.")
            return False, f"No labelled images in the validation dataset: {validation_dataset_image_folder_path}."
        # 6) make the checkpoint folder if it does not exist
        os.makedirs(checkpoint_folder, exist_ok=True)

        # check if enough data for retraining
        if os.path.exists(retraining_dataset_path) and len(os.listdir(retraining_dataset_path)) == 0:
            print("No data for retraining object detection model.")
            return False, "No data for retraining object detection model."
        # make checkpoint folder if it does not exist
        os.makedirs(checkpoint_folder, exist_ok=True)
        # fabricate the training yaml config
        print("Fabricating the training yaml config...")
        root_path = os.getcwd()
        train_path = os.path.join(retraining_dataset_path, 'images')
        val_path = os.path.join(validation_dataset_path, 'images')
        # training_config = {
        #     'path': root_path,
        #     'train': train_path,
        #     'val': val_path,
        #     'names': {
        #         0: "Building",
        #     }
        # }
        training_config = {
            'path': root_path,
            'train': train_path,
            'val': val_path,
            'names': {
                0: "Meadow",
                1: "Soft winter wheat",
                2: "Corn",
                3: "Winter barley",
                4: "Winter rapeseed",
                5: "Spring barley",
                6: "Sunflower",
                7: "Grapevine",
                8: "Beet",
                9: "Winter triticale",
                10: "Winter durum wheat",
                11: "Fruits,  vegetables, flowers",
                12: "Potatoes",
                13: "Leguminous fodder",
                14: "Soybeans",
                15: "Orchard",
                16: "Mixed cereal",
                17: "Sorghum"
            }
        }
        # export the training yaml config to checkpoint folder
        retraining_yaml_path = os.path.join(checkpoint_folder, 'training_config.yaml')
        with open(retraining_yaml_path, 'w') as f:
            yaml.dump(training_config, f, allow_unicode=True, sort_keys=False)
        # retrain the model
        print("Instantiating the model...")
        model = YOLO(model_weights_path)
        print("Retraining the model...")
        model.train(data=retraining_yaml_path, epochs=3, batch=1, imgsz=model_input_size, project=checkpoint_folder, name='tmp', device=[1], plots=False)   # move the best weights
        best_weights_path = os.path.join(checkpoint_folder, 'tmp', 'weights', 'best.pt')
        # best_weights_path = os.path.join(checkpoint_folder, 'tmp', 'weights', 'last.pt') # zli85: change to last.pt to always update the weights
        shutil.move(best_weights_path, os.path.join(checkpoint_folder, 'best.pt'))
        # remove all the tmp* folders
        for folder in os.listdir(checkpoint_folder):
            if folder.startswith('tmp'):
                shutil.rmtree(os.path.join(checkpoint_folder, folder))
        return True, "Model retrained successfully."
    except Exception as e:
        print(f'Error retraining object detection model: {e}')
        return False, f"Error retraining object detection model with message: {e}"


def retrain_classification_model(
    model_weights_path: str, 
    model_input_size: tuple,
    retraining_dataset_path: str,
    validation_dataset_path: str,
    checkpoint_folder: str,
    ) -> tuple:
    try:
        # preflight check part 1 for data and model availability
        print("Checking data and model availability...")
        # 1). check if the retraining dataset exists and has data
        if not os.path.exists(retraining_dataset_path):
            print("No data for retraining crop monitoring model in the given retraining dataset path.")
            return False, f"No data for retraining crop monitoring model in the given retraining dataset path: {retraining_dataset_path}."
        # 2). check if the validation dataset exists
        if not os.path.exists(validation_dataset_path):
            print("No data for validation crop monitoring model in the given validation dataset path.")
            return False, f"No data for validation crop monitoring model in the given validation dataset path: {validation_dataset_path}."
        # 3). check if the model weights path exists
        if not os.path.exists(model_weights_path):
            print("Model weights path does not exist.")
            return False, f"Model weights path does not exist in the given path: {model_weights_path}."
        # 4.1) check if the retraining dataset has data 
        retraining_dataset_image_folder_path = os.path.join(retraining_dataset_path, 'images')
        retraining_dataset_labels_folder_path = os.path.join(retraining_dataset_path, 'labels')
        if not os.path.exists(retraining_dataset_image_folder_path) or len(os.listdir(retraining_dataset_image_folder_path)) == 0:
            print("No images in the retraining dataset.")
            return False, f"No images in the retraining dataset: {retraining_dataset_image_folder_path}."
        if not os.path.exists(retraining_dataset_labels_folder_path) or len(os.listdir(retraining_dataset_labels_folder_path)) == 0:
            print("No labels in the retraining dataset.")
            return False, f"No labels in the retraining dataset: {retraining_dataset_labels_folder_path}."
        # 4.2) check if the validation dataset has data
        validation_dataset_image_folder_path = os.path.join(validation_dataset_path, 'images')
        validation_dataset_labels_folder_path = os.path.join(validation_dataset_path, 'labels')
        if not os.path.exists(validation_dataset_image_folder_path) or len(os.listdir(validation_dataset_image_folder_path)) == 0:
            print("No images in the validation dataset.")
            return False, f"No images in the validation dataset: {validation_dataset_image_folder_path}."
        if not os.path.exists(validation_dataset_labels_folder_path) or len(os.listdir(validation_dataset_labels_folder_path)) == 0:
            print("No labels in the validation dataset.")
            return False, f"No labels in the validation dataset: {validation_dataset_labels_folder_path}."
        # 5). deal with the case where there are images but no labels
        # move the images without labels to the unlabelled folder
        # 5.1) check for retraining images without labels
        retraining_image_unlabelled_folder_path = os.path.join(retraining_dataset_path, 'unlabelled')
        os.makedirs(retraining_image_unlabelled_folder_path, exist_ok=True)
        retraining_image_filename_list = os.listdir(retraining_dataset_image_folder_path)
        retraining_label_filename_list = os.listdir(retraining_dataset_labels_folder_path)
        for image_filename in retraining_image_filename_list:
            label_filename = image_filename.replace('.jpg', '.txt').replace('.png', '.txt')
            if label_filename not in retraining_label_filename_list:
                # move the image file to the unlabelled folder
                shutil.move(os.path.join(retraining_dataset_image_folder_path, image_filename), os.path.join(retraining_image_unlabelled_folder_path, image_filename))
        # double check if there are no labelled images in the retraining dataset
        if len(os.listdir(retraining_dataset_image_folder_path)) == 0:
            print("No labelled images in the retraining dataset.")
            return False, f"No labelled images in the retraining dataset: {retraining_dataset_image_folder_path}."
        # 5.2) check for validation images without labels
        validation_image_unlabelled_folder_path = os.path.join(validation_dataset_path, 'unlabelled')
        os.makedirs(validation_image_unlabelled_folder_path, exist_ok=True)
        validation_image_filename_list = os.listdir(validation_dataset_image_folder_path)
        validation_label_filename_list = os.listdir(validation_dataset_labels_folder_path)
        for image_filename in validation_image_filename_list:
            label_filename = image_filename.replace('.jpg', '.txt').replace('.png', '.txt')
            if label_filename not in validation_label_filename_list:
                # move the image file to the unlabelled folder
                shutil.move(os.path.join(validation_dataset_image_folder_path, image_filename), os.path.join(validation_image_unlabelled_folder_path, image_filename))
        # double check if there are no labelled images in the validation dataset
        if len(os.listdir(validation_dataset_image_folder_path)) == 0:
            print("No labelled images in the validation dataset.")
            return False, f"No labelled images in the validation dataset: {validation_dataset_image_folder_path}."
        # 6) check if the validation dataset has data
        validation_dataset_image_folder_path = os.path.join(validation_dataset_path, 'images')
        validation_dataset_labels_folder_path = os.path.join(validation_dataset_path, 'labels')
        if not os.path.exists(validation_dataset_image_folder_path) or len(os.listdir(validation_dataset_image_folder_path)) == 0:
            print("No images in the validation dataset.")
            return False, f"No images in the validation dataset: {validation_dataset_image_folder_path}."
        if not os.path.exists(validation_dataset_labels_folder_path) or len(os.listdir(validation_dataset_labels_folder_path)) == 0:
            print("No labels in the validation dataset.")
            return False, f"No labels in the validation dataset: {validation_dataset_labels_folder_path}."
        # deal with the case where there are images but no labels
        # move the images without labels to the unlabelled folder
        validation_image_unlabelled_folder_path = os.path.join(validation_dataset_path, 'unlabelled')
        os.makedirs(validation_image_unlabelled_folder_path, exist_ok=True)
        validation_image_filename_list = os.listdir(validation_dataset_image_folder_path)
        validation_label_filename_list = os.listdir(validation_dataset_labels_folder_path)
        for image_filename in validation_image_filename_list:
            label_filename = image_filename.replace('.jpg', '.txt').replace('.png', '.txt')
            if label_filename not in validation_label_filename_list:
                # move the image file to the unlabelled folder
                shutil.move(os.path.join(validation_dataset_image_folder_path, image_filename), os.path.join(validation_image_unlabelled_folder_path, image_filename))
        # double check if there are no labelled images in the validation dataset
        if len(os.listdir(validation_dataset_image_folder_path)) == 0:
            print("No labelled images in the validation dataset.")
            return False, f"No labelled images in the validation dataset: {validation_dataset_image_folder_path}."
        # 6) make the checkpoint folder if it does not exist
        os.makedirs(checkpoint_folder, exist_ok=True)

        # check if enough data for retraining
        if os.path.exists(retraining_dataset_path) and len(os.listdir(retraining_dataset_path)) == 0:
            print("No data for retraining object detection model.")
            return False, "No data for retraining object detection model."
        # make checkpoint folder if it does not exist
        os.makedirs(checkpoint_folder, exist_ok=True)
        # define the class_idx_label_dict
        # class_idx_label_dict = {
        #     0: 'AnnualCrop',
        #     1: 'Forest', 
        #     2: 'HerbaceousVegetation',
        #     3: 'Highway',
        #     4: 'Industrial',
        #     5: 'Pasture',
        #     6: 'PermanentCrop',
        #     7: 'Residential',
        #     8: 'River',
        #     9: 'SeaLake'
        # }
        class_idx_label_dict = {
            0: 'HerbaceousVegetation',
            1: 'Other',
            2: 'SeaLake'
        }
        # convert training and validation datasets into yolo format
        # combine them into the same folder with sub folders train and valid
        # tiling images according to the labels and saving them into different folders
        # 1). make a folder in the checkpoint folder for the training and validation datasets
        retrain_dataset_folder = os.path.join(checkpoint_folder, 'dataset')
        os.makedirs(retrain_dataset_folder, exist_ok=True)
        # 2). make a folder for the training and validation datasets
        # retrain_train_folder = os.path.join(retrain_dataset_folder, 'train')
        # retrain_valid_folder = os.path.join(retrain_dataset_folder, 'val')
        # retrain_test_folder = os.path.join(retrain_dataset_folder, 'test')
        # os.makedirs(retrain_train_folder, exist_ok=True)
        # os.makedirs(retrain_valid_folder, exist_ok=True)
        # os.makedirs(retrain_test_folder, exist_ok=True)
        # # 2.2) make a folder for every value in the class_idx_label_dict in each split folder
        
        # for class_idx in class_idx_label_dict.keys():
        #     class_label = class_idx_label_dict[class_idx]
        #     os.makedirs(os.path.join(retrain_train_folder, class_label), existretrain_train_folder_ok=True)
        #     os.makedirs(os.path.join(retrain_valid_folder, class_label), exist_ok=True)
        #     os.makedirs(os.path.join(retrain_test_folder, class_label), exist_ok=True)
        # copy the retrain_classification_backup_data as the retraub dataset
        retrain_classification_backup_data_path = '/home/zli85/oec_pinxiang/experiment/ground_segment/verify_and_retrain/retrain_classification_backup_data_3cls'
        retrain_backup_data_train_folder = os.path.join(retrain_classification_backup_data_path, 'train')
        retrain_backup_data_valid_folder = os.path.join(retrain_classification_backup_data_path, 'valid')
        retrain_backup_data_test_folder = os.path.join(retrain_classification_backup_data_path, 'test')
        retrain_train_folder = os.path.join(retrain_dataset_folder, 'train')
        retrain_valid_folder = os.path.join(retrain_dataset_folder, 'val')
        retrain_test_folder = os.path.join(retrain_dataset_folder, 'test')
        shutil.copytree(retrain_backup_data_train_folder, retrain_train_folder)
        shutil.copytree(retrain_backup_data_valid_folder, retrain_valid_folder)
        shutil.copytree(retrain_backup_data_test_folder, retrain_test_folder)
        # 2.3) copy one image from the retrain_classification_backup_data for each class in each split folder

        # 3). copying the images and saving them into the train and valid folders
        # 3.1) copying the images in the retraining dataset
        retraining_image_folder_path = os.path.join(retraining_dataset_path, 'images')
        retraining_label_folder_path = os.path.join(retraining_dataset_path, 'labels')
        for image_filename in os.listdir(retraining_image_folder_path):
            # get the image and label file paths
            image_file_path = os.path.join(retraining_image_folder_path, image_filename)
            label_file_path = os.path.join(retraining_label_folder_path, image_filename.replace('.jpg', '.txt').replace('.png', '.txt'))
            # read the label files
            with open(label_file_path, 'r') as f:
                patch_label = int(f.read().strip())
            # get the patch fodler
            patch_label_name = class_idx_label_dict[patch_label]
            # make the label folder if it does not exist
            patch_label_folder = os.path.join(retrain_train_folder, patch_label_name)
            os.makedirs(patch_label_folder, exist_ok=True)
            # save the image into the patch label folder
            shutil.copy(image_file_path, patch_label_folder)
        # 3.2) copying the images in the validation dataset
        validation_image_folder_path = os.path.join(validation_dataset_path, 'images')
        validation_label_folder_path = os.path.join(validation_dataset_path, 'labels')
        for image_filename in os.listdir(validation_image_folder_path):
            # get the image and label file paths
            image_file_path = os.path.join(validation_image_folder_path, image_filename)
            label_file_path = os.path.join(validation_label_folder_path, image_filename.replace('.jpg', '.txt').replace('.png', '.txt'))
            # read the label files
            with open(label_file_path, 'r') as f:
                patch_label = int(f.read().strip())
            # get the patch fodler
            patch_label_name = class_idx_label_dict[patch_label]
            # make the label folder if it does not exist
            patch_label_folder = os.path.join(retrain_valid_folder, patch_label_name)
            os.makedirs(patch_label_folder, exist_ok=True)
            # save the image into the patch label folder
            shutil.copy(image_file_path, patch_label_folder)   
        # retrain the model
        print("Instantiating the model...")
        model = YOLO(model_weights_path)
        print("Retraining the model...")
        try:
            # model.train(
            #     data=retrain_dataset_folder, 
            #     epochs=10, 
            #     batch=1, 
            #     imgsz=model_input_size, 
            #     project=checkpoint_folder, 
            #     name='tmp', 
            #     plots=False,
            #     optimizer='Adam', 
            #     lr0=1e-4,
            #     dropout=0.2,
            # )   # move the best weights
            model.train(
                data=retrain_dataset_folder, 
                epochs=10, 
                batch=4, 
                imgsz=model_input_size, 
                project=checkpoint_folder, 
                name='tmp', 
                plots=False,
                # optimizer='Adam', 
                # lr0=1e-4,
                # dropout=0.2,
            )   # move the best weights
        except Exception as e:
            print(f"Error retraining the model: {e}")
        best_weights_path = os.path.join(checkpoint_folder, 'tmp', 'weights', 'best.pt')
        # best_weights_path = os.path.join(checkpoint_folder, 'tmp', 'weights', 'last.pt') # zli85: change to last.pt to always update the weights
        shutil.move(best_weights_path, os.path.join(checkpoint_folder, 'best.pt'))
        # remove all the tmp* folders
        for folder in os.listdir(checkpoint_folder):
            if folder.startswith('tmp'):
                shutil.rmtree(os.path.join(checkpoint_folder, folder))
        return True, "Model retrained successfully."
    except Exception as e:
        # print with traceback details
        print(f'Error retraining classification model: {e}')
        return False, f"Error retraining object detection model with message: {e}"



if __name__ == '__main__':
    # retrain_crop_monitoring_model(
    #     model_weights_path='files/ground_station/weights/latests/crop_monitoring/best.pt',
    #     retraining_dataset_path='files/ground_station/downlink/retraining/crop_monitoring',
    #     validation_dataset_path='files/ground_station/verification/crop_monitoring',
    #     checkpoint_folder='files/ground_station/weights/checkpoints/crop_monitoring_yolov8n_seg'
    # )        
    # retrain_object_detection_model(
    #     model_weights_path='files/ground_station/weights/latests/object_detection_yolov8x.pt',
    #     retraining_dataset_path='files/ground_station/downlink/retraining/object_detection',
    #     validation_dataset_path='files/ground_station/verification/object_detection',
    #     checkpoint_folder='files/ground_station/weights/checkpoints/object_detection_yolov8x'
    # )
    retrain_classification_model(
        model_weights_path='/home/zli85/oec/experiment/scripts/retraining_set_sampling/weights/classification/best.pt',
        model_input_size=(512, 512),
        retraining_dataset_path='/home/zli85/oec/experiment/scripts/retraining_set_sampling/validation_set/classification_retrain',
        validation_dataset_path='/home/zli85/oec/experiment/scripts/retraining_set_sampling/validation_set/classification',
        checkpoint_folder='test_classification_checkpoints'
    )


