import os
import cv2
import numpy as np
from utils.dataset import UNetDataset
from utils.infer import infer_UNet
from unet import UNet
import torch
from scipy.spatial import distance

class UNetTask(object):
    def __init__(self, model_path, model_channel_filepath, model_input_size, device):
        # self.model = UNet(n_channels=3, n_classes=1, f_channels=model_channel_filepath)
        self.model = None
        self.model_path = model_path
        self.model_channel_filepath = model_channel_filepath
        self.model_input_size = model_input_size
        self.device = device

        # if len(model_path) > 0:
        #     self.model.load_state_dict(model_path)
        # self.model.to(device)

        self.onboard_image_label_path_list = [] # (image_path, label_path)
        self.image_path_pred_result_dict = {} # {image_path: {'dice': dice_value, 'entropy': entropy_value}}

    def update_model(self, new_model_path, new_model_channel_filepath):
        self.model_path = new_model_path
        self.model_channel_filepath = new_model_channel_filepath
    
    def load_model(self, new_model_path, new_model_channel_filepath):
        self.model = UNet(n_channels=3, n_classes=1, f_channels=new_model_channel_filepath)
        if len(new_model_path) > 0:
            self.model.load_state_dict(torch.load(new_model_path))
        self.model.to(self.device)
    
    def unload_model(self):
        del self.model
        self.model = None

    def predict(self, input_data_folder):
        # load model if not already loaded
        if self.model is None:
            self.load_model(self.model_path, self.model_channel_filepath)

        # Prepare dataset
        image_folder = os.path.join(input_data_folder, 'images')
        label_folder = os.path.join(input_data_folder, 'masks')

        # add images to the onboard list
        image_file_name_list = os.listdir(image_folder)
        for image_file_name in image_file_name_list:
            image_path = os.path.join(image_folder, image_file_name)
            label_path = os.path.join(label_folder, image_file_name)
            self.onboard_image_label_path_list.append((image_path, label_path))

        # Perform predictions
        dataset = UNetDataset(image_folder, label_folder, im_size=self.model_input_size)
        image_name_pred_result_dict, mean_dice, mean_entropy = infer_UNet(self.model, dataset, self.device)
        image_path_pred_result_dict = {}
        for image_name, result in image_name_pred_result_dict.items():
            image_path = os.path.join(image_folder, image_name)
            image_path_pred_result_dict[image_path] = result
        self.image_path_pred_result_dict.update(image_path_pred_result_dict)

        # Clean up
        del dataset
        self.unload_model()
        return mean_dice, mean_entropy

    def get_prediction_results(self):
        return self.image_path_pred_result_dict

    def get_onboard_image_label_path_list(self):
        return self.onboard_image_label_path_list

    def remove_downlinked_onboard_image_label_path(self, downlinked_image_path_list):
        self.onboard_image_label_path_list = [item for item in self.onboard_image_label_path_list if item[0] not in downlinked_image_path_list]

    def sort_onboard_image_label_path_list_with_distance_to_center(self):
        # read all the images and get the center of the images
        image_path_image_dict = {} # {image_path: image}
        for image_path, _ in self.onboard_image_label_path_list:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = np.array(image, dtype=np.float32) / 255.0
            image = image.flatten()
            image_path_image_dict[image_path] = image

        # calculate the distance to the center
        image_center = np.mean(np.array(list(image_path_image_dict.values())), axis=0)
        image_center_dict = {} # {image_path: distance_to_center}
        for image_path, image in image_path_image_dict.items():
            distance_to_center = distance.cosine(image, image_center) # use cosine distance
            image_center_dict[image_path] = distance_to_center

        # sort the onboard image list based on the distance to the center
        self.onboard_image_label_path_list = sorted(self.onboard_image_label_path_list, key=lambda x: image_center_dict[x[0]])







