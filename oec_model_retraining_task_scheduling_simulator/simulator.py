import os
import json
import torch
import shutil
import numpy as np

from utils.logger import getMyLogger
from unet import UNet
from allocator.allocator_mpc import MPCAllocator
from allocator.allocator_random import RandomAllocator
from allocator.allocator_acc import AccAllocator
from allocator.allocator_data import Datallocator
from utils.dataset import UNetDataset
from unet_task import UNetTask
from retrain_models import retrain_cloud_detection_model

class Simulator(object):
    def __init__(self, input_file_path, allocator_type='mpc', cache_folder='records'):
        
        self.inputs = self.load_inputs(input_file_path)
        self.prediction_horizon = self.inputs['prediction_horizon']
        self.num_time_steps = self.inputs['N_T']
        self.model_input_size = self.inputs['model_input_size']
        self.is_replay = self.inputs['is_replay']
        self.retrain_train_val_split_ratio = self.inputs['retrain_train_val_split_ratio']
        self.random_seed = self.inputs['random_seed']
        self.cache_folder = cache_folder
        self.task_retraining_upper_bound = self.inputs['task_retraining_upper_bound']
        self.timestep_to_sim = self.inputs['timestep_to_sim']
        self.task_onboard_interval = self.inputs['task_onboard_interval']
        self.location_first_appearance_timepoint_dict = self.inputs['location_first_appearance_timepoint_dict']
        self.connection_sim_time_dict = self.inputs['connection_sim_time_dict']
        self.average_onboard_task_per_publication = self.inputs['average_onboard_task_per_publication']

        # initialize loggers
        log_path = os.path.join(cache_folder, f"log_simulator_{allocator_type}.log")
        self.logger = getMyLogger(log_path)

        # set random seed for reproducibility
        np.random.seed(self.random_seed)

        # get device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # initialize the allocator
        if allocator_type == 'mpc':
            self.allocator = MPCAllocator(self.inputs, cache_folder)
        elif allocator_type == 'random':
            self.allocator = RandomAllocator(self.inputs, cache_folder)
        elif allocator_type == 'acc':
            self.allocator = AccAllocator(self.inputs, cache_folder)
        elif allocator_type == 'data':
            self.allocator = Datallocator(self.inputs, cache_folder)
        else:
            raise ValueError(f"Unknown allocator type: {allocator_type}")
        self.logger.info(f"Allocator initialized: {allocator_type}")


        # initialize time point location volume dict
        self.timepoint_location_num_tiles_dict = {}
        for timepoint, location_num_tiles_dict in self.inputs['timepoint_location_volume_dict'].items():
            self.timepoint_location_num_tiles_dict[int(timepoint)] = location_num_tiles_dict

        # initialize bandwith sequence
        self.bandwidth_sequence = self.inputs['b_sequence']

        # initialize onboard task list
        # task_location_list = list(self.inputs['location_image_path_list'].keys())[:self.num_tasks]
        self.onboard_task = {} # {task_id: {roi: roi, model: model, model_weight_path: model_weight_path, model_channel_file_path: model_channel_file_path, time_step_accuracy_list_dict: {time_step: accuracy_list}, status: status (active/finished) }}
        self.onboard_roi_task_id_dict = {} # {roi: task_id} # this can be used to check if a roi is already onboarded, and to get the task_id for a roi
        # for task_id in range(self.num_tasks):
        #     self.task_list.append({
        #         'task_id': task_id,
        #         'roi': task_location_list[task_id],
        #         'model_weight_path': self.inputs['base_model_path'],
        #         'model_channel_file_path': self.inputs['model_channel_file_path'],
        #         'model_input_size': self.model_input_size,
        #         'model': UNetTask(self.inputs['base_model_path'], self.inputs['model_channel_file_path'], self.model_input_size, self.device),
        #         'time_step_accuracy_list_dict': {} # {time_step: accuracy_list}
        #     })

        # # load task image list
        # self.task_roi_image_path_list_dict = {}
        # for task in self.task_list:
        #     self.task_roi_image_path_list_dict[task['roi']] = self.inputs['location_image_path_list'][task['roi']]

        # load captured image path list from the inputs
        self.location_image_path_list_dict = self.inputs['location_image_path_list']

        # # get captured image path list per time step
        # timestep_location_num_tiles_dict = self.inputs['timepoint_location_volume_dict']
        # self.captured_image_path_list = {} # task_id: {time_step: [image_path1, image_path2, ...]}
        # for time_step in range(self.num_time_steps):
        #     self.captured_image_path_list[time_step] = {}
        #     for task in self.task_list:
        #         task_id = task['task_id']
        #         roi = task['roi']
        #         num_images = num_images_per_task_per_time_step[time_step][task_id]
        #         if num_images == 0:
        #             continue
        #         self.captured_image_path_list[time_step][task_id] = self.task_roi_image_path_list_dict[roi][:num_images]
        #         # pop the used images from the list
        #         self.task_roi_image_path_list_dict[roi] = self.task_roi_image_path_list_dict[roi][num_images:]        
        # print("Captured image path list initialized:", self.captured_image_path_list)

        # initialize downlink records
        self.time_step_task_downlink_amount_dict = {} # {time_step: {task_id: downlink_amount}}
        # self.current_total_downlink_amount = []
        # initialize accuracy records
        self.time_step_task_accuracy_dict = {} # {time_step: {task_id: accuracy}} # this is the accuracy used for evaluation
        self.time_step_task_validation_accuracy_dict = {} # {time_step: {task_id: validation_accuracy}} # this is the accuracy used for decision making
        # self.current_accuracy_list = []
        # initialize total utility record
        self.time_step_utility_dict = {} # {time_step: time_step_utility}
        self.total_utility = 0

        # initialize time step counter
        self.current_time_step = 0
        # initialize task onboard interval counter
        self.task_onboard_counter = 0

        # create simulator cache directory
        self.cache_dir = os.path.join(cache_folder, f"simulator_{allocator_type}_{self.random_seed}")
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)


    def load_inputs(self, input_file_path):
        with open(input_file_path, 'r') as f:
            inputs = json.load(f)
        return inputs

    def load_model(self, model_weight_path, model_channel_file_path):
        model = UNet(n_channels=3, n_classes=1, f_channels=model_channel_file_path)
        if len(model_weight_path) > 0:
            model.load_state_dict(torch.load(model_weight_path))
        else:
            # raise Warning("No model weights provided, initializing model with random weights.")
            print("No model weights provided, initializing model with random weights.")
        model.eval()
        return model

    def run(self):
        # # onboard all tasks
        # ## get the number of tiles for area of interest for this time step
        # for time_step, location_num_tiles_dict in self.timepoint_location_num_tiles_dict.items():
        #     time_step_location_num_tiles_dict = self.timepoint_location_num_tiles_dict[time_step]
        #     ## update the onboard task list
        #     for roi, num_tiles in time_step_location_num_tiles_dict.items():
        #         if roi not in self.onboard_roi_task_id_dict:
        #             ### create a new task
        #             task_id = len(self.onboard_task)
        #             self.onboard_roi_task_id_dict[roi] = task_id
        #             self.onboard_task[task_id] = {
        #                 'roi': roi,
        #                 'model_weight_path': self.inputs['base_model_path'],
        #                 'model_channel_file_path': self.inputs['model_channel_file_path'],
        #                 'model_input_size': self.model_input_size,
        #                 'model': UNetTask(self.inputs['base_model_path'], self.inputs['model_channel_file_path'], self.model_input_size, self.device),
        #                 'time_step_accuracy_list_dict': {}, # {time_step: accuracy_list},
        #                 'status': 'active',
        #                 'current_accuracy': 0,
        #                 'downlinked_image_path_list': [],
        #                 'num_totoal_captured_images': 0,
        #                 'roi_image_path_list': self.location_image_path_list_dict[roi]
        #             }
        #             self.logger.info(f"Task {task_id} onboarded for ROI {roi}.")     
        # Simulation logic would go here
        for _ in range(self.timestep_to_sim):
            self.make_step()
            # update utility
            new_total_utility = self.get_utility()
            self.time_step_utility_dict[self.current_time_step] = new_total_utility - self.total_utility
            self.total_utility = new_total_utility
            self.logger.info(f"Utility for time step {self.current_time_step}: {self.time_step_utility_dict[self.current_time_step]}")
            self.logger.info(f"Total utility until time step {self.current_time_step}: {self.total_utility}")
            self.logger.info(f"Downlink records until time step {self.current_time_step}: {self.time_step_task_downlink_amount_dict}")
            self.write_utilities_to_file(self.cache_folder)
            self.write_downlink_records_to_file(self.cache_folder)
            self.write_accuracy_records_to_file(self.cache_folder)
            self.write_accumulated_utilities_to_file(self.cache_folder)
            self.write_roi_task_id_dict_to_file(self.cache_folder)
            self.write_validation_accuracy_records_to_file(self.cache_folder)
            self.write_task_inference_results_to_file(self.cache_folder)
            self.write_task_important_timepoints_to_file(self.cache_folder)

    def make_step(self):
        # time step initialization
        self.logger.info(f"Starting time step {self.current_time_step}.")
        ## get this time step's location-volume dict
        time_step_location_num_tiles_dict = self.timepoint_location_num_tiles_dict[self.current_time_step]
        ## update the onboard task list if at the task onboard interval
        ### get the real timepoint for this time step
        real_timepoint = self.connection_sim_time_dict[str(self.current_time_step)] if self.current_time_step > 0 else 0
        ### check if the real timepoint is above the current task onboard interval
        if real_timepoint >= self.task_onboard_counter * self.task_onboard_interval:
            self.task_onboard_counter += 1
            #### get the time steps before the next task publish
            time_step_list_before_next_task_publish = []
            for time_step in range(self.current_time_step, self.timestep_to_sim + 1):
                time_step_real_timepoint = self.connection_sim_time_dict[str(time_step)]
                if time_step_real_timepoint >= self.task_onboard_counter * self.task_onboard_interval:
                    break
                time_step_list_before_next_task_publish.append(time_step)
            #### for each time step in the list, get the locations to be onboarded
            ##### get the maximum number of onboard tasks for this publication following a Poisson distribution
            maximum_num_onboard_tasks = np.random.poisson(self.average_onboard_task_per_publication)
            location_set_to_be_onboarded = set()
            is_max_num_onboard_tasks_reached = False
            for time_step in time_step_list_before_next_task_publish:
                for roi, num_tiles in self.timepoint_location_num_tiles_dict[time_step].items():
                    if roi not in self.onboard_roi_task_id_dict:
                        ##### check if the maximum number of onboard tasks is reached
                        if len(location_set_to_be_onboarded) >= maximum_num_onboard_tasks:
                            is_max_num_onboard_tasks_reached = True
                            break
                        ##### add it to the set of locations to be onboarded
                        location_set_to_be_onboarded.add(roi)
                if is_max_num_onboard_tasks_reached:
                    break
            #### onboard the locations in the set
            for roi in location_set_to_be_onboarded:
                if roi in self.onboard_roi_task_id_dict:
                    continue
                ### create a new task
                task_id = len(self.onboard_task)
                self.onboard_roi_task_id_dict[roi] = task_id
                self.onboard_task[task_id] = {
                    'roi': roi,
                    'model_weight_path': self.inputs['base_model_path'],
                    'model_channel_file_path': self.inputs['model_channel_file_path'],
                    'model_input_size': self.model_input_size,
                    'model': UNetTask(self.inputs['base_model_path'], self.inputs['model_channel_file_path'], self.model_input_size, self.device),
                    'time_step_accuracy_list_dict': {}, # {time_step: accuracy_list},
                    'status': 'active',
                    'current_accuracy': 0,
                    'downlinked_image_path_list': [],
                    'num_totoal_captured_images': 0,
                    'roi_image_path_list': self.location_image_path_list_dict[roi],
                    'important_time_points': {
                        'onboard': self.current_time_step,
                    } # this dictionary is used to record important time points for the task, including the onboard_timepoint and finish_timepoint 
                }
                self.logger.info(f"Task {task_id} onboarded for ROI {roi}.")        
            #### clear the set of locations to be onboarded
            location_set_to_be_onboarded.clear()

        ## make downlink decision (if time step is not 0)
        if self.current_time_step > 0:
            ### initialize downlink record for this time step
            self.time_step_task_downlink_amount_dict[self.current_time_step] = {}
            # prior to ground-satellite contact
            ## prepare v_sequence, b_sequence, state_vol, state_down, and state_acc for this time step with all the onboard tasks
            active_task_ids = [task_id for task_id, task_info in self.onboard_task.items() if task_info['status'] == 'active']
            if len(active_task_ids) == 0:
                downlink_decision = []
            else:
                v_sequence = [[0] * len(active_task_ids) for _ in range(self.prediction_horizon)]
                state_vol = [0] * len(active_task_ids)
                state_down = [0] * len(active_task_ids)
                state_acc = [0] * len(active_task_ids)
                b_sequence = self.bandwidth_sequence[self.current_time_step:self.current_time_step+self.prediction_horizon]
                sequence_idx_task_id_dict = {idx: task_id for idx, task_id in enumerate(active_task_ids)}
                task_id_sequence_idx_dict = {task_id: idx for idx, task_id in enumerate(active_task_ids)}
                ### prepare state_vol, state_down, and state_acc
                for task_id in active_task_ids:
                    task_info = self.onboard_task[task_id]
                    sequence_idx = task_id_sequence_idx_dict[task_id]
                    state_vol[sequence_idx] = task_info['num_totoal_captured_images']
                    state_down[sequence_idx] = len(task_info['downlinked_image_path_list'])
                    state_acc[sequence_idx] = task_info['current_accuracy']
                ### prepare v_sequence
                for v_sequence_time_step in range(self.prediction_horizon):
                    location_num_tiles_dict = self.timepoint_location_num_tiles_dict[self.current_time_step + v_sequence_time_step]
                    print("location_num_tiles_dict:", location_num_tiles_dict)
                    for location, num_tiles in location_num_tiles_dict.items():
                        if location not in self.onboard_roi_task_id_dict:
                            continue
                        task_id = self.onboard_roi_task_id_dict[location]
                        if task_id not in active_task_ids:
                            continue
                        sequence_idx = task_id_sequence_idx_dict[task_id]
                        v_sequence[v_sequence_time_step][sequence_idx] = num_tiles
                self.logger.info(f"v_sequence for time step {self.current_time_step}: {v_sequence}")
                self.logger.info(f"b_sequence for time step {self.current_time_step}: {b_sequence}")
                self.logger.info(f"state_vol for time step {self.current_time_step}: {state_vol}")
                self.logger.info(f"state_down for time step {self.current_time_step}: {state_down}")
                self.logger.info(f"state_acc for time step {self.current_time_step}: {state_acc}")
                downlink_decision = self.allocator.make_downlink_decision(self.current_time_step, v_sequence, b_sequence, state_vol, state_down, state_acc)
            for sequence_idx in range(len(downlink_decision)):
                task_id = sequence_idx_task_id_dict[sequence_idx]
                self.time_step_task_downlink_amount_dict[self.current_time_step][task_id] = downlink_decision[sequence_idx]
            self.logger.info(f"Downlink decision for time step {self.current_time_step}: {downlink_decision}")        

        # during ground-satellite contact
        ## downlink images for retraining
        if self.current_time_step > 0:
            for task_id, task_info in self.onboard_task.items():
                if task_id in self.time_step_task_downlink_amount_dict[self.current_time_step]:
                    downlink_amount = self.time_step_task_downlink_amount_dict[self.current_time_step][task_id]
                    if downlink_amount > 0:
                        # update current total downlink amount
                        # self.current_total_downlink_amount[task_id] += downlink_amount
                        self.logger.info(f"Task {task_id} downlinked {downlink_amount} images.")
                        # compute number of images for training and validation
                        num_train_images = int(self.retrain_train_val_split_ratio * downlink_amount)
                        num_val_images = downlink_amount - num_train_images
                        # sort task's onboard image label path list with distance to center
                        task_info['model'].sort_onboard_image_label_path_list_with_distance_to_center()
                        # get onboard image label path list
                        onboard_image_label_path_list = task_info['model'].get_onboard_image_label_path_list()
                        # prepare retraining cache directory
                        if self.is_replay:
                            retrain_cache_dir = os.path.join(self.cache_dir, 'retrain', str(task_id))
                        else:
                            retrain_cache_dir = os.path.join(self.cache_dir, 'retrain', str(self.current_time_step), str(task_id))
                        os.makedirs(retrain_cache_dir, exist_ok=True)
                        retraining_train_dataset_folder = os.path.join(retrain_cache_dir, 'train')
                        os.makedirs(retraining_train_dataset_folder, exist_ok=True)
                        retraining_val_dataset_folder = os.path.join(retrain_cache_dir, 'val')
                        os.makedirs(retraining_val_dataset_folder, exist_ok=True)
                        retrain_train_image_folder = os.path.join(retraining_train_dataset_folder, 'images')
                        retrain_train_label_folder = os.path.join(retraining_train_dataset_folder, 'labels')
                        retrain_val_image_folder = os.path.join(retraining_val_dataset_folder, 'images')
                        retrain_val_label_folder = os.path.join(retraining_val_dataset_folder, 'labels')
                        os.makedirs(retrain_train_image_folder, exist_ok=True)
                        os.makedirs(retrain_train_label_folder, exist_ok=True)
                        os.makedirs(retrain_val_image_folder, exist_ok=True)
                        os.makedirs(retrain_val_label_folder, exist_ok=True)          
                        # select images for validation with equal distance to center (equally from the fetched onboard image label path list)
                        self.logger.info(f"From {len(onboard_image_label_path_list)} onboard images for task {task_id} selected {num_val_images} images for validation.")
                        sample_step = len(onboard_image_label_path_list) // num_val_images
                        selected_val_images = [onboard_image_label_path_list[i] for i in range(0, len(onboard_image_label_path_list), sample_step)]
                        # copy selected validation images to retraining validation dataset folder
                        for image_path, label_path in selected_val_images:
                            shutil.copy(image_path, retrain_val_image_folder)
                            shutil.copy(label_path, retrain_val_label_folder)
                        # update the onboard image label path list by removing the selected validation images
                        onboard_image_label_path_list = [item for item in onboard_image_label_path_list if item not in selected_val_images]             
                        # randomly select num_train_images images from onboard_image_label_path_list
                        self.logger.info(f"From {len(onboard_image_label_path_list)} onboard images for task {task_id} selected {downlink_amount} images for retraining.")
                        selected_train_images_indice_list = np.random.choice(np.arange(len(onboard_image_label_path_list)), num_train_images, replace=False)
                        selected_train_images = [onboard_image_label_path_list[i] for i in selected_train_images_indice_list]
                        # copy selected images to cache directory for retraining
                        for image_path, label_path in selected_train_images:
                            shutil.copy(image_path, retrain_train_image_folder)
                            shutil.copy(label_path, retrain_train_label_folder)
                        # get selected train and val image paths
                        selected_image_list = selected_train_images + selected_val_images
                        # update the onboard image label path list by removing the selected training images
                        task_info['model'].remove_downlinked_onboard_image_label_path(selected_image_list)
                        # update downlinked image path list
                        task_info['downlinked_image_path_list'] += selected_image_list

        # the online inference and offline retraining are performed in the same time period in the real system
        # contact finished, satellite moving
        ## for every task with image in this time step, do inference
        self.logger.info(f"Performing inference for time step {self.current_time_step}.")
        self.logger.info(f"Number of onboard tasks: {len(self.onboard_task)}")
        self.time_step_task_accuracy_dict[self.current_time_step] = {}
        for task_id, task_info in self.onboard_task.items():
            # skip if the task does not have any images in this time step
            if task_info['roi'] not in time_step_location_num_tiles_dict:
                self.logger.info(f"Task {task_id} does not have any images in time step {self.current_time_step}.")
                continue
            # get the images for this task in this time step
            num_captured_images_this_time_step = time_step_location_num_tiles_dict[task_info['roi']]
            task_captured_image_list_this_time_step = task_info['roi_image_path_list'][:num_captured_images_this_time_step]
            # update the onboard task info
            task_info['num_totoal_captured_images'] += num_captured_images_this_time_step
            task_info['roi_image_path_list'] = task_info['roi_image_path_list'][num_captured_images_this_time_step:]
            # perform inference
            # copy the images to the model for inference
            inference_cache_dir = os.path.join(self.cache_dir, 'inference', str(self.current_time_step), str(task_id))
            os.makedirs(inference_cache_dir, exist_ok=True)
            inference_image_folder = os.path.join(inference_cache_dir, 'images')
            os.makedirs(inference_image_folder, exist_ok=True)
            inference_label_folder = os.path.join(inference_cache_dir, 'masks')
            os.makedirs(inference_label_folder, exist_ok=True)
            for image_path in task_captured_image_list_this_time_step:
                # Copy image to inference cache directory
                shutil.copy(image_path, inference_image_folder)
                label_path = image_path.replace('images', 'masks')
                shutil.copy(label_path, inference_label_folder)
            # use task model to perform inference
            mean_dice, mean_entropy = task_info['model'].predict(inference_cache_dir)
            predictions = task_info['model'].get_prediction_results()
            task_info['time_step_accuracy_list_dict'][self.current_time_step] = predictions
            self.logger.info(f"Task {task_id} completed inference for time step {self.current_time_step}. Predictions: {predictions}")
            # record accuracy
            self.time_step_task_accuracy_dict[self.current_time_step][task_id] = mean_dice

        ## retrain model and update model weights
        if self.current_time_step > 0:
            if self.current_time_step not in self.time_step_task_validation_accuracy_dict:
                self.time_step_task_validation_accuracy_dict[self.current_time_step] = {}
            for task_id, task_info in self.onboard_task.items():
                if task_id in self.time_step_task_downlink_amount_dict[self.current_time_step]:
                    downlink_amount = self.time_step_task_downlink_amount_dict[self.current_time_step][task_id]
                    if downlink_amount > 0:
                        # update current total downlink amount
                        self.logger.info(f"Task {task_id} downlinked {downlink_amount} images.")
                        # select images for retraining
                        onboard_image_label_path_list = task_info['model'].get_onboard_image_label_path_list()
                        self.logger.info(f"Onboard images for task {task_id}: {onboard_image_label_path_list}")
                        # randomly select downlink_amount images from onboard_image_label_path_list
                        selected_images_indice_list = np.random.choice(np.arange(len(onboard_image_label_path_list)), int(downlink_amount), replace=False)
                        selected_images = [onboard_image_label_path_list[i] for i in selected_images_indice_list]
                        # copy selected images to cache directory for retraining
                        if self.is_replay:
                            retrain_cache_dir = os.path.join(self.cache_dir, 'retrain', str(task_id))
                        else:
                            retrain_cache_dir = os.path.join(self.cache_dir, 'retrain', str(self.current_time_step), str(task_id))
                        assert os.path.exists(retrain_cache_dir), "Retraining cache directory does not exist."
                        retrain_train_dataset_folder = os.path.join(retrain_cache_dir, 'train')
                        retrain_val_dataset_folder = os.path.join(retrain_cache_dir, 'val')
                        assert os.path.exists(retrain_train_dataset_folder), "Retraining train dataset folder does not exist."
                        assert os.path.exists(retrain_val_dataset_folder), "Retraining validation dataset folder does not exist."
                        # check the number of images in the retraining dataset folders
                        num_train_images = len(os.listdir(os.path.join(retrain_train_dataset_folder, 'images')))
                        num_val_images = len(os.listdir(os.path.join(retrain_val_dataset_folder, 'images')))
                        ## if not both train and val dataset folders have images, skip retraining
                        if num_train_images == 0 or num_val_images == 0:
                            self.logger.info(f"Task {task_id} does not have enough images for retraining. Skipping retraining.")
                            continue
                        # make a retraining checkpoint cache folder
                        retrain_checkpoint_cache_dir = os.path.join(retrain_cache_dir, 'checkpoint')
                        os.makedirs(retrain_checkpoint_cache_dir, exist_ok=True)
                        # retrain the model
                        dice_mean, dice_std, entropy_mean, entropy_std = retrain_cloud_detection_model(task_info['model_weight_path'], task_info['model_input_size'], retrain_train_dataset_folder, retrain_val_dataset_folder, retrain_checkpoint_cache_dir)
                        # check if the retrained accuracy is better than the current accuracy
                        if dice_mean > task_info['current_accuracy']:
                            # update model weights
                            if os.path.exists(os.path.join(retrain_checkpoint_cache_dir, 'best.pt')):
                                self.logger.info(f"Updating model weights for task {task_id} with new weights from {retrain_checkpoint_cache_dir}.")
                                task_info['model_weight_path'] = os.path.join(retrain_checkpoint_cache_dir, 'best.pt')
                                self.logger.info(f"Updating model channel file for task {task_id} with new weights from {task_info['model_weight_path']} and channels from {task_info['model_channel_file_path']}.") 
                                task_info['model'].update_model(task_info['model_weight_path'], task_info['model_channel_file_path'])
                            # check i
                            # update model current accuracy as the evaluation result of the validation set
                            task_info['current_accuracy'] = dice_mean
                            self.time_step_task_validation_accuracy_dict[self.current_time_step][task_id] = dice_mean
                        else:
                            self.logger.info(f"Retrained model accuracy is not better than the current accuracy for task {task_id}.")
                            self.time_step_task_validation_accuracy_dict[self.current_time_step][task_id] = task_info['current_accuracy']
                        # check the number of downlinked images, if it exceed the number of task retraning upper bound, set the task status to finished
                        if task_info['num_totoal_captured_images'] >= self.task_retraining_upper_bound:
                            task_info['status'] = 'finished'
                            task_info['important_time_points']['finish'] = self.current_time_step
                            self.logger.info(f"Task {task_id} status set to finished.")
                        

        # increment time step
        self.current_time_step += 1

    def get_utility(self):
        total_utility = 0
        for task_id, task_info in self.onboard_task.items():
            image_path_pred_result_dict = task_info['model'].get_prediction_results()
            for result in image_path_pred_result_dict.values():
                total_utility += result['dice']
        return total_utility

    def write_utilities_to_file(self, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        output_file_path = os.path.join(output_folder, f"utility_{str(self.allocator.__class__.__name__)}_{self.random_seed}.json")
        with open(output_file_path, 'w') as f:
            json.dump({
                'time_step_utility_dict': self.time_step_utility_dict,
                'total_utility': self.total_utility
            }, f, indent=4)
        
    def write_accumulated_utilities_to_file(self, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        output_file_path = os.path.join(output_folder, f"accumulated_utility_{str(self.allocator.__class__.__name__)}_{self.random_seed}.json")
        # compute accumulated utility
        accumulated_utility_dict = {}
        accumulated_utility = 0
        self.time_step_utility_dict = dict(sorted(self.time_step_utility_dict.items()))
        for time_step, utility in self.time_step_utility_dict.items():
            accumulated_utility += utility
            accumulated_utility_dict[time_step] = accumulated_utility
        with open(output_file_path, 'w') as f:
            json.dump(accumulated_utility_dict, f, indent=4)
    
    def write_downlink_records_to_file(self, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        output_file_path = os.path.join(output_folder, f"downlink_records_{str(self.allocator.__class__.__name__)}_{self.random_seed}.json")
        # convert eah value to int
        for time_step, downlink_amount_dict in self.time_step_task_downlink_amount_dict.items():
            for task_id, downlink_amount in downlink_amount_dict.items():
                downlink_amount_dict[task_id] = int(downlink_amount)
        with open(output_file_path, 'w') as f:
            json.dump(self.time_step_task_downlink_amount_dict, f, indent=4)

    def write_roi_task_id_dict_to_file(self, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        output_file_path = os.path.join(output_folder, f"roi_task_id_dict_{str(self.allocator.__class__.__name__)}_{self.random_seed}.json")
        with open(output_file_path, 'w') as f:
            json.dump(self.onboard_roi_task_id_dict, f, indent=4)

    def write_task_important_timepoints_to_file(self, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        output_file_path = os.path.join(output_folder, f"task_important_timepoints_{str(self.allocator.__class__.__name__)}_{self.random_seed}.json")
        task_important_timepoints = {}
        for task_id, task_info in self.onboard_task.items():
            task_important_timepoints[task_id] = task_info['important_time_points']
        with open(output_file_path, 'w') as f:
            json.dump(task_important_timepoints, f, indent=4)

    # this function write down the validation accuracy for each time step
    # this data is available for models in the real system and used for decision making
    def write_validation_accuracy_records_to_file(self, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        output_file_path = os.path.join(output_folder, f"validation_accuracy_records_{str(self.allocator.__class__.__name__)}_{self.random_seed}.json")
        with open(output_file_path, 'w') as f:
            json.dump(self.time_step_task_validation_accuracy_dict, f, indent=4)

    # this function write down actual model average accuracy for each time step
    # this data should not be available for models in the real system
    def write_accuracy_records_to_file(self, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        output_file_path = os.path.join(output_folder, f"accuracy_records_{str(self.allocator.__class__.__name__)}_{self.random_seed}.json")
        with open(output_file_path, 'w') as f:
            json.dump(self.time_step_task_accuracy_dict, f, indent=4)

    # this function write down the model prediction results for each time step
    def write_task_inference_results_to_file(self, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        output_file_path = os.path.join(output_folder, f"task_inference_results_{str(self.allocator.__class__.__name__)}_{self.random_seed}.json")
        task_inference_results = {}
        for task_id, task_info in self.onboard_task.items():
            task_inference_results[task_id] = task_info['model'].get_prediction_results()
        with open(output_file_path, 'w') as f:
            json.dump(task_inference_results, f, indent=4)

if __name__ == "__main__":
    input_file_path = "inputs_sentinel2_sentinel_gs.json"  # Path to the input file
    os.makedirs('debug_records', exist_ok=True)
    simulator = Simulator(input_file_path, allocator_type='mpc', cache_folder='debug_records')  # Initialize the simulator
    simulator.run()  # Start the simulation
    simulator.write_utilities_to_file("utility")  # Write utilities to file
    print("Simulation completed.")
    print("Total utility:", simulator.get_utility())  # Print total utility after simulation
    # Further simulation logic would go here
    # e.g., simulator.run()