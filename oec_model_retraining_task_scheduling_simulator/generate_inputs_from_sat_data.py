import os
import numpy as np
import random
import json

def calculate_number_of_downlinkable_tiles(
        connection_duration, 
        connection_working_time_from_last_connection, 
        connection_num_captured_images_from_last_connection, 
        data_capture_interval,
        number_of_tiles_per_image,
        num_tiles_downlinkable_per_second):
    """
    Calculate the bandwidth limit for a connection
    :param connection_duration: duration of the connection
    :param connection_working_time_from_last_connection: working time from the last connection
    :param connection_num_captured_images_from_last_connection: number of captured images from the last connection
    :param num_tiles_downlinkable_per_second: number of tiles that can be downlinked per second
    :return: bandwidth limit
    """
    # calculate the number of tiles that can be downlinked
    num_of_total_images = connection_working_time_from_last_connection // data_capture_interval * number_of_tiles_per_image
    # calculate the time-share for the considered task
    time_share = connection_num_captured_images_from_last_connection / num_of_total_images * data_capture_interval
    # calculate the number of tiles that can be downlinked
    num_tiles_downlinkable = int(time_share * num_tiles_downlinkable_per_second)
    # return the bandwidth limit
    return num_tiles_downlinkable






def generate_inputs(
    constellation_name,
    prediction_horizon,
    output_file_name,
    NUM_TILES_PER_IMAGE = 100,
    DATA_CAPTURE_INTERVAL = 20,
    NUM_TILES_DOWNLINKABLE_PER_SECOND = 12500,
    ACCURACY_CONVERGE_THRESHOLD = 0.01,
    TASK_RETRAINING_UPPER_BOUND = 100,
    TIME_STEP_TO_SIM = 100,
    TASK_ONBOARD_INTERVAL = 86400 * 1,
    AVERAGE_ONBOARD_TASK_PER_PUBLICATION = 10,
    RANDOM_SEED = 0
):
    random_seed = RANDOM_SEED
    task_scheduling_connection_info_dict_file_path = f'sat_data/{constellation_name}/task_scheduling_connection_info_dict.json'
    task_scheduling_timepoint_location_num_captured_images_file_path = f'sat_data/{constellation_name}/task_scheduling_timepoint_location_num_captured_images.json'

    # read the connection info dict
    with open(task_scheduling_connection_info_dict_file_path, 'r') as f:
        connection_info_dict = json.load(f)

    # read the timepoint location num captured images dict
    with open(task_scheduling_timepoint_location_num_captured_images_file_path, 'r') as f:
        timepoint_location_num_captured_images_dict = json.load(f)

    # get the set of locations
    location_set = set()
    for timepoint in timepoint_location_num_captured_images_dict:
        for location in timepoint_location_num_captured_images_dict[timepoint]:
            # trim the post fix of the location name
            location = location.split('_')[0]
            location_set.add(location)

    number_of_locations = len(location_set)

    # sort the locations in ascending order
    location_list = list(location_set)
    location_list.sort()


    # initialize a v_sequence and b_sequence
    timepoint_location_volume_dict = {} # {timepoint: {location: num_tiles}}
    location_first_appearance_timepoint_dict = {} # {location: first_appearance_timepoint}
    connection_sim_time_dict = {} # {connection: sim_time}
    b_sequence = []
    sim_time_step = 0
    #  for each timepoint, initialize with size as the number of locations
    for timepoint in timepoint_location_num_captured_images_dict:
        timepoint_location_volume_dict[sim_time_step] = {}
        # skip the timepoint if there are no images captured
        if len(timepoint_location_num_captured_images_dict[timepoint]) == 0:
            continue
        # for each location, get the number of images captured
        for location in timepoint_location_num_captured_images_dict[timepoint]:
            # trim the post fix of the location name
            location_name = location.split('_')[0]
            num_images_captured = timepoint_location_num_captured_images_dict[timepoint][location]
            if location_name not in timepoint_location_volume_dict[sim_time_step]:
                timepoint_location_volume_dict[sim_time_step][location_name] = 0
            timepoint_location_volume_dict[sim_time_step][location_name] += num_images_captured * NUM_TILES_PER_IMAGE
            # update the first appearance timepoint of the location
            if location_name not in location_first_appearance_timepoint_dict:
                location_first_appearance_timepoint_dict[location_name] = sim_time_step
        # check the connection info dict to get the data limit
        if timepoint == '0':
            b_sequence.append(0)
            connection_sim_time_dict[sim_time_step] = 0
            sim_time_step += 1
            continue
        connection_info_this_timepoint = connection_info_dict[timepoint]
        connection_duration = connection_info_this_timepoint['duration']
        connection_working_time_from_last_connection = connection_info_this_timepoint['working_time_from_last_connection']
        connection_num_captured_images_from_last_connection = connection_info_this_timepoint['num_captured_images_from_last_connection']
        num_tiles_downlinkable = calculate_number_of_downlinkable_tiles(
            connection_duration, 
            connection_working_time_from_last_connection, 
            connection_num_captured_images_from_last_connection, 
            DATA_CAPTURE_INTERVAL,
            NUM_TILES_PER_IMAGE,
            NUM_TILES_DOWNLINKABLE_PER_SECOND)
        b_sequence.append(num_tiles_downlinkable)
        # get the beginning of the connection in real time point
        connection_sim_time_dict[sim_time_step] = connection_info_this_timepoint['begin_timepoint']
        # update the sim time step
        sim_time_step += 1
    # convert each value in the timepoint_location_num_tiles_dict to an integer
    for timepoint in timepoint_location_volume_dict:
        for location in timepoint_location_volume_dict[timepoint]:
            timepoint_location_volume_dict[timepoint][location] = int(timepoint_location_volume_dict[timepoint][location])
        
    N_T = len(timepoint_location_volume_dict)
    N_K = number_of_locations

    
    np.random.seed(random_seed)  # For reproducibility
    random.seed(random_seed)  # For reproducibility
    location_dataset_path = 'location_evaluation_datasets'  # Path to location datasets
    # base_model_path = os.path.join('weights', 'best.pt')  # Path to model weights
    # model_channel_file_path = os.path.join('weights', 'best_channels.txt')  # Path to model channel file
    base_model_path = os.path.join('weights', 'pruned', 'building1', 'best.pt')  # Path to model weights
    model_channel_file_path = os.path.join('weights', 'pruned', 'building1', 'best_channels.txt')  # Path to model channel file
    model_input_size = (64, 64)  # Model input size
    retrain_train_val_split_ratio = 0.8  # Train validation split
    is_replay = True  # If enable image replay for retraining


    # get location dataset image list
    location_image_path_list = {} # {location: [image_path1, image_path2, ...]}
    for location in os.listdir(location_dataset_path):
        location_image_path_list[location] = []
        for image in os.listdir(os.path.join(location_dataset_path, location, 'images')):
            location_image_path_list[location].append(os.path.join(location_dataset_path, location, 'images', image))
        # sort the image paths in ascending order of the captured time
        location_image_path_list[location].sort()



    # save the sequences to a file
    inputs = {
        'random_seed': random_seed,
        'prediction_horizon': prediction_horizon,
        "timepoint_location_volume_dict": timepoint_location_volume_dict,
        "b_sequence": b_sequence,
        'N_T': N_T,
        'N_K': N_K,
        'location_image_path_list': location_image_path_list,
        'base_model_path': base_model_path,
        'model_channel_file_path': model_channel_file_path,
        'model_input_size': model_input_size,
        'retrain_train_val_split_ratio': retrain_train_val_split_ratio,
        'is_replay': is_replay,
        'accuracy_converge_threshold': ACCURACY_CONVERGE_THRESHOLD,
        'task_retraining_upper_bound': TASK_RETRAINING_UPPER_BOUND,
        'timestep_to_sim': TIME_STEP_TO_SIM,
        'task_onboard_interval': TASK_ONBOARD_INTERVAL,
        'location_first_appearance_timepoint_dict': location_first_appearance_timepoint_dict,
        'connection_sim_time_dict': connection_sim_time_dict,
        'average_onboard_task_per_publication': AVERAGE_ONBOARD_TASK_PER_PUBLICATION
    }
    with open(output_file_name, 'w') as f:
        json.dump(inputs, f, indent=4)



if __name__ == "__main__":

    NUM_TILES_PER_IMAGE = 100 # Number of tiles per full-size image
    DATA_CAPTURE_INTERVAL = 20 # Seconds between two images captured on the satellite
    NUM_TILES_DOWNLINKABLE_PER_SECOND = 12500 # Number of tiles that can be downlinked per second
    ACCURACY_CONVERGE_THRESHOLD = 0.01  # Threshold for accuracy convergence
    TASK_RETRAINING_UPPER_BOUND = 300  # Upper bound for number of images to retrain the model
    TIME_STEP_TO_SIM = 100  # Number of time steps to simulate
    # TASK_ONBOARD_INTERVAL = 10 # Number of rounds to publish new tasks
    TASK_ONBOARD_INTERVAL = 86400 * 1 # Seconds between two task onboardings, for example, 86400 seconds = 1 day
    AVERAGE_ONBOARD_TASK_PER_PUBLICATION = 10 # Average number of tasks onboarded per publication
    RANDOM_SEED = 0  # Random seed for reproducibility

    # for mpc
    prediction_horizon = 20  # Prediction horizon for MPC typically 50
    
    

    # set file paths
    constellation_name = 'starlink_top10_gs'
    # constellation_name = 'sentinel2_sentinel_gs'

    output_file_name = f"inputs_{constellation_name}.json"  # Output file name

    generate_inputs(
        constellation_name,
        prediction_horizon,
        output_file_name,
        NUM_TILES_PER_IMAGE,
        DATA_CAPTURE_INTERVAL,
        NUM_TILES_DOWNLINKABLE_PER_SECOND,
        ACCURACY_CONVERGE_THRESHOLD,
        TASK_RETRAINING_UPPER_BOUND,
        TIME_STEP_TO_SIM,
        TASK_ONBOARD_INTERVAL,
        AVERAGE_ONBOARD_TASK_PER_PUBLICATION,
        RANDOM_SEED
    )