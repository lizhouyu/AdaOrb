import os
import json
import argparse

from simulator import Simulator
from generate_inputs_from_sat_data import generate_inputs

if __name__ == "__main__":
    # hyperparameters
    NUM_TILES_PER_IMAGE = 100 # Number of tiles per full-size image
    ACCURACY_CONVERGE_THRESHOLD = 0.01  # Threshold for accuracy convergence
    TASK_RETRAINING_UPPER_BOUND = 300  # Upper bound for number of images to retrain the model
    TIME_STEP_TO_SIM = 100  # Number of time steps to simulate
    TASK_ONBOARD_INTERVAL = 86400 * 1 # Seconds between two task onboardings, for example, 86400 seconds = 1 day
    AVERAGE_ONBOARD_TASK_PER_PUBLICATION = 10 # Average number of tasks onboarded per publication
    NUM_TILES_DOWNLINKABLE_PER_SECOND = 12500  # Number of tiles that can be downlinked per second
    ALLOCATOR = "mpc"  # Allocator to use
    # PREDICTION_HORIZON = 10  # Prediction horizon for the MPC allocator

    args = argparse.ArgumentParser()
    args.add_argument("-c", "--constellation_name", type=str, default="starlink_top10_gs")
    args.add_argument("-p", "--prediction_horizon_list", type=str, default="10,15,20,25,30,35,40")
    args.add_argument("-r", "--random_seed_list", type=str, default="0,1,2")
    args = args.parse_args()

    prediction_horizon_list = [int(i) for i in args.prediction_horizon_list.split(",")]
    random_seed_list = [int(i) for i in args.random_seed_list.split(",")]

    # output root folder
    output_data_folder = 'data'
    prediction_horizon_exp_folder = 'prediction_horizon_exp'
    output_root_folder = os.path.join(output_data_folder, prediction_horizon_exp_folder)
    os.makedirs(output_root_folder, exist_ok=True)

    constellation_name_list = [args.constellation_name]
    # constellation_name_list = ['sentinel2_sentinel_gs']

    for constellation_name in constellation_name_list:
        # make output folder for each constellation
        constellation_output_folder = os.path.join(output_root_folder, constellation_name)
        os.makedirs(constellation_output_folder, exist_ok=True)
        # data_capture_interval = 10 if 'starlink' in constellation_name else 20 # the data capture interval is 10 seconds for starlink and 20 seconds for sentinel2
        data_capture_interval = 20 # zli85 12/29/2024: the data capture interval is 20 seconds for all constellations
        for prediction_horizon in prediction_horizon_list:
            # get the root folder for the number of tiles downlinkable per second
            prediction_horizon_output_folder = os.path.join(constellation_output_folder, f"prediction_horizon_{prediction_horizon}")
            os.makedirs(prediction_horizon_output_folder, exist_ok=True)
            for random_seed in random_seed_list:
                # get the output folder for each random seed
                random_seed_output_folder = os.path.join(prediction_horizon_output_folder, f"random_seed_{random_seed}")
                os.makedirs(random_seed_output_folder, exist_ok=True)
                # set the input file path
                exp_input_file_path = os.path.join(random_seed_output_folder, f"inputs.json")
                # set the output record folder
                exp_record_folder = os.path.join(random_seed_output_folder, f"records")
                os.makedirs(exp_record_folder, exist_ok=True)
                # Generate inputs
                generate_inputs(
                    constellation_name,
                    prediction_horizon,
                    exp_input_file_path, # output file path for the generated inputs
                    NUM_TILES_PER_IMAGE,
                    data_capture_interval,
                    NUM_TILES_DOWNLINKABLE_PER_SECOND,
                    ACCURACY_CONVERGE_THRESHOLD,
                    TASK_RETRAINING_UPPER_BOUND,
                    TIME_STEP_TO_SIM,
                    TASK_ONBOARD_INTERVAL,
                    AVERAGE_ONBOARD_TASK_PER_PUBLICATION,
                    random_seed
                )
                # Run simulator
                simulator = Simulator(
                    exp_input_file_path,
                    ALLOCATOR,
                    exp_record_folder
                )
                simulator.run()
                # Save results
                simulator.write_utilities_to_file(exp_record_folder)
                simulator.write_downlink_records_to_file(exp_record_folder)
                simulator.write_accuracy_records_to_file(exp_record_folder)
                simulator.write_accumulated_utilities_to_file(exp_record_folder)