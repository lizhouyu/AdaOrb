import os
import json
import argparse

from simulator import Simulator

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-c", "--constellation_name", type=str, default="starlink_top10_gs") # starlink_top10_gs or sentinel2_sentinel_gs
    args = arg_parser.parse_args()

    constellation_name = args.constellation_name

    input_path = f"inputs_{constellation_name}.json"
    output_root_folder = 'data'
    record_folder = os.path.join(output_root_folder, f"records_{constellation_name}")
    os.makedirs(record_folder, exist_ok=True)
    allocator_list = ["random", "mpc", "acc", "data"]

    for allocator in allocator_list:
        simulator = Simulator(input_path, allocator, cache_folder=record_folder)
        simulator.run()
        # Save results
        os.makedirs(record_folder, exist_ok=True)
        simulator.write_utilities_to_file(record_folder)
        simulator.write_downlink_records_to_file(record_folder)
        simulator.write_accuracy_records_to_file(record_folder)
        simulator.write_accumulated_utilities_to_file(record_folder)