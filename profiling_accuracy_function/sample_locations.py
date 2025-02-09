import os
import numpy as np
import shutil
from tqdm import tqdm

if __name__ == "__main__":
    # Set the random seed
    np.random.seed(0)
    # set number of bootstrap locations
    num_bootstrap_locations = 5
    # train val split ratio
    train_val_split_ratio = 0.8

    location_datasets_folder = 'location_datasets'

    output_boostrap_folder = 'location_bootstrap_datasets'
    output_evaluation_folder = 'location_evaluation_datasets'

    # Create the output folder
    os.makedirs(output_boostrap_folder, exist_ok=True)
    os.makedirs(output_evaluation_folder, exist_ok=True)

    # Get the list of files in the location datasets folder
    location_list = os.listdir(location_datasets_folder)

    # randomly select the bootstrap locations
    bootstrap_locations = np.random.choice(location_list, num_bootstrap_locations, replace=False)
    # get the evaluation locations
    evaluation_locations = [location for location in location_list if location not in bootstrap_locations]

    # make a copy of the bootstrap location folders
    for location in tqdm(bootstrap_locations):
        boostrap_location_folder = os.path.join(location_datasets_folder, location)
        output_location_folder = os.path.join(output_boostrap_folder, location)

        bootstrap_location_image_folder = os.path.join(boostrap_location_folder, 'images')
        bootstrap_location_mask_folder = os.path.join(boostrap_location_folder, 'masks')

        boostrap_location_image_filename_list = os.listdir(bootstrap_location_image_folder)
        
        # split the image files into train and val
        for image_filename in tqdm(boostrap_location_image_filename_list):
            image_filepath = os.path.join(bootstrap_location_image_folder, image_filename)
            mask_filepath = os.path.join(bootstrap_location_mask_folder, image_filename)
            if np.random.rand() < train_val_split_ratio:
                output_image_folder = os.path.join(output_location_folder, 'train', 'images')
                output_mask_folder = os.path.join(output_location_folder, 'train', 'masks')
            else:
                output_image_folder = os.path.join(output_location_folder, 'val', 'images')
                output_mask_folder = os.path.join(output_location_folder, 'val', 'masks')

            os.makedirs(output_image_folder, exist_ok=True)
            os.makedirs(output_mask_folder, exist_ok=True)

            shutil.copy(image_filepath, output_image_folder)
            shutil.copy(mask_filepath, output_mask_folder)
        

    # make a copy of the evaluation locations
    for location in tqdm(evaluation_locations):
        evaluation_location_folder = os.path.join(location_datasets_folder, location)
        output_location_folder = os.path.join(output_evaluation_folder, location)
        shutil.copytree(evaluation_location_folder, output_location_folder)

    print(f"Bootstrap locations: {bootstrap_locations}")
    print(f"Evaluation locations: {evaluation_locations}")
    
