import os
import shutil

def copy_directory(experiment_name):
    # Define the source and destination directories
    source_dir = f'D:/classes/CS330/project/CS330_MOT/Deep-EIoU/interpolation/{experiment_name}'
    destination_dir = 'D:/classes/CS330/project/CS330_MOT/TrackEval/data/res'

    # Check if the destination directory exists, if not, create it
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Copy the directory
    shutil.copytree(source_dir, os.path.join(destination_dir, experiment_name), dirs_exist_ok=True)

# Example usage
# dataset_type = 'test'
# copy_directory(dataset_type)
