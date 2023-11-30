import os
import shutil

def copy_directory(dataset_type):
    # Define the source and destination directories
    source_dir = f'C:/Users/akayl/Desktop/CS330_MOT/Deep-EIoU/interpolation/sportsmot-{dataset_type}'
    destination_dir = 'C:/Users/akayl/Desktop/CS330_MOT/TrackEval/data/res'

    # Check if the destination directory exists, if not, create it
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Copy the directory
    shutil.copytree(source_dir, os.path.join(destination_dir, 'sportsmot-' + dataset_type), dirs_exist_ok=True)

# Example usage
dataset_type = 'train'
copy_directory(dataset_type)
