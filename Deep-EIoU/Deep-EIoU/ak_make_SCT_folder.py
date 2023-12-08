import os
import shutil

def copy_results(src_directory, dest_directory, filename_suffix='_results.txt'):
    # Ensure the destination directory exists
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)

    # Walk through the source directory
    for root, dirs, files in os.walk(src_directory):

        for file in files:
            # Check if the current file ends with the specified suffix
            if file.endswith(filename_suffix):
                src_file_path = os.path.join(root, file)
                dest_file_path = os.path.join(dest_directory, file)

                # Copy the file to the destination directory
                shutil.copy2(src_file_path, dest_file_path)
                print(f"Copied {src_file_path} to {dest_file_path}")

# Set your source and destination directories
# dataset_type = 'test'
# src_directory = 'D:/classes/CS330/project/CS330_MOT/Deep-EIoU/Deep-EIoU/YOLOX_outputs/sportsmot-' + dataset_type
# dest_directory = 'D:/classes/CS330/project/CS330_MOT/Deep-EIoU/SCT/sportsmot-' + dataset_type

# print(src_directory)
# print(dest_directory)

# # Call the function with your specified directories
# copy_results(src_directory, dest_directory)
