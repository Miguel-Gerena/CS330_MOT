import os
import random
import shutil
import sys
sys.path.append('./ak_util/')
import load_by_sport


def copy_files_recursively(source_dir, target_dir):
    for item in os.listdir(source_dir):
        source_item = os.path.join(source_dir, item)
        target_item = os.path.join(target_dir, item)

        if os.path.isdir(source_item):
            # If it's a directory, create a corresponding directory in the target and copy its contents
            os.makedirs(target_item, exist_ok=True)
            copy_files_recursively(source_item, target_item)
        else:
            # If it's a file, copy it
            shutil.copy2(source_item, target_item)


def count_folders(directory):
    return len([name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))])

def check_for_overlap(sport, test_set, train_set, val_set):
    # Convert lists to sets for faster operation
    test_set_ids = set(test_set)
    train_set_ids = set(train_set)
    val_set_ids = set(val_set)
    
    # Check for intersection between sets
    if (test_set_ids & train_set_ids) or (test_set_ids & val_set_ids) or (train_set_ids & val_set_ids):
        print(f"Overlap found in {sport} videos.")
    else:
        print(f"No overlap in {sport} videos.")

def verify_video_ids_location(video_id_list, target_folder, folder_name):
    missing_video_ids = []
    for video_id in video_id_list:
        expected_path = os.path.join(target_folder, video_id)
        # Check if the expected folder for the video ID exists in the target directory
        if not os.path.isdir(expected_path):
            missing_video_ids.append(video_id)

    if missing_video_ids:
        print(f"\nMissing video IDs in the {folder_name} folder: {missing_video_ids}")
    else:
        print(f"\nAll video IDs are correctly placed in the {folder_name} folder.")

# Function to read video IDs from a file and verify against a list
def verify_video_ids_in_file(video_id_list, file_path):
    # Read the video IDs from the file
    with open(file_path, 'r') as file:
        file_video_ids = [line.strip() for line in file]

    # Convert both lists to sets and compare
    set_from_list = set(video_id_list)
    set_from_file = set(file_video_ids)

    if set_from_list == set_from_file:
        print(f"All video IDs in {file_path} are correct.")
    else:
        missing_ids = list(set_from_list - set_from_file)
        additional_ids = list(set_from_file - set_from_list)
        if missing_ids:
            print(f"Missing video IDs in {file_path}: {missing_ids}")
        if additional_ids:
            print(f"Additional video IDs in {file_path} not in the original list: {additional_ids}")



# Define the source directories and the target directory
dataset_path = './data_cs/sportsmot_publish/dataset/'
train_path = os.path.join(dataset_path, 'train/')
val_path = os.path.join(dataset_path, 'val/')
new_folder_path =  f'{dataset_path}/combined_train_val'
# new_folder_path = os.path.join(dataset_path, 'combined_train_val')

# txt_dataset_path = './splits_txt/'
# train_txt_path = os.path.join(txt_dataset_path, 'og_train.txt')
# val_txt_path = os.path.join(txt_dataset_path, 'og_val.txt')
# txt_newfolder_path = os.path.join(txt_dataset_path, 'combined_train_val_txt')

##### Creates a new directory with the given files copied in there #####
# Create the new folder if it doesn't exist
os.makedirs(new_folder_path, exist_ok=True)

# Copy files from train and val to the new folder
copy_files_recursively(train_path, new_folder_path)
copy_files_recursively(val_path, new_folder_path)

print("Total number of folders immediately in the combined folder:", count_folders(new_folder_path))


# ##### Creates a new directory with the given files copied in there for txt folder #####
# merged_txt_path = os.path.join(txt_newfolder_path, 'combined_train_val.txt')
# # Merge the contents of train.txt and val.txt into a new file
# with open(merged_txt_path, 'w') as outfile:
#     # First, read and write the contents of train.txt
#     with open(train_txt_path, 'r') as infile:
#         outfile.write(infile.read())
#         outfile.write("\n")  # Optionally add a newline between the contents of the files

#     # Next, read and write the contents of val.txt
#     with open(val_txt_path, 'r') as infile:
#         outfile.write(infile.read())

# # Confirm the merge
# print(f"The files train.txt and val.txt have been merged into {merged_txt_path}")

# # Count the number of rows in the merged file
# with open(merged_txt_path, 'r') as file:
#     row_count = sum(1 for row in file)
# print(f"The merged file contains {row_count} rows.")



# sports_videos = load_by_sport.get_sport_split_dict('./train_val_combined_counts.json')
# test_count = 20
# train_count = 5
# val_count = 5

# # Dictionaries to hold the test, train, and val video IDs for each sport
# test_videos = {sport: [] for sport in sports_videos}
# train_videos = {sport: [] for sport in sports_videos}
# val_videos = {sport: [] for sport in sports_videos}

# random.seed(0)
# # Function to split video IDs into test, train, val lists for each sport
# def split_videos(sport, video_list):
#     random.shuffle(video_list)  # Shuffle the list to ensure randomness
#     test_videos[sport] = video_list[:test_count]
#     train_videos[sport] = video_list[test_count:test_count + train_count]
#     val_videos[sport] = video_list[test_count + train_count:test_count + train_count + val_count]

# # Apply the function to each sport
# for sport, videos in sports_videos.items():
#     split_videos(sport, videos)

# for sport in sports_videos:
#     print(f"Sport: {sport}")
#     print(f"Test Videos ({len(test_videos[sport])}): {test_videos[sport]}")
#     print(f"Train Videos ({len(train_videos[sport])}): {train_videos[sport]}")
#     print(f"Val Videos ({len(val_videos[sport])}): {val_videos[sport]}")
#     print() 

# # for sport in sports_videos:
# #     check_for_overlap(sport, test_videos[sport], train_videos[sport], val_videos[sport])

# ##################### next combine the sports so you have 1 train/val/test split, 
# actual_train_txt_path = './splits_txt/train.txt'
# actual_val_txt_path = './splits_txt/val.txt'
# actual_test_txt_path = './splits_txt/test.txt'
# # Initialize empty lists for combined video IDs
# combined_test_videos = []
# combined_train_videos = []
# combined_val_videos = []

# # Iterate over the dictionaries and extend the combined lists
# for sport in sports_videos:
#     combined_test_videos.extend(test_videos[sport])
#     combined_train_videos.extend(train_videos[sport])
#     combined_val_videos.extend(val_videos[sport])

# # # Print the combined lists and their lengths
# print(f"Combined Test List ({len(combined_test_videos)}): {combined_test_videos}")
# print(f"Combined Train List ({len(combined_train_videos)}): {combined_train_videos}")
# print(f"Combined Val List ({len(combined_val_videos)}): {combined_val_videos}")

# def write_ids_to_file(video_ids, file_path):
#     # Ensure the directory exists before writing the file
#     os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
#     with open(file_path, 'w') as file:
#         # Write each video ID to the file on a new line
#         for video_id in video_ids:
#             file.write(video_id + '\n')

# # Write the video IDs to their respective files
# write_ids_to_file(combined_train_videos, actual_train_txt_path)
# write_ids_to_file(combined_val_videos, actual_val_txt_path)
# write_ids_to_file(combined_test_videos, actual_test_txt_path)

# # verify_video_ids_in_file(combined_train_videos, actual_train_txt_path)
# # verify_video_ids_in_file(combined_val_videos, actual_val_txt_path)
# # verify_video_ids_in_file(combined_test_videos, actual_test_txt_path)


# # # then get the data corresponding to each video id and put into acutal train/val/test files
# actual_train_folder_path = './dataset/train/'
# actual_val_folder_path = './dataset/val/'
# actual_test_folder_path = './dataset/test/'
# combined_train_val_dataset_path = './dataset/combined_train_val/'

# os.makedirs(actual_train_folder_path, exist_ok=True)
# os.makedirs(actual_val_folder_path, exist_ok=True)
# os.makedirs(actual_test_folder_path, exist_ok=True)

# def copy_video_folders(video_id_list, target_folder):
#     # Loop thru each combined list checking the dataset/combined_train_val folder
#     for video_id in video_id_list:
#         source_folder = os.path.join(combined_train_val_dataset_path, video_id)
#         target_folder_path = os.path.join(target_folder, video_id)
        
#         # Check if the source folder exists
#         if os.path.isdir(source_folder):
#             # Move the folder
#             shutil.copytree(source_folder, target_folder_path)
#         else:
#             print(f"Folder for video ID {video_id} not found in combined_train_val.")

# copy_video_folders(combined_test_videos, actual_test_folder_path)
# copy_video_folders(combined_train_videos, actual_train_folder_path)
# copy_video_folders(combined_val_videos, actual_val_folder_path)

# # verify_video_ids_location(combined_test_videos, actual_test_folder_path, 'test')
# # verify_video_ids_location(combined_train_videos, actual_train_folder_path, 'train')
# # verify_video_ids_location(combined_val_videos, actual_val_folder_path, 'val')

# # print("Total number of folders immediately in the train folder:", count_folders(actual_train_folder_path))
# # print("Total number of folders immediately in the val folder:", count_folders(actual_val_folder_path))
# # print("Total number of folders immediately in the test folder:", count_folders(actual_test_folder_path))