import os
import shutil
import json

def copy_video_to_test(video_id, combined_train_val_folder, dataset_folder):
    # Check if the video_id folder exists in combined_train_val
    source_folder = os.path.join(combined_train_val_folder, video_id)
    if os.path.exists(source_folder):
        
        destination_folder = os.path.join(dataset_folder, video_id)
        
        # Copy the folder from combined_train_val to dataset folder
        shutil.copytree(source_folder, destination_folder)
        # print(f"Video {video_id} copied to test dataset.")
    else:
        print(f"Video {video_id} not found in combined_train_val.")

def copy_videos_from_json(json_file, combined_train_val_folder, dataset_folder):

    with open(json_file, 'r') as f:
        test_counts_data = json.load(f)

    # Loop through the video IDs in the JSON file
    for video_id, counts in test_counts_data.items():
        copy_video_to_test(video_id, combined_train_val_folder, dataset_folder)

    print("Copying completed.")

# Example usage:
test_counts_file = './test_counts.json'
train_counts_file = './train_counts.json'
val_counts_file = './val_counts.json'
combined_train_val_folder = './combined_train_val'
test_dataset_folder = './test'
train_dataset_folder = './train'
val_dataset_folder = './val'

copy_videos_from_json(test_counts_file, combined_train_val_folder, test_dataset_folder)
copy_videos_from_json(train_counts_file, combined_train_val_folder, train_dataset_folder)
copy_videos_from_json(val_counts_file, combined_train_val_folder, val_dataset_folder)
