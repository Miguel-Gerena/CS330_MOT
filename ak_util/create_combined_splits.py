import json
import os
import shutil


train_json_path = './data_cs/combined_counts/train_counts.json'
val_json_path = './data_cs/combined_counts/val_counts.json'

combined_dataset_path = './dataset/combined'

new_dataset_path = './data_cs/train'



# Function to extract video IDs from a JSON file
def extract_video_ids(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
        return list(data.keys())  # Assuming video IDs are keys

# Function to copy folders based on video IDs
def copy_video_folders(video_ids, src_path, dest_path):
    for video_id in video_ids:
        src_folder = os.path.join(src_path, video_id)
        dest_folder = os.path.join(dest_path, video_id)
        if os.path.exists(src_folder):
            shutil.copytree(src_folder, dest_folder)

# Extract video IDs from JSON files
train_video_ids = extract_video_ids(train_json_path)
# val_video_ids = extract_video_ids(val_json_path)

# Create new dataset directory if it doesn't exist
if not os.path.exists(new_dataset_path):
    os.makedirs(new_dataset_path)

# Copy video folders from combined dataset to new dataset
copy_video_folders(train_video_ids, combined_dataset_path, new_dataset_path)
# copy_video_folders(val_video_ids, combined_dataset_path, new_dataset_path)