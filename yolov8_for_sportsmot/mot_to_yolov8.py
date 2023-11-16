import os
import numpy as np
import cv2
from tqdm import tqdm
import shutil

# Define the paths
DATA_PATH = r"/Users/laviniap/Desktop/Autumn 2023 2024/a_CS 330 - Deep Multi-Task and Meta Learning/Project/Project Repository/CS330_MOT/data/sportsmot_publish/dataset"
OUT_PATH = os.path.join(DATA_PATH, "Sportsmot_dataset_yolov8")
SPLITS = ["train", "val", "test"]
SPLITS = ["test"]
HALF_VIDEO = False

# Function to convert SportsMot annotation to YOLO format
def convert_annotation(sportsmot_line, img_width, img_height):
    # Split the line
    elements = sportsmot_line.split(',')
    # Extract values
    bb_left, bb_top, bb_width, bb_height = map(float, elements[2:6])
    # Convert to YOLO format
    x_center = (bb_left + bb_width / 2) / img_width
    y_center = (bb_top + bb_height / 2) / img_height
    width = bb_width / img_width
    height = bb_height / img_height
    object_class = int(elements[1])
    return f"{object_class} {x_center} {y_center} {width} {height}"

# Loop over the splits
for split in SPLITS:
    data_path = os.path.join(DATA_PATH, split)
    out_path_images = os.path.join(OUT_PATH, f"{split}", "images")
    out_path_labels = os.path.join(OUT_PATH, f"{split}", "labels")
    os.makedirs(out_path_images, exist_ok=True)
    os.makedirs(out_path_labels, exist_ok=True)

    # Iterate through video sequences
    video_list = os.listdir(data_path)
    for seq in tqdm(sorted(video_list)):
        if ".DS_Store" in seq:
            continue
        seq_path = os.path.join(data_path, seq)
        img_path = os.path.join(seq_path, "img1")
        if split != "test":
            ann_path = os.path.join(seq_path, "gt/gt.txt")
            anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=",")
        images = os.listdir(img_path)
        num_images = len([image for image in images if "jpg" in image])

        # Decide the image range based on HALF_VIDEO setting
        if HALF_VIDEO and ("half" in split):
            image_range = [0, num_images // 2] if "train" in split else [num_images // 2 + 1, num_images - 1]
        else:
            image_range = [0, num_images - 1]

        # Process each image in the sequence
        for i in range(num_images):
            if i < image_range[0] or i > image_range[1]:
                continue
            img = cv2.imread(os.path.join(data_path, f"{seq}/img1/{i + 1:06d}.jpg"))
            img_file_copy_destination = os.path.join(out_path_images, f"{seq}_{i + 1:06d}.jpg")
            label_file_copy_destination = os.path.join(out_path_labels, f"{seq}_{i + 1:06d}.txt")
            shutil.copyfile(os.path.join(data_path, f"{seq}/img1/{i + 1:06d}.jpg"), img_file_copy_destination)

            height, width = img.shape[:2]
            # Create YOLO annotation file for each image
            if split != "test":
                frame_annotations = [ann for ann in anns if int(ann[0]) == i + 1]
                with open(label_file_copy_destination, 'w') as label_file:
                    for ann in frame_annotations:
                        string_array = ','.join(map(str, ann.astype(int)))
                        yolov8_line = convert_annotation(string_array, width, height)
                        label_file.write(yolov8_line + '\n')
