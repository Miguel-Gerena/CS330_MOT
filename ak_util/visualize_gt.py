import cv2
import matplotlib.pyplot as plt
import os

# Fixed image dimensions
imWidth = 1280
imHeight = 720

# Base paths
train_folder = './dataset/train/'
base_output_folder = './gt_plots/'

# Set to both values to None if you want to run on entire dataset
max_videos = 2 
max_frames_per_video = 11 

# Function to parse the ground truth file
def parse_gt_file(gt_file_path):
    boxes = {}
    with open(gt_file_path, 'r') as f:
        for line in f:
            frame, obj_id, bb_left, bb_top, bb_width, bb_height, conf, x, y = map(int, line.strip().split(','))
            if frame not in boxes:
                boxes[frame] = []
            boxes[frame].append((bb_left, bb_top, bb_width, bb_height, obj_id))
    return boxes

# Function to draw bounding boxes on an image
def draw_boxes(image, boxes):
    for bb_left, bb_top, bb_width, bb_height, obj_id in boxes:
        x_min, y_min = bb_left, bb_top
        x_max, y_max = bb_left + bb_width, bb_top + bb_height
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(image, f'ID {obj_id}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image

# List all video folders in the train folder
video_folders = [os.path.join(train_folder, d) for d in os.listdir(train_folder) if os.path.isdir(os.path.join(train_folder, d))]

processed_videos = 0
for video_folder in video_folders:
    if max_videos is not None and processed_videos >= max_videos:
        break
    image_folder = os.path.join(video_folder, 'img1/')
    gt_file_path = os.path.join(video_folder, 'gt/gt.txt')

    # Check if the ground truth file and image folder exist
    if os.path.exists(image_folder) and os.path.isfile(gt_file_path):
        ground_truths = parse_gt_file(gt_file_path)

        # Extract video name and create output directory
        video_name = os.path.basename(video_folder)
        output_folder = os.path.join(base_output_folder, video_name)
        os.makedirs(output_folder, exist_ok=True)

        # Determine the range of frames to process
        if max_frames_per_video is None:
            frame_range = sorted(ground_truths.keys())
        else:
            frame_range = sorted(ground_truths.keys())[:max_frames_per_video]

        # Process the frames
        for frame_number in frame_range:
            frame_filename = f"{str(frame_number).zfill(6)}.jpg"
            frame_path = os.path.join(image_folder, frame_filename)

            frame = cv2.imread(frame_path)

            if frame is not None and frame_number in ground_truths:
                frame_with_boxes = draw_boxes(frame, ground_truths[frame_number])
                output_path = os.path.join(output_folder, f"frame_{frame_filename}")
                cv2.imwrite(output_path, frame_with_boxes)
            else:
                print(f"Frame {frame_path} could not be loaded or does not have ground truth data. Please check the file path and format.")

        print(f"Processed and saved frames for video {video_name} in {output_folder}.")

        processed_videos += 1

    else:
        print(f"Image folder or gt.txt file does not exist for video {video_folder}. Skipping...")

print("Completed processing based on the specified configuration.")

