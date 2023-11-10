import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, sport_video_paths, num_support_videos, num_query_videos, num_support, num_query, frame_selection_strategy):
        self.support_set, self.query_set = create_support_query_split(sport_video_paths, num_support_videos, num_query_videos, num_support, num_query, frame_selection_strategy)
        self.support, self.query = combine_support_query_sets(self.support_set, self.query_set)
        self.support_frames_tensor, self.support_labels_tensor, self.query_frames_tensor, self.query_labels_tensor = convert_support_query_to_tensors(self.support, self.query, 'cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.support_frames_tensor)

    def __getitem__(self, idx):
        support_frames = self.support_frames_tensor[idx]
        support_labels = self.support_labels_tensor[idx]
        query_frames = self.query_frames_tensor[idx]
        query_labels = self.query_labels_tensor[idx]
        return support_frames, support_labels, query_frames, query_labels
    


#### Parses the ground truth file to be a dictionary for a given video ####
def parse_ground_truth(gt_path):
    with open(gt_path, 'r') as file:
        annotations = [line.strip().split(',') for line in file.readlines()]
    # Extract relevant information, ignoring 'x' and 'y' values.
    parsed_annotations = [
        {'frame_id': int(parts[0]), 'player_id': int(parts[1]), 'bbox': tuple(map(int, parts[2:6]))}
        for parts in annotations
    ]
    return parsed_annotations

def random_frame_selection(all_frames, num_support):
    return random.sample(all_frames, num_support)

def consecutive_frame_selection(all_frames, num_support, exclude_frames=[]):
    frames_set = set(exclude_frames)
    available_frames = [frame for frame in all_frames if frame not in frames_set]
    if num_support > len(available_frames):
        raise ValueError("Not enough available frames to select the requested number of consecutive frames.")
    start_index = random.randint(0, len(available_frames) - num_support)
    return available_frames[start_index:start_index + num_support]

def create_support_query_split(sport_video_paths, num_support_videos, num_query_videos, k_support, k_query, frame_selection_strategy):
    support_set = {}
    query_set = {}

    for sport, videos in sport_video_paths.items():
        video_ids = list(videos.keys())
        random.shuffle(video_ids)  # Shuffle to ensure random selection

        # Select two videos for the query set, ensuring one is different from the support set
        if len(video_ids) < num_support_videos + num_query_videos:
            raise ValueError(f"Not enough videos for sport {sport} to create a query set.")
        support_set_ids = video_ids[:num_support_videos]
        query_set_ids = video_ids[num_support_videos: num_support_videos + num_query_videos]

        support_data = [get_data_for_video(video_id, videos[video_id], k_support, frame_selection_strategy) for video_id in support_set_ids]
        support_set[sport] = support_data

        exclude_frames = set(os.path.basename(frame) for data in support_data for frame in data['frames'])

        if callable(frame_selection_strategy):
            def modified_selection_strategy(all_frames, k):
                return frame_selection_strategy([frame for frame in all_frames if os.path.basename(frame) not in exclude_frames], k)
            query_selection_strategy = modified_selection_strategy
        else:
            query_selection_strategy = frame_selection_strategy

        same_video_for_query = support_set_ids
        different_video_for_query = query_set_ids

        query_data_same = [get_data_for_video(video_id, videos[video_id], k_query, query_selection_strategy) for video_id in same_video_for_query]
        query_data_different = [get_data_for_video(video_id, videos[video_id], k_query, query_selection_strategy) for video_id in different_video_for_query]
        query_set[sport] = {
            'same_video': query_data_same,
            'different_video': query_data_different
        }

    return support_set, query_set

def get_data_for_video(video_id, video_path, num_support, frame_selection_function):
    frame_directory = os.path.join(video_path, 'img1')
    all_frames = sorted([frame for frame in os.listdir(frame_directory) if frame.endswith('.jpg')])
    gt_path = os.path.join(video_path, 'gt', 'gt.txt')
    ground_truths = parse_ground_truth(gt_path)

    selected_frames = frame_selection_function(all_frames, num_support)

    selected_ground_truths = {}
    for frame in selected_frames:
        frame_number = int(frame[:-4])
        ground_truth_entry = next((gt for gt in ground_truths if gt['frame_id'] == frame_number), None)
        if ground_truth_entry:
            selected_ground_truths[frame] = ground_truth_entry
        else:
            print(f"Warning: Frame {frame_number} is not in ground truths.")

    frame_paths = {frame: os.path.join(frame_directory, frame) for frame in selected_frames}
    return {'frames': frame_paths, 'annotations': selected_ground_truths}

def combine_support_query_sets(support_sets, query_sets):
    combined_support_set = {'frames': [], 'annotations': []}
    combined_query_set = {'frames': [], 'annotations': []}

    for sport, batches in support_sets.items():
        for batch in batches:
            combined_support_set['frames'].extend([(frame_id, frame) for frame_id, frame in batch['frames'].items()])
            combined_support_set['annotations'].extend([annotation for _, annotation in batch['annotations'].items()])

    for sport, query in query_sets.items():
        for data in query['same_video'] + query['different_video']:
            combined_query_set['frames'].extend([(frame_id, frame) for frame_id, frame in data['frames'].items()])
            combined_query_set['annotations'].extend([annotation for _, annotation in data['annotations'].items()])

    return combined_support_set, combined_query_set


#### Load the image, ensuring that it is in RGB format ####
def load_image(path):
    with Image.open(path) as img:
        img = img.convert('RGB')
        image_array = np.array(img)
        # Transpose the array to have the channel dimension first if necessary
        image_array = image_array.transpose((2, 0, 1))
    return image_array


def convert_annotations_to_features(annotations):
    features = []
    for anno in annotations:
        frame_id = anno['frame_id']
        player_id = anno['player_id']
        bbox = anno['bbox']
        features.append([frame_id, player_id] + list(bbox))
    return np.array(features, dtype=np.float32)

def convert_to_tensors(data_dict, device):
    frames = np.stack([load_image(path) for _, path in data_dict['frames']])
    frames_tensor = torch.tensor(frames, dtype=torch.float32).to(device)
    annotations_features = convert_annotations_to_features(data_dict['annotations'])
    annotations_tensor = torch.tensor(annotations_features, dtype=torch.float32).to(device)
    return frames_tensor, annotations_tensor

def convert_support_query_to_tensors(support_set, query_set, device):
    support_frames_tensor, support_annotations_tensor = convert_to_tensors(support_set, device)
    query_frames_tensor, query_annotations_tensor = convert_to_tensors(query_set, device)
    return support_frames_tensor, support_annotations_tensor, query_frames_tensor, query_annotations_tensor


