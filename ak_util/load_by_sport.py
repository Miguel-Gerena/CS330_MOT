import json
import os
import random

# data_path= './train_counts.json'

def get_sport_split_dict(counts_json_path):
    # Load the data from JSON file
    with open(counts_json_path, 'r') as file:
        data = json.load(file)

    # Initialize a list to hold different sports videos
    basketball_videos = []
    football_videos = []
    volleyball_videos = []

    # Iterate over the items in the dictionary
    for video_id, sports_counts in data.items():
        # Check the sport the video is associated with
        if sports_counts["Basketball"] == 1:
            basketball_videos.append(video_id)
        elif sports_counts["Football"] == 1:
            football_videos.append(video_id)
        elif sports_counts["Volleyball"] == 1:
            volleyball_videos.append(video_id)
        else:
            print("ERROR: Video is not associated with a category")
    # return basketball_videos, football_videos, volleyball_videos
    sports_to_videos = {
    'basketball': basketball_videos,
    'football': football_videos,
    'volleyball': volleyball_videos
    }
    return sports_to_videos

# Gets the file paths for each of the video ids 
def find_sport_video_paths(actual_video_path, sports_to_videos):
    sport_video_paths = {}

    for sport, video_ids in sports_to_videos.items():
        video_paths = {}
        for video_id in video_ids:
            video_path = os.path.join(actual_video_path, video_id)
            if os.path.exists(video_path):
                video_paths[video_id] = video_path
            else:
                print(f"Warning: Video path for ID {video_id} in sport {sport} does not exist.")
        sport_video_paths[sport] = video_paths

    return sport_video_paths

# Gets the sports dictionary and adds each videos file path to the dictionary
def get_videos_file_path_dict(counts_json_path, actual_video_path):
    # Get the lists of video ids for each sport
    sports_list_dict = get_sport_split_dict(counts_json_path)
    # Create a dictionary with the video paths and cooresponding video ids
    sports_dict = find_sport_video_paths(actual_video_path, sports_list_dict)

    return sports_dict

# Parses the ground truth file to be a dictionary for a given video
def parse_ground_truth(gt_path):
    with open(gt_path, 'r') as file:
        annotations = [line.strip().split(',') for line in file.readlines()]
    # Extract relevant information, ignoring 'x' and 'y' values.
    parsed_annotations = [
        {'frame_id': int(parts[0]), 'player_id': int(parts[1]), 'bbox': tuple(map(int, parts[2:6]))}
        for parts in annotations
    ]
    return parsed_annotations

# Assigns the frames randomly
def random_frame_selection(all_frames, num_support):
    return random.sample(all_frames, num_support)

#Assigns the frames in a consecutive sequence
def consecutive_frame_selection(all_frames, num_support, exclude_frames=[]):
    frames_set = set(exclude_frames)
    available_frames = [frame for frame in all_frames if frame not in frames_set]
    if num_support > len(available_frames):
        raise ValueError("Not enough available frames to select the requested number of consecutive frames.")
    start_index = random.randint(0, len(available_frames) - num_support)
    return available_frames[start_index:start_index + num_support]

# Creates the support/query splits
def create_support_query_split(sport_video_paths, num_support, num_query, frame_selection_strategy):
    support_set = {}
    query_set = {}

    for sport, videos in sport_video_paths.items():
        video_ids = list(videos.keys())
        random.shuffle(video_ids)  # Shuffle to ensure random selection

        # Select two videos for the query set, ensuring one is different from the support set
        if len(video_ids) < 2:
            raise ValueError(f"Not enough videos for sport {sport} to create a query set.")
        
        # Select one random video for the support set
        support_video_id = video_ids.pop()
        print(f"\n\n{sport}, video id {support_video_id} selected for support set\nSupport set:")
        
        support_set[sport] = get_data_for_video(support_video_id, videos[support_video_id], num_support, frame_selection_strategy)

        same_video_for_query = support_video_id  # This will provide frames not used in support set
        different_video_for_query = video_ids.pop()  # This is a different video for generalization

        if frame_selection_strategy == consecutive_frame_selection:
            # Define a modified strategy for the query set that excludes frames from the support set
            def modified_selection_strategy(all_frames, num_support):
                return consecutive_frame_selection(
                    all_frames,
                    num_support,
                    exclude_frames=[os.path.basename(path) for path in support_set[sport]['frames'].values()]
                )
            query_selection_strategy = modified_selection_strategy
        else:
            query_selection_strategy = frame_selection_strategy
        print("query set:")
        query_set[sport] = {
            'same_video': get_data_for_video(same_video_for_query, videos[same_video_for_query], num_query, query_selection_strategy),
            'different_video': get_data_for_video(different_video_for_query, videos[different_video_for_query], num_query, frame_selection_strategy)
        }
        # print(f"{sport}: Query set videos {query_set[sport]}")

    return support_set, query_set

def get_data_for_video(video_id, video_path, num_support, frame_selection_function):
    frame_directory = os.path.join(video_path, 'img1')
    # Get all frame filenames and ground truth annotations
    all_frames = sorted([frame for frame in os.listdir(frame_directory) if frame.endswith('.jpg')])
    print(f"frames count for {video_id} is {len(all_frames)}")
    gt_path = os.path.join(video_path, 'gt', 'gt.txt')
    ground_truths = parse_ground_truth(gt_path) 

    selected_frames = frame_selection_function(all_frames, num_support)
    
    # Extract the corresponding ground truth for the selected frames
    selected_ground_truths = {}
    for frame in selected_frames:
        frame_number = int(frame[:-4])  # Convert filename to frame number
        # Find the ground truth entry for the current frame number
        ground_truth_entry = next((gt for gt in ground_truths if gt['frame_id'] == frame_number), None)
        if ground_truth_entry:
            # Found the matching ground truth data
            selected_ground_truths[frame] = ground_truth_entry
        else:
            # The frame is NOT in ground truths, handle appropriately
            print(f"Warning: Frame {frame_number} is not in ground truths.")

    # print(f"selected ground truths for {video_id} are {selected_ground_truths}")
   
    # Create a mapping from selected frame filenames to their full path
    frame_paths = {frame: os.path.join(frame_directory, frame) for frame in selected_frames}
    # print(f"Frame paths for {video_id} are {frame_paths}")
    return {'frames': frame_paths, 'annotations': selected_ground_truths}

