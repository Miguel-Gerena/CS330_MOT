import numpy as np
import os
import random
import torch
from torch.utils.data import IterableDataset
import time
import imageio
import json
from collections import defaultdict
import shutil

def move_datasets(data_set):
    with open(f"{data_set}_counts_by_sport.json", 'r') as f:
        data = json.load(f)
    train_folders = [value for value in data.values()]
    train = train_folders[0] + train_folders[1] + train_folders[2] 

def get_jsons_by_sport(count_folder= "./data/combined_counts/", files=["test_counts.json", "train_counts.json", "val_counts.json"]):
    """
    This will create files to store the video ids by sport in json format.
    Args:
        count_folder: folder where the counts are stored
        files: files to decompose by sport

    """
    for file in files:
        file = count_folder + file
        dataset = defaultdict(list)

        with open(f'{file}', 'r') as f:
            data = json.load(f)

        for video_key, values in data.items():
            for key, value in values.items():
                if value == 1:
                    dataset[key].append(video_key)

        with open(f'{count_folder}{file.split(".")[0]}_by_sport.json', 'w') as file:
            json.dump(dataset, file, indent=2)


class DataGenerator(IterableDataset):
    def __init__(
        self,
        num_videos,
        frames_per_video,
        batch_type,
        number_of_sports=3,
        config={},
        cache=False,
        generate_new_tasks=False
    ):
        """
        Args:
            num_videos: number of videos to load per sport
            frames_per_video: num samples to generate per class in one batch (K+1)
            batch_type: train/val/test
            number_of_sports: Number of sports for classification 
            config: data_folder - folder where the data is located
                    img_size - size of the input images
            cache: whether to cache the images loaded
        """
        # This order can be shuffled to create new tasks
        
        self.frames_per_video = frames_per_video
        self.number_of_sports = number_of_sports

        self.data_folder = config.get("data_folder", f"./data/combined_train_val/")
        self.img_size = config.get("img_size", (1280, 720, 3))

        with open(f"./data/combined_counts/{batch_type}_counts_by_sport.json", 'r') as f:
            self.videoID_by_sport = json.load(f)

        self.dim_input = np.prod(self.img_size)
        self.dim_output = frames_per_video
        self.num_videos = num_videos 
        self.image_caching = cache
        self.generate_new_tasks = generate_new_tasks
        self.stored_images = {}

        # to track what was the last frame that was sampled for each sport
        self.last_sample = defaultdict(int)


    def image_file_to_array(self, filename, id, dim_input):
        """
        Takes an image path and returns numpy array
        Args:
            filename: Image filename
            id: last shot provided for the dataset
            dim_input: Flattened shape of image
        Returns:
            1 channel image
        """
        id += 1
        image_number = str(id)
        image_number = "0" * (6 - len(image_number)) + image_number
        filename = filename + f"/img1/{image_number}.jpg"
        if self.image_caching and (filename in self.stored_images):
            return self.stored_images[filename]
        image = imageio.imread(filename)  # misc.imread(filename)
        image = image.reshape([dim_input])
        image = image.astype(np.float32) / image.max()
        image = 1.0 - image
        if self.image_caching:
            self.stored_images[filename] = image
        return image, id

    def _sample(self):
        """
        Samples a batch for training, validation, or testing
        Returns:
            A tuple of (1) Image batch and (2) Label batch:
                1. image batch has shape [K+1, num_videos, num_sports, rgb_image_size] and is a numpy array
                2. label batch has shape [K+1, num_videos, num_sports, num_sports] and is a numpy array
            where K is the number of "shots", N is number of classes
        """
        if self.generate_new_tasks:
            randomize = np.array(["Basketball", "Football","Volleyball"])
            np.random.shuffle(randomize)
            self.sports_order = {randomize[i]:i  for i in range(len(randomize))}

        samples = defaultdict(list)
        for key, value in self.videoID_by_sport.items():
            samples[key] = random.sample(value, self.num_videos)
        

        images = np.ones((self.frames_per_video, self.num_videos, self.number_of_sports, self.dim_input), np.float32)
        labels = np.ones((self.frames_per_video, self.num_videos, self.number_of_sports, self.number_of_sports), np.float32)
        for key, video_id in samples.items():
            for i in range(len(video_id)):
                for k in range(self.frames_per_video):

                        images[k % self.frames_per_video][i][self.sports_order[key]], self.last_sample[key] = \
                            self.image_file_to_array(self.data_folder + video_id[i], self.last_sample[key], self.dim_input)
                        labels[k % self.frames_per_video][i][self.sports_order[key]] = np.eye(self.number_of_sports)[self.sports_order[key]] 
        
        # Step 4: Shuffle the order of examples from the query set
        # randomize = np.arange(self.num_classes)
        # np.random.shuffle(randomize)
        # images[-1] = images[-1][randomize]
        # labels[-1] = labels[-1][randomize]

        return (images, labels, self.sports_order)


    def __iter__(self):
        while True:
            yield self._sample()
