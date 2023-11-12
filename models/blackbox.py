import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np


def initialize_weights(model):
    if type(model) in [nn.Linear]:
        nn.init.xavier_uniform_(model.weight)
        nn.init.zeros_(model.bias)
    elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
        nn.init.orthogonal_(model.weight_hh_l0)
        nn.init.xavier_uniform_(model.weight_ih_l0)
        nn.init.zeros_(model.bias_hh_l0)
        nn.init.zeros_(model.bias_ih_l0)


class Black(nn.Module):
    def __init__(self, num_players, frames_per_video_support, frames_per_video_query , hidden_dim, num_sports=3, img_w=1280, img_h=720, color_channels=3):
        super(Black, self).__init__()
        self.num_players = num_players
        self.frames_per_video_support = frames_per_video_support 
        self.frames_per_video_query = frames_per_video_query
        self.num_sports = num_sports
        self.data_columns_in_gt = 6

        self.layer1 = torch.nn.RNN(img_w * img_h * color_channels + num_players * 6, hidden_dim, batch_first=True)  # the multiplaction is # of players * data columns
        self.layer2 = torch.nn.LSTM(hidden_dim, num_players * self.data_columns_in_gt, batch_first=True)
        initialize_weights(self.layer1)
        initialize_weights(self.layer2)
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

    def forward(self, input_images, input_labels):
        """
        MANN
        Args:
            input_images: [B, K+1, N, 784] flattened images
            labels: [B, K+1, N, N] ground truth labels
        Returns:
            [B, K+1, N, N] predictions
        """
        #############################
        ### START CODE HERE ###

        # Step 1: Concatenate the full (support & query) set of labels and images
        input_labels_shape = list(input_labels.shape)
        correct_shape_for_labels = input_labels_shape[:-2] + [self.num_players * self.data_columns_in_gt]
        input_labels = torch.reshape(input_labels, correct_shape_for_labels)
        concatenated_sets = torch.concat((input_images, input_labels), -1)
        
        # Step 2: Zero out the labels from the concatenated corresponding to the query set
        input_image_shape = input_images.shape
        concatenated_sets[:, -self.frames_per_video_query ,:, :, input_image_shape[-self.frames_per_video_query ]:] = torch.zeros((self.num_players * self.data_columns_in_gt ))

        # Step 3: Pass the concatenated set sequentially to the memory-augmented network
        concatenated_sets = torch.reshape(concatenated_sets, (input_image_shape[0],  self.num_sports * (self.frames_per_video_support + self.frames_per_video_query), concatenated_sets.shape[-1]))
        calcs, _ = self.layer1(concatenated_sets)
        calcs, _ = self.layer2(calcs)
        # Step 3: Return the predictions with [B, K+1, N, N] shape
        return torch.reshape(calcs, (input_labels_shape))

        ### END CODE HERE ###

    def loss_function_id(self, preds, labels):
        """
        Computes MANN loss
        Args:
            preds: [B, K+1, N, N] network output
            labels: [B, K+1, N, N] labels
        Returns:
            scalar loss
        Note:
            Loss should only be calculated on the N test images
            Loss should be a scalar since mean reduction is used for cross entropy loss
            You would want to use F.cross_entropy here, specifically:
            with predicted unnormalized logits as input and ground truth class indices as target.
            Your logits would be of shape [B*N, N], and label indices would be of shape [B*N].
        """
        #############################

        loss = None

        ### START CODE HERE ###
        # Step 1: extract the predictions for the query set
        query_set = preds[:, -1, :, :]
        query_set = torch.reshape(query_set, (query_set.shape[0]*self.num_classes, self.num_classes))

        # Step 2: extract the true labels for the query set and reverse the one hot-encoding  
        true_labels_query_set = labels[:, -1] 
        true_labels_query_set = torch.argmax(true_labels_query_set, dim=-1)
        true_labels_query_set = torch.flatten(true_labels_query_set)

        # Step 3: compute the Cross Entropy Loss for the query set only!
        loss = F.cross_entropy(query_set, true_labels_query_set)
        ### END CODE HERE ###
        return loss
    
    def loss_function_bounding_box(self, preds, labels):
        """
        Computes MANN loss
        Args:
            preds: [B, K+1, N, N] network output
            labels: [B, K+1, N, N] labels
        Returns:
            scalar loss
        Note:
            Loss should only be calculated on the N test images
            Loss should be a scalar since mean reduction is used for cross entropy loss
            You would want to use F.cross_entropy here, specifically:
            with predicted unnormalized logits as input and ground truth class indices as target.
            Your logits would be of shape [B*N, N], and label indices would be of shape [B*N].
        """
        #############################

        loss = None

        ### START CODE HERE ###
        # Step 1: extract the predictions for the query set
        query_set = preds[:, -1, :, :]
        query_set = torch.reshape(query_set, (query_set.shape[0]*self.num_classes, self.num_classes))

        # Step 2: extract the true labels for the query set and reverse the one hot-encoding  
        true_labels_query_set = labels[:, -1] 
        true_labels_query_set = torch.argmax(true_labels_query_set, dim=-1)
        true_labels_query_set = torch.flatten(true_labels_query_set)

        # Step 3: compute the Cross Entropy Loss for the query set only!
        loss = F.cross_entropy(query_set, true_labels_query_set)
        ### END CODE HERE ###
        return loss
