"""Utilities for scoring the model."""
import torch
import cv2
import numpy as np
import torch.nn.functional as F


def calculate_accuracy(logits, labels, ONE_ZERO=False):
    # Convert logits to probabilities
    if ONE_ZERO:
        logits[logits[:,:] > 0] = 1
        logits[logits[:,:] < 0] = 0
        predicted_classes = logits
    else:
        probabilities = torch.softmax(logits, dim=-1)
        predicted_classes = torch.argsort(probabilities, dim=-1, descending=True)

    correct_count = 0
    total_valid_predictions = 0

    # Iterate over each frame and sport
    for i in range(labels.shape[0]):  # Loop over frames
        for j in range(labels.shape[1]):  # Loop over sports
            if not ONE_ZERO:
                valid_labels = labels[i, j, labels[i, j] != 0]
            else:
                valid_labels = labels

            # Get the top-N predictions where N is the number of valid labels
            top_n_predictions = predicted_classes[i, j, :len(valid_labels)]
            top_n_predictions, _ = torch.sort(top_n_predictions)
            # Count correct predictions
            for prediction in top_n_predictions:
                if prediction in valid_labels:
                    correct_count += 1


            total_valid_predictions += len(valid_labels)

    # Calculate accuracy
    accuracy = correct_count / total_valid_predictions if total_valid_predictions > 0 else 0.0

    return accuracy

def calculate_accuracy_and_f1(logits, labels, f1, acc, ONE_ZERO=False):
    # Convert logits to probabilities
    probabilities = torch.softmax(logits, dim=-1)
    predicted_classes = torch.argsort(probabilities, dim=-1, descending=True)

    correct_count = 0
    total_valid_predictions = 0

    # Iterate over each frame and sport
    for i in range(labels.shape[0]):  # Loop over frames
        for j in range(labels.shape[1]):  # Loop over sports
            valid_labels = labels[i, j, labels[i, j] != 0]

            # Get the top-N predictions where N is the number of valid labels
            top_n_predictions = predicted_classes[i, j, :len(valid_labels)]
            top_n_predictions, _ = torch.sort(top_n_predictions)
            # Count correct predictions
            for prediction in top_n_predictions:
                if prediction in valid_labels:
                    correct_count += 1
            f1.update(top_n_predictions, valid_labels)
            acc.update(top_n_predictions, valid_labels)


            total_valid_predictions += len(valid_labels)

    # Calculate accuracy
    accuracy = correct_count / total_valid_predictions if total_valid_predictions > 0 else 0.0

    return accuracy, f1, acc

# def calculate_accuracy_and_f1(logits, labels, acc_class, f1_class):
#     for i in range(labels.shape[0]):  # Loop over frames
#         for j in range(labels.shape[1]):  # Loop over sports








# def calculate_accuracy(logits, labels):
#     # Convert logits to probabilities
#     probabilities = torch.softmax(logits, dim=-1)
#     # Get the class with the highest probability for each frame
#     predicted_classes = torch.argmax(probabilities, dim=-1)

#     # Count how many predictions match the labels
#     correct_count = torch.sum(predicted_classes == labels).item()

#     # Calculate the total number of frames
#     total_frames = labels.size(0)

#     # Calculate accuracy
#     accuracy = correct_count / total_frames if total_frames > 0 else 0.0

#     return accuracy



def calculate_iou(predicted_bboxes, true_bboxes):
    """
    Calculate the Intersection over Union (IoU) for bounding boxes.

    Args:
    - predicted_bboxes (Tensor): Predicted bounding boxes, shape [num_frames, num_players, 4]
    - true_bboxes (Tensor): True bounding boxes, shape [num_frames, num_players, 4]

    Returns:
    - float: Mean IoU over all bounding boxes
    """
    # Coordinates of the intersection boxes
    inter_left = torch.max(predicted_bboxes[..., 0], true_bboxes[..., 0])
    inter_top = torch.max(predicted_bboxes[..., 1], true_bboxes[..., 1])
    inter_right = torch.min(predicted_bboxes[..., 0] + predicted_bboxes[..., 2], true_bboxes[..., 0] + true_bboxes[..., 2])
    inter_bottom = torch.min(predicted_bboxes[..., 1] + predicted_bboxes[..., 3], true_bboxes[..., 1] + true_bboxes[..., 3])

    # Intersection area
    inter_area = torch.clamp(inter_right - inter_left, min=0) * torch.clamp(inter_bottom - inter_top, min=0)

    # Union area
    pred_area = predicted_bboxes[..., 2] * predicted_bboxes[..., 3]
    true_area = true_bboxes[..., 2] * true_bboxes[..., 3]
    union_area = pred_area + true_area - inter_area

    iou = inter_area / torch.clamp(union_area, min=1e-6)

    # Handle cases where IoU is NaN
    iou[torch.isnan(iou)] = 0

    return iou.mean().item()


