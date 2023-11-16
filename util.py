"""Utilities for scoring the model."""
import torch


def score(logits, labels):
    """Returns the mean accuracy of a model's predictions on a set of examples.

    Args:
        logits (torch.Tensor): model predicted logits
            shape (examples, classes)
        labels (torch.Tensor): classification labels from 0 to num_classes - 1
            shape (examples,)
    """
    try:
        assert logits.dim() == 2
        assert labels.dim() == 1
        assert logits.shape[0] == labels.shape[0]
    except AssertionError:
        raise ValueError(
            f"Input shapes are invalid: logits shape {logits.shape}, labels shape {labels.shape}"
        )

    y = torch.argmax(logits, dim=-1) == labels
    y = y.type(torch.float)
    return torch.mean(y).item()




def calculate_accuracy(logits, labels):
    # Convert logits to probabilities
    probabilities = torch.softmax(logits, dim=-1)
    predicted_classes = torch.argsort(probabilities, dim=-1, descending=True)

    correct_count = 0
    total_valid_predictions = 0

    # Iterate over each frame and sport
    for i in range(labels.shape[0]):  # Loop over frames
        for j in range(labels.shape[1]):  # Loop over sports
            valid_labels = labels[i, j, labels[i, j] != -1]

            # Get the top-N predictions where N is the number of valid labels
            top_n_predictions = predicted_classes[i, j, :len(valid_labels)]

            # Count correct predictions
            for prediction in top_n_predictions:
                if prediction in valid_labels:
                    correct_count += 1

            total_valid_predictions += len(valid_labels)

    # Calculate accuracy
    accuracy = correct_count / total_valid_predictions if total_valid_predictions > 0 else 0.0

    return accuracy





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
