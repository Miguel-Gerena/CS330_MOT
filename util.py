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
    predicted_classes = torch.argmax(logits.squeeze(0), dim=-1)
    true_classes = labels.squeeze(0)
    accuracy = (predicted_classes == true_classes).float().mean().item()
    return accuracy