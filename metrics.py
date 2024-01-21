import torch
from abc import ABC, abstractmethod

class SegmentationMetric(ABC):
    """Abstract segmentation metric class."""
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def __call__(self, predictions, targets):
        """Compute the metric."""
        pass

class PixelAccuracy(SegmentationMetric):
    """Pixel accuracy metric."""
    def __init__(self):
        super(PixelAccuracy, self).__init__("Pixel Accuracy")

    def __call__(self, predictions, targets):
        correct_pixels = (predictions == targets).sum().item()
        total_pixels = predictions.numel()
        accuracy = correct_pixels / total_pixels
        return accuracy

class IntersectionOverUnion(SegmentationMetric):
    """Intersection over Union (IoU) metric."""
    def __init__(self, smooth=1e-6):
        super(IntersectionOverUnion, self).__init__("IoU")
        self.smooth = smooth

    def __call__(self, predictions, targets):
        intersection = (predictions & targets).sum().item()
        union = (predictions | targets).sum().item()
        iou = (intersection + self.smooth) / (union + self.smooth)
        return iou

class DiceCoefficient(SegmentationMetric):
    """DICE Coefficient metric."""
    def __init__(self, smooth=1e-6):
        super(DiceCoefficient, self).__init__("DICE Coefficient")
        self.smooth = smooth

    def __call__(self, predictions, targets):
        intersection = (predictions & targets).sum().item()
        dice = (2 * intersection + self.smooth) / (predictions.sum().item() + targets.sum().item() + self.smooth)
        return dice
