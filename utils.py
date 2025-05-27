import torch
import math
import torch.nn.functional as F
from torchmetrics.segmentation import DiceScore, MeanIoU
from torchmetrics.classification import (
    MulticlassRecall,
    MulticlassPrecision,
    MulticlassAccuracy,
)


def check_loss_nan(loss, verbose=True):
    """
    Checks if the provided loss (PyTorch tensor or float) contains NaN values.

    Parameters:
    - loss: Union[float, torch.Tensor]
        The loss value or tensor to be checked for NaNs.
    - verbose: bool (default=True)
        If True, prints a warning message when a NaN is detected.

    Returns:
    - bool
        True if NaN is detected, False otherwise.
    """
    if isinstance(loss, torch.Tensor):
        if torch.isnan(loss).any():
            if verbose:
                print("[WARNING] Loss contains NaN values (PyTorch Tensor).")
            return True
        return False
    elif isinstance(loss, float):
        if math.isnan(loss):
            if verbose:
                print("[WARNING] Loss is NaN (float value).")
            return True
        return False
    else:
        raise TypeError(
            f"Unsupported loss type: {type(loss)}. Expected float or torch.Tensor"
        )


def check_target_range(target_tensor, name="target"):
    """
    Kiểm tra tensor target có chứa giá trị ngoài khoảng [0, 1] hay không.

    Args:
        target_tensor (torch.Tensor): Tensor cần kiểm tra.
        name (str): Tên tensor để hiển thị trong thông báo lỗi.

    Raises:
        ValueError: Nếu phát hiện giá trị nằm ngoài khoảng [0, 1].
    """
    if not isinstance(target_tensor, torch.Tensor):
        raise TypeError("Đầu vào phải là một torch.Tensor")

    min_val = target_tensor.min()
    max_val = target_tensor.max()

    if min_val < 0 or max_val > 1:
        return True
    else:
        return False


class SegmentationMetrics:
    def __init__(self, num_classes=2, device='cpu', include_background=True, iou_foreground_only=False, threshold=0.5):
        """
        Initialize segmentation metrics for binary segmentation.

        Args:
            num_classes (int): Number of classes (default: 2 for binary).
            device (str or torch.device): Device to run metrics on (default: 'cpu').
            include_background (bool): Whether to include background class in Dice score (default: True).
            iou_foreground_only (bool): If True, compute IoU only for foreground class (default: False).
            threshold (float): Threshold to convert sigmoid outputs to binary (default: 0.5).
        """
        if num_classes != 2:
            raise ValueError("This implementation is designed for binary segmentation (num_classes=2).")
        self.num_classes = num_classes
        self.device = torch.device(device)
        self.include_background = include_background
        self.iou_foreground_only = iou_foreground_only
        self.threshold = threshold

        # Initialize metrics for binary segmentation
        self.mean_iou = MeanIoU(num_classes=num_classes).to(self.device)
        self.recall = MulticlassRecall(num_classes=num_classes, average='micro').to(self.device)
        self.precision = MulticlassPrecision(num_classes=num_classes, average='micro').to(self.device)
        self.accuracy = MulticlassAccuracy(num_classes=num_classes, average='micro').to(self.device)
        self.dice = DiceScore(
            num_classes=num_classes,
            include_background=include_background,
            average='micro',
            input_format='index'
        ).to(self.device)

        # For custom IoU calculation
        self.intersection = 0
        self.union = 0
        self.total_pixels = 0

    def update(self, preds, targets):
        """
        Update metrics with new predictions and targets.

        Args:
            preds (torch.Tensor): Predicted probabilities after sigmoid with shape (bs, 1, w, h).
            targets (torch.Tensor): Ground truth labels with shape (bs, 1, w, h), (bs, num_classes, w, h), or (bs, w, h).
        """
        preds = preds.to(self.device)
        targets = targets.to(self.device)

        # Validate input shapes
        if preds.ndim != 4 or preds.shape[1] != 1:
            raise ValueError(f"Expected preds shape (bs, 1, w, h), got {preds.shape}")
        
        # Allow targets to have shape (bs, 1, w, h), (bs, num_classes, w, h), or (bs, w, h)
        if targets.ndim == 4:
            if targets.shape[1] == 1:
                targets = targets.squeeze(1)  # Shape: (bs, w, h)
            elif targets.shape[1] == self.num_classes:
                targets = torch.argmax(targets, dim=1)  # Shape: (bs, w, h)
            else:
                raise ValueError(f"Expected targets shape (bs, 1, w, h) or (bs, {self.num_classes}, w, h), got {targets.shape}")
        elif targets.ndim != 3 or targets.shape[1:] != preds.shape[2:]:
            raise ValueError(f"Expected targets shape (bs, w, h) with w, h matching preds, got {targets.shape}")

        # Convert predictions to binary class indices (0 or 1)
        preds = (preds > self.threshold).long().squeeze(1)  # Shape: (bs, w, h), values 0 or 1

        # Validate predictions are binary
        if preds.max() > 1 or preds.min() < 0:
            raise ValueError(f"Predictions must contain only 0 or 1 after thresholding, got range [{preds.min()}, {preds.max()}]")

        # Ensure targets are long type and binary
        if targets.dtype not in (torch.long, torch.int):
            targets = (targets > self.threshold).long() if targets.dtype.is_floating_point else targets.long()
        if targets.max() > 1 or targets.min() < 0:
            raise ValueError(f"Target values must be 0 or 1 for binary segmentation, got range [{targets.min()}, {targets.max()}]")

        # Update torchmetrics
        self.mean_iou.update(preds, targets)
        self.recall.update(preds, targets)
        self.precision.update(preds, targets)
        self.accuracy.update(preds, targets)
        self.dice.update(preds, targets)

        # Custom IoU calculation for foreground class
        if self.iou_foreground_only:
            foreground_preds = (preds == 1)
            foreground_targets = (targets == 1)
            intersection = (foreground_preds & foreground_targets).float().sum()
            union = (foreground_preds | foreground_targets).float().sum()
            self.intersection += intersection
            self.union += union
            self.total_pixels += preds.numel()

    def compute(self):
        """
        Compute all metrics.

        Returns:
            dict: Dictionary containing computed metrics.
        """
        metrics = {
            "recall": self.recall.compute().item(),
            "precision": self.precision.compute().item(),
            "accuracy": self.accuracy.compute().item(),
            "dice": self.dice.compute().item(),
        }

        # Compute IoU
        if self.iou_foreground_only:
            iou = 0.0 if self.union == 0 else self.intersection / (self.union + 1e-8)
            metrics["mean_iou"] = iou.item()
        else:
            metrics["mean_iou"] = self.mean_iou.compute().item()

        return metrics

    def reset(self):
        """
        Reset all metrics to their initial state.
        """
        self.mean_iou.reset()
        self.recall.reset()
        self.precision.reset()
        self.accuracy.reset()
        self.dice.reset()
        self.intersection = 0
        self.union = 0
        self.total_pixels = 0

# # Khởi tạo metrics
# metrics = SegmentationMetrics(num_classes=2, device="cuda", iou_foreground_only=True)

# # Ví dụ đầu vào
# bs, num_classes, w, h = 2, 2, 256, 256
# preds = torch.randn(bs, num_classes, w, h)  # Logits
# targets = torch.randint(0, 2, (bs, w, h))  # Ground truth nhị phân (class indices)
# # Hoặc targets one-hot: targets = torch.nn.functional.one_hot(targets, num_classes=2).permute(0, 3, 1, 2)

# # Cập nhật và tính metric
# print(targets, preds)
# metrics.update(preds, targets)
# results = metrics.compute()
# print(results)
