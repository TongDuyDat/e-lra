import torch

# Định nghĩa các class metric tùy chỉnh
class BaseMetric:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def _compute_confusion_matrix(self, pred, target):
        # Chuyển pred thành nhị phân bằng ngưỡng
        pred = (pred > self.threshold).float()
        target = target.float()

        # Tính TP, FP, TN, FN
        TP = (pred * target).sum()  # 1 và 1
        FP = (pred * (1 - target)).sum()  # 1 và 0
        TN = ((1 - pred) * (1 - target)).sum()  # 0 và 0
        FN = ((1 - pred) * target).sum()  # 0 và 1

        return TP, FP, TN, FN

class Dice(BaseMetric):
    def __call__(self, pred, target):
        TP, FP, TN, FN = self._compute_confusion_matrix(pred, target)
        dice = (2 * TP) / (2 * TP + FP + FN + 1e-8)
        return dice

class IoU(BaseMetric):
    def __call__(self, pred, target):
        TP, FP, TN, FN = self._compute_confusion_matrix(pred, target)
        iou = TP / (TP + FP + FN + 1e-8)
        return iou

class Recall(BaseMetric):
    def __call__(self, pred, target):
        TP, FP, TN, FN = self._compute_confusion_matrix(pred, target)
        recall = TP / (TP + FN + 1e-8)
        return recall

class Precision(BaseMetric):
    def __call__(self, pred, target):
        TP, FP, TN, FN = self._compute_confusion_matrix(pred, target)
        precision = TP / (TP + FP + 1e-8)
        return precision

class Accuracy(BaseMetric):
    def __call__(self, pred, target):
        TP, FP, TN, FN = self._compute_confusion_matrix(pred, target)
        accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
        return accuracy

class F2(BaseMetric):
    def __call__(self, pred, target):
        TP, FP, TN, FN = self._compute_confusion_matrix(pred, target)
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f2 = (5 * precision * recall) / (4 * precision + recall + 1e-8)
        return f2

# Tích hợp vào class SegmentationMetrics
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

        # Khởi tạo các metric tùy chỉnh
        self.dice_metric = Dice(threshold=self.threshold)
        self.iou_metric = IoU(threshold=self.threshold)
        self.recall_metric = Recall(threshold=self.threshold)
        self.precision_metric = Precision(threshold=self.threshold)
        self.accuracy_metric = Accuracy(threshold=self.threshold)
        self.f2_metric = F2(threshold=self.threshold)

        # Biến để lưu trữ giá trị tích lũy
        self.dice_values = []
        self.iou_values = []
        self.recall_values = []
        self.precision_values = []
        self.accuracy_values = []
        self.f2_values = []

        # For custom IoU calculation (foreground only)
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
        preds_binary = (preds > self.threshold).float().squeeze(1)  # Shape: (bs, w, h), values 0 or 1

        # Validate predictions are binary
        if preds_binary.max() > 1 or preds_binary.min() < 0:
            raise ValueError(f"Predictions must contain only 0 or 1 after thresholding, got range [{preds_binary.min()}, {preds_binary.max()}]")

        # Ensure targets are float type and binary
        if targets.dtype not in (torch.long, torch.int):
            targets = (targets > self.threshold).float() if targets.dtype.is_floating_point else targets.float()
        if targets.max() > 1 or targets.min() < 0:
            raise ValueError(f"Target values must be 0 or 1 for binary segmentation, got range [{targets.min()}, {targets.max()}]")

        # Tính toán metrics cho từng batch và lưu trữ
        batch_size = preds.shape[0]
        for i in range(batch_size):
            pred_i = preds_binary[i]  # Shape: (w, h)
            target_i = targets[i]  # Shape: (w, h)

            # Tính các metric và lưu vào danh sách
            self.dice_values.append(self.dice_metric(pred_i, target_i))
            self.iou_values.append(self.iou_metric(pred_i, target_i))
            self.recall_values.append(self.recall_metric(pred_i, target_i))
            self.precision_values.append(self.precision_metric(pred_i, target_i))
            self.accuracy_values.append(self.accuracy_metric(pred_i, target_i))
            self.f2_values.append(self.f2_metric(pred_i, target_i))

        # Custom IoU calculation for foreground class
        if self.iou_foreground_only:
            foreground_preds = (preds_binary == 1)
            foreground_targets = (targets == 1)
            intersection = (foreground_preds & foreground_targets).float().sum()
            union = (foreground_preds | foreground_targets).float().sum()
            self.intersection += intersection
            self.union += union
            self.total_pixels += preds_binary.numel()

    def compute(self):
        """
        Compute all metrics.

        Returns:
            dict: Dictionary containing computed metrics.
        """
        # Tính trung bình các giá trị metric
        metrics = {
            "dice": torch.tensor(self.dice_values).mean().item() if self.dice_values else 0.0,
            "recall": torch.tensor(self.recall_values).mean().item() if self.recall_values else 0.0,
            "precision": torch.tensor(self.precision_values).mean().item() if self.precision_values else 0.0,
            "accuracy": torch.tensor(self.accuracy_values).mean().item() if self.accuracy_values else 0.0,
            "f2": torch.tensor(self.f2_values).mean().item() if self.f2_values else 0.0,
        }

        # Compute IoU
        if self.iou_foreground_only:
            iou = 0.0 if self.union == 0 else self.intersection / (self.union + 1e-8)
            metrics["mean_iou"] = iou.item()
        else:
            metrics["mean_iou"] = torch.tensor(self.iou_values).mean().item() if self.iou_values else 0.0

        return metrics

    def reset(self):
        """
        Reset all metrics to their initial state.
        """
        self.dice_values = []
        self.iou_values = []
        self.recall_values = []
        self.precision_values = []
        self.accuracy_values = []
        self.f2_values = []
        self.intersection = 0
        self.union = 0
        self.total_pixels = 0