import os
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SegmentationVisualizer:
    """
    Class để visualize predictions từ mô hình phân đoạn ảnh dựa trên GAN và tính toán các metrics.

    Args:
        model_path (str): Đường dẫn tới file checkpoint của mô hình.
        model_class (torch.nn.Module, optional): Lớp mô hình (mặc định là GanModel).
        transform (albumentations.Compose, optional): Pipeline biến đổi ảnh. Nếu None, sử dụng mặc định.
        device (str, optional): Thiết bị để chạy mô hình ('cuda' hoặc 'cpu'). Mặc định là 'cuda' nếu có GPU.
    """
    def __init__(self, model_path, model_class=None, transform=None, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path, model_class)
        self.transform = transform if transform else self._default_transform()
        
    def _load_model(self, model_path, model_class):
        """
        Tải mô hình từ checkpoint.

        Args:
            model_path (str): Đường dẫn tới file checkpoint.
            model_class (torch.nn.Module, optional): Lớp mô hình. Nếu None, sử dụng GanModel mặc định.

        Returns:
            torch.nn.Module: Mô hình đã được tải và chuyển sang chế độ eval.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        # Mặc định sử dụng GanModel nếu không cung cấp model_class
        if model_class is None:
            from models import GanModel, Generator, DiscriminatorWithLRA
            model = GanModel(
                generator=Generator(input_shape=(3, 256, 256)),
                discriminator=DiscriminatorWithLRA(4),
                model_name="GAN",
                version="1.0",
                description="GAN for image segmentation",
            )
        else:
            model = model_class()
        
        model.load_checkpoint(model_path)
        model.eval()
        model.to(self.device)
        return model

    def _default_transform(self):
        """
        Tạo pipeline biến đổi ảnh mặc định.

        Returns:
            albumentations.Compose: Pipeline biến đổi ảnh.
        """
        return A.Compose([
            A.Resize(height=256, width=256),
            ToTensorV2(),
        ])

    def calculate_metrics(self, ground_truth, predicted, epsilon=1e-6):
        """
        Tính toán các chỉ số Dice, IoU, Recall, Precision, Accuracy, và F2 bằng PyTorch.

        Args:
            ground_truth (torch.Tensor): Mask ground truth với shape [1, 256, 256] (binary: 0 hoặc 1)
            predicted (torch.Tensor): Mask predict với shape [1, 256, 256] (binary: 0 hoặc 1)
            epsilon (float): Hằng số nhỏ để tránh chia cho 0

        Returns:
            dict: Từ điển chứa các chỉ số Dice, IoU, Recall, Precision, Accuracy, F2
        """
        assert ground_truth.shape == predicted.shape, "Ground truth và predicted phải có cùng kích thước"
        assert ground_truth.shape == (1, 256, 256), "Shape của tensor phải là [1, 256, 256]"

        ground_truth = (ground_truth > 0).float()
        predicted = (predicted > 0).float()

        TP = torch.sum((ground_truth == 1) & (predicted == 1))
        TN = torch.sum((ground_truth == 0) & (predicted == 0))
        FP = torch.sum((ground_truth == 0) & (predicted == 1))
        FN = torch.sum((ground_truth == 1) & (predicted == 0))

        TP = TP.float()
        TN = TN.float()
        FP = FP.float()
        FN = FN.float()

        dice = (2 * TP) / (2 * TP + FP + FN + epsilon)
        iou = TP / (TP + FP + FN + epsilon)
        recall = TP / (TP + FN + epsilon)
        precision = TP / (TP + FP + epsilon)
        accuracy = (TP + TN) / (TP + TN + FP + FN + epsilon)
        f2 = (5 * precision * recall) / (4 * precision + recall + epsilon)

        return {
            "Dice": dice.item(),
            "IoU": iou.item(),
            "Recall": recall.item(),
            "Precision": precision.item(),
            "Accuracy": accuracy.item(),
            "F2": f2.item()
        }

    def visualize_predictions(
        self,
        image_mask_pairs,
        threshold=0.5,
        figsize=(15, 3),
        show_titles=True
    ):
        """
        Visualize predictions từ mô hình phân đoạn và tính toán metrics cho mỗi cặp.

        Args:
            image_mask_pairs (list of tuples): List các cặp (image_path, mask_path).
            threshold (float): Ngưỡng để nhị phân hóa mask dự đoán.
            figsize (tuple): Kích thước figure cho matplotlib.
            show_titles (bool): Có hiển thị tiêu đề trên các subplot hay không.
        """
        self.model.eval()
        has_ground_truth = any(mask_path is not None for _, mask_path in image_mask_pairs)
        num_rows = len(image_mask_pairs)
        num_cols = 3 if has_ground_truth else 2

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(figsize[0], figsize[1] * num_rows))
        if num_rows == 1:
            axes = axes.reshape(1, -1)

        for i, (image_path, mask_path) in enumerate(image_mask_pairs):
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Image file not found: {image_path}")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            transformed = self.transform(image=image)
            input_tensor = transformed["image"].unsqueeze(0).to(self.device)
            input_tensor = input_tensor / 255.0

            with torch.no_grad():
                pred_mask = self.model(input_tensor)
            pred_mask = pred_mask.cpu().numpy()[0, 0]  # Shape: [256, 256]
            pred_mask = (pred_mask > threshold).astype(np.uint8)  # Shape: [256, 256]

            axes[i, 0].imshow(image_rgb)
            if show_titles:
                axes[i, 0].set_title("Image")
            axes[i, 0].axis("off")

            col_idx = 1

            if has_ground_truth and mask_path is not None:
                if not os.path.exists(mask_path):
                    raise FileNotFoundError(f"Mask file not found: {mask_path}")
                mask_gt = cv2.imread(mask_path, 0)
                mask_gt = cv2.resize(mask_gt, (256, 256))
                mask_gt = (mask_gt > 128).astype(np.uint8)

                axes[i, col_idx].imshow(mask_gt, cmap="gray")
                if show_titles:
                    axes[i, col_idx].set_title("Ground Truth")
                axes[i, col_idx].axis("off")
                col_idx += 1

                mask_gt_tensor = torch.from_numpy(mask_gt).float().unsqueeze(0)
                pred_mask_tensor = torch.from_numpy(pred_mask).float().unsqueeze(0)
                metrics = self.calculate_metrics(mask_gt_tensor, pred_mask_tensor)

                metrics_str = "\n".join([f"{key}: {value:.4f}" for key, value in metrics.items()])
                axes[i, col_idx-1].text(
                    0, -0.1, metrics_str, transform=axes[i, col_idx-1].transAxes,
                    fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8)
                )

            axes[i, col_idx].imshow(pred_mask, cmap="gray")
            if show_titles:
                axes[i, col_idx].set_title("Predicted Mask")
            axes[i, col_idx].axis("off")

        plt.tight_layout()
        plt.show()

# Ví dụ sử dụng
if __name__ == "__main__":
    # Khởi tạo visualizer
    visualizer = SegmentationVisualizer(
        model_path="D:/NCKH/NCKH2025/LGPS/logs/logs_20250605_175911_LRA/logs_20250605_175911/weights/best_gan_model.pth"
    )

    # Danh sách cặp ảnh và mask
    image_mask_pairs = [
        ("test/images/149.png", "test/masks/149.png"),
        ("test/images/208.png", "test/masks/208.png"),
        ("test/images/176.png", "test/masks/176.png"),
        ("test/images/3.png", "test/masks/3.png"),
    ]

    # Gọi phương thức visualize
    visualizer.visualize_predictions(image_mask_pairs)