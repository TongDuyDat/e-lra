import os
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from models import  *
import albumentations as A
from albumentations.pytorch import ToTensorV2

def calculate_metrics(ground_truth, predicted, epsilon=1e-6):
    """
    Tính toán các chỉ số Dice, IoU, Recall, Precision, Accuracy, và F2 bằng PyTorch.
    
    Args:
        ground_truth (torch.Tensor): Mask ground truth với shape [1, 256, 256] (binary: 0 hoặc 1)
        predicted (torch.Tensor): Mask predict với shape [1, 256, 256] (binary: 0 hoặc 1)
        epsilon (float): Hằng số nhỏ để tránh chia cho 0
    
    Returns:
        dict: Từ điển chứa các chỉ số Dice, IoU, Recall, Precision, Accuracy, F2
    """
    # Đảm bảo ground truth và predicted có cùng kích thước
    assert ground_truth.shape == predicted.shape, "Ground truth và predicted phải có cùng kích thước"
    assert ground_truth.shape == (1, 256, 256), "Shape của tensor phải là [1, 256, 256]"

    # Chuyển thành tensor nhị phân (0 hoặc 1)
    ground_truth = (ground_truth > 0).float()
    predicted = (predicted > 0).float()
    
    # Tính TP, TN, FP, FN
    TP = torch.sum((ground_truth == 1) & (predicted == 1))
    TN = torch.sum((ground_truth == 0) & (predicted == 0))
    FP = torch.sum((ground_truth == 0) & (predicted == 1))
    FN = torch.sum((ground_truth == 1) & (predicted == 0))
    
    # Chuyển sang float để tính toán
    TP = TP.float()
    TN = TN.float()
    FP = FP.float()
    FN = FN.float()
    
    # Dice = 2 * (TP) / (2 * TP + FP + FN)
    dice = (2 * TP) / (2 * TP + FP + FN + epsilon)
    
    # IoU = TP / (TP + FP + FN)
    iou = TP / (TP + FP + FN + epsilon)
    
    # Recall = TP / (TP + FN)
    recall = TP / (TP + FN + epsilon)
    
    # Precision = TP / (TP + FP)
    precision = TP / (TP + FP + epsilon)
    
    # Accuracy = (TP + TN) / (TP + TN + FP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN + epsilon)
    
    # F2 = (5 * P * R) / (4 * P + R), với P là Precision, R là Recall
    f2 = (5 * precision * recall) / (4 * precision + recall + epsilon)
    
    # Trả về từ điển chứa các chỉ số
    metrics = {
        "Dice": dice.item(),
        "IoU": iou.item(),
        "Recall": recall.item(),
        "Precision": precision.item(),
        "Accuracy": accuracy.item(),
        "F2": f2.item()
    }
    
    return metrics

def visualize_predictions(
    image_mask_pairs,
    model,
    transform,
    threshold=0.5,
    figsize=(15, 3),
    show_titles=True
):
    """
    Visualize predictions from a GAN-based segmentation model and calculate metrics for each pair.

    Args:
        image_mask_pairs (list of tuples): List of (image_path, mask_path) pairs.
        model (torch.nn.Module): Trained segmentation model.
        transform (albumentations.Compose): Image transformation pipeline.
        threshold (float): Threshold for binarizing predicted mask.
        figsize (tuple): Figure size for matplotlib.
        show_titles (bool): Whether to show titles on subplots.
    """
    model.eval()

    # Kiểm tra xem có cặp nào chứa mask không
    has_ground_truth = any(mask_path is not None for _, mask_path in image_mask_pairs)
    num_rows = len(image_mask_pairs)
    num_cols = 3 if has_ground_truth else 2

    # Tạo figure với số lượng subplot phù hợp
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(figsize[0], figsize[1] * num_rows))
    if num_rows == 1:
        axes = axes.reshape(1, -1)

    for i, (image_path, mask_path) in enumerate(image_mask_pairs):
        # Đọc ảnh gốc
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Tiền xử lý ảnh
        transformed = transform(image=image)
        input_tensor = transformed["image"].unsqueeze(0).to(next(model.parameters()).device)
        input_tensor = input_tensor / 255.0

        # Dự đoán mask
        with torch.no_grad():
            pred_mask = model(input_tensor)
        pred_mask = pred_mask.cpu().numpy()[0, 0]  # Shape: [256, 256]
        pred_mask = (pred_mask > threshold).astype(np.uint8)  # Shape: [256, 256]

        # Hiển thị ảnh gốc
        axes[i, 0].imshow(image_rgb)
        if show_titles:
            axes[i, 0].set_title("Image")
        axes[i, 0].axis("off")

        col_idx = 1  # Chỉ số cột tiếp theo

        # Xử lý ground truth và tính toán metrics nếu có
        if has_ground_truth and mask_path is not None:
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask file not found: {mask_path}")
            mask_gt = cv2.imread(mask_path, 0)  # Đọc dưới dạng grayscale
            mask_gt = cv2.resize(mask_gt, (256, 256))  # Shape: [256, 256]
            mask_gt = (mask_gt > 128).astype(np.uint8)  # Shape: [256, 256]

            # Hiển thị ground truth
            axes[i, col_idx].imshow(mask_gt, cmap="gray")
            if show_titles:
                axes[i, col_idx].set_title("Ground Truth")
            axes[i, col_idx].axis("off")
            col_idx += 1

            # Chuyển ground truth và predicted thành tensor với shape [1, 256, 256]
            mask_gt_tensor = torch.from_numpy(mask_gt).float().unsqueeze(0)  # Shape: [1, 256, 256]
            pred_mask_tensor = torch.from_numpy(pred_mask).float().unsqueeze(0)  # Shape: [1, 256, 256]

            # Tính toán metrics
            metrics = calculate_metrics(mask_gt_tensor, pred_mask_tensor)

            # Tạo chuỗi metrics để hiển thị
            metrics_str = "\n".join([f"{key}: {value:.4f}" for key, value in metrics.items()])
            # Hiển thị metrics bên dưới hàng subplot
            axes[i, col_idx-1].text(
                0, -0.1, metrics_str, transform=axes[i, col_idx-1].transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8)
            )

        # Hiển thị mask dự đoán
        axes[i, col_idx].imshow(pred_mask, cmap="gray")
        if show_titles:
            axes[i, col_idx].set_title("Predicted Mask")
        axes[i, col_idx].axis("off")

    plt.tight_layout()
    plt.show()

# Khởi tạo mô hình và transform
model = GanModel(
    generator=Generator(input_shape=(3, 256, 256)),
    discriminator=DiscriminatorWithConvCRF(4),
    model_name="GAN",
    version="1.0",
    description="GAN for image segmentation",
)
model.load_checkpoint("best_model.pth")
model.eval()

transform = A.Compose([
    A.Resize(height=256, width=256),
    ToTensorV2(),
])

# Danh sách cặp ảnh và mask
image_mask_pairs = [
    ("test/images/149.png", "test/masks/149.png"),
    ("test/images/208.png", "test/masks/208.png"),
    ("test/images/176.png", "test/masks/176.png"),
    ("test/images/3.png", "test/masks/3.png"),
]

# Gọi hàm
visualize_predictions(image_mask_pairs, model, transform)