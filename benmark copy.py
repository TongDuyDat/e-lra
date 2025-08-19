import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from data.processing_data_benmark import DataBenchmark
from loss_functions import SegmentationMetrics
from models.model import GanModel  # Import từ metrics.py
from models import Generator, DiscriminatorWithLRA  # Import các thành phần cần thiết tướ
def benchmark_model(
    model_path,
    dataset_configs,
    phase="val",
    batch_size=8,
    device=None,
    datasets=None,
    model_class=None,
    threshold=0.5,
    verbose=True,
    output_csv="benchmark_results.csv",
    plot_output="benchmark_plot.png",
    pr_output="pr_curves.png",
    roc_output="roc_curves.png"
):
    """
    Đánh giá hiệu suất mô hình phân đoạn trên nhiều bộ dữ liệu, lưu kết quả vào CSV và vẽ biểu đồ.
    Vẽ riêng PR curves và ROC curves cho bài toán segmentation.

    Args:
        model_path (str): Đường dẫn tới file checkpoint của mô hình.
        dataset_configs (list of dict): Danh sách các config dataset, mỗi config chứa:
            - 'config_path' (str): Đường dẫn tới file cấu hình dataset.
            - 'name' (str): Tên của bộ dữ liệu (dùng để hiển thị).
        phase (str): Giai đoạn dữ liệu ('train', 'val', hoặc 'test'). Mặc định là 'val'.
        batch_size (int): Kích thước batch cho DataLoader. Mặc định là 8.
        device (str): Thiết bị để chạy mô hình ('cuda' hoặc 'cpu'). Mặc định là 'cuda' nếu có GPU.
        datasets (list, optional): Danh sách các dataset đã tải sẵn. Nếu None, tải từ config.
        model_class (torch.nn.Module, optional): Lớp mô hình. Nếu None, sử dụng GanModel mặc định.
        threshold (float): Ngưỡng để nhị phân hóa nhãn và dự đoán. Mặc định là 0.5.
        verbose (bool): In kết quả chi tiết nếu True. Mặc định là True.
        output_csv (str): Đường dẫn file CSV để lưu kết quả. Mặc định là 'benchmark_results.csv'.
        plot_output (str): Đường dẫn file PNG để lưu biểu đồ metrics. Mặc định là 'benchmark_plot.png'.
        pr_output (str): Đường dẫn file PNG để lưu PR curves. Mặc định là 'pr_curves.png'.
        roc_output (str): Đường dẫn file PNG để lưu ROC curves. Mặc định là 'roc_curves.png'.

    Returns:
        dict: Từ điển chứa kết quả đánh giá cho mỗi bộ dữ liệu, với key là tên dataset
              và value là dict chứa các chỉ số (Dice, IoU, Recall, Precision, Accuracy, F2, PR_AUC, ROC_AUC).
    """
    # Xác định thiết bị
    device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)

    # Tải mô hình
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    if model_class is None:
        model = GanModel(
            generator=Generator(input_shape=(3, 256, 256)),
            discriminator=DiscriminatorWithLRA(4),
            model_name="GAN",
            version="1.0",
            description="GAN for image segmentation",
        )
    else:
        model = model_class

    model.load_checkpoint(model_path)
    model.eval()
    model.to(device)

    # Khởi tạo metrics từ metrics.py
    metrics = SegmentationMetrics(
        num_classes=2,
        device=device,
        include_background=True,
        iou_foreground_only=False,
        threshold=threshold
    )

    # Lưu kết quả và dữ liệu curve
    results_all = {}
    csv_data = []
    curve_data = {}  # Lưu precision, recall, fpr, tpr cho PR/ROC curves

    # Đánh giá trên từng bộ dữ liệu
    for idx, config in enumerate(dataset_configs):
        dataset_name = config.get('name', 'Unnamed Dataset')
        config_path = config['config_path']

        if not os.path.exists(config_path):
            print(f"Warning: Config file not found for {dataset_name}: {config_path}. Skipping...")
            continue

        # Tải dataset
        try:
            if datasets is not None and idx < len(datasets):
                dataset = datasets[idx]
                dataset.phase = phase
                print(f"Using pre-loaded dataset for {dataset_name} with phase {dataset.phase}")
            else:
                dataset = DataBenchmark(config_path=config_path, phase=phase)
                print(f"Loaded dataset from config for {dataset_name}")
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {str(e)}. Skipping...")
            continue

        # Reset metrics
        metrics.reset()
        all_preds = []
        all_targets = []

        print(f"Evaluating on {dataset_name}...")
        # Đánh giá trên dataset
        with torch.no_grad():
            for images, masks in tqdm(dataloader, desc=f"Evaluating on {dataset_name} ({phase})"):
                images = images.to(device).to(torch.float32)
                masks = masks.to(device).to(torch.float32)

                # Dự đoán từ mô hình
                preds = model(images)

                # Xử lý dự đoán cho segmentation
                if preds.shape[1] == 1:  # Binary segmentation: (bs, 1, h, w)
                    preds_flat = preds.squeeze(1)  # Shape: (bs, h, w)
                elif preds.shape[1] == 2:  # Multi-class: (bs, num_classes, h, w)
                    preds_flat = preds[:, 1, :, :]  # Lấy kênh foreground
                else:
                    raise ValueError(f"Unsupported preds shape for segmentation: {preds.shape}")

                # Áp dụng sigmoid nếu cần (đảm bảo xác suất liên tục)
                preds_flat = torch.sigmoid(preds_flat)  # Shape: (bs, h, w)

                # Xử lý nhãn (masks)
                if masks.ndim == 4 and masks.shape[1] == 1:  # Shape: (bs, 1, h, w)
                    masks_flat = masks.squeeze(1)  # Shape: (bs, h, w)
                elif masks.ndim == 3:  # Shape: (bs, h, w)
                    masks_flat = masks
                elif masks.ndim == 4 and masks.shape[1] == 2:  # Shape: (bs, num_classes, h, w)
                    masks_flat = torch.argmax(masks, dim=1).float()  # Shape: (bs, h, w)
                else:
                    raise ValueError(f"Unsupported masks shape for segmentation: {masks.shape}")

                # Nhị phân hóa nhãn nếu cần
                if masks_flat.dtype.is_floating_point:
                    masks_flat = (masks_flat > threshold).float()  # Đảm bảo nhãn là 0 hoặc 1

                # Kiểm tra nhãn chỉ chứa 0 hoặc 1
                unique_targets = torch.unique(masks_flat)
                if not torch.all(torch.isin(unique_targets, torch.tensor([0, 1], device=device))):
                    raise ValueError(f"Targets must be binary (0 or 1) for {dataset_name}, got: {unique_targets}")

                # Chuyển sang numpy và làm phẳng
                preds_flat = preds_flat.cpu().numpy().flatten()
                masks_flat = masks_flat.cpu().numpy().flatten()

                all_preds.extend(preds_flat)
                all_targets.extend(masks_flat)

                # Cập nhật metrics
                metrics.update(preds, masks)

        # Tính các metric từ SegmentationMetrics
        results = metrics.compute()

        # Tính PR và ROC curves
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        # Kiểm tra nhãn là nhị phân
        unique_targets_np = np.unique(all_targets)
        if not np.all(np.isin(unique_targets_np, [0, 1])):
            print(f"Error: Non-binary targets in {dataset_name}: {unique_targets_np}")
            results['pr_auc'] = 0.0
            results['roc_auc'] = 0.0
            curve_data[dataset_name] = {
                'precision': np.array([0]),
                'recall': np.array([0]),
                'fpr': np.array([0]),
                'tpr': np.array([0]),
                'pr_auc': 0.0,
                'roc_auc': 0.0
            }
        else:
            # Kiểm tra dự đoán là xác suất liên tục
            if np.all(np.isin(all_preds, [0, 1])):
                print(f"Warning: Predictions in {dataset_name} are binary, expected probabilities")
                results['pr_auc'] = 0.0
                results['roc_auc'] = 0.0
                curve_data[dataset_name] = {
                    'precision': np.array([0]),
                    'recall': np.array([0]),
                    'fpr': np.array([0]),
                    'tpr': np.array([0]),
                    'pr_auc': 0.0,
                    'roc_auc': 0.0
                }
            else:
                precision, recall, _ = precision_recall_curve(all_targets, all_preds)
                pr_auc = auc(recall, precision)
                fpr, tpr, _ = roc_curve(all_targets, all_preds)
                roc_auc = auc(fpr, tpr)

                # Lưu dữ liệu curve
                curve_data[dataset_name] = {
                    'precision': precision,
                    'recall': recall,
                    'fpr': fpr,
                    'tpr': tpr,
                    'pr_auc': pr_auc,
                    'roc_auc': roc_auc
                }

                results['pr_auc'] = pr_auc
                results['roc_auc'] = roc_auc

        results_all[dataset_name] = results

        # Thêm vào dữ liệu CSV
        csv_row = {'Dataset': dataset_name}
        csv_row.update({metric.capitalize(): value for metric, value in results.items()})
        csv_data.append(csv_row)

        # In kết quả nếu verbose
        if verbose:
            print(f"\nEvaluation results for {dataset_name} ({phase}):")
            for metric, value in results.items():
                print(f"{metric.capitalize()}: {value:.4f}")

    # Lưu kết quả vào file CSV
    if csv_data:
        df = pd.DataFrame(csv_data)
        df.to_csv(output_csv, index=False)
        if verbose:
            print(f"\nResults saved to {output_csv}")

    # Vẽ biểu đồ metrics
    if csv_data:
        plot_metrics(results_all, plot_output)
        if verbose:
            print(f"Metrics plot saved to {plot_output}")

    # Vẽ PR và ROC curves
    if curve_data:
        plot_pr_roc_curves_separate(curve_data, pr_output, roc_output)
        if verbose:
            print(f"PR curves saved to {pr_output}")
            print(f"ROC curves saved to {roc_output}")

    return results_all

def plot_metrics(results_all, output_path):
    """
    Vẽ biểu đồ cột so sánh các chỉ số giữa các dataset.
    """
    datasets = list(results_all.keys())
    metrics = ['dice', 'mean_iou', 'recall', 'precision', 'accuracy', 'f2', 'pr_auc', 'roc_auc']
    metric_labels = ['Dice', 'IoU', 'Recall', 'Precision', 'Accuracy', 'F2', 'PR AUC', 'ROC AUC']

    # Chuẩn bị dữ liệu cho biểu đồ
    data = {metric: [results_all[ds][metric] for ds in datasets] for metric in metrics}

    # Thiết lập biểu đồ
    fig, ax = plt.subplots(figsize=(14, 6))
    bar_width = 0.1
    index = range(len(datasets))

    # Vẽ các cột cho từng chỉ số
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax.bar(
            [x + bar_width * i for x in index],
            data[metric],
            bar_width,
            label=label,
            edgecolor='black'
        )

    # Tùy chỉnh biểu đồ
    ax.set_xlabel('Datasets')
    ax.set_ylabel('Score')
    ax.set_title('Comparison of Segmentation Metrics Across Datasets')
    ax.set_xticks([x + bar_width * 3.5 for x in index])
    ax.set_xticklabels(datasets, rotation=45)
    ax.legend()
    ax.set_ylim(0, 1)  # Các chỉ số nằm trong khoảng [0, 1]
    plt.tight_layout()

    # Lưu biểu đồ
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_pr_roc_curves_separate(curve_data, pr_output, roc_output):
    """
    Vẽ PR curves và ROC curves riêng biệt cho tất cả dataset trong bài toán segmentation.
    """
    datasets = list(curve_data.keys())

    # Vẽ PR curves
    plt.figure(figsize=(8, 6))
    for dataset_name in datasets:
        precision = curve_data[dataset_name]['precision']
        recall = curve_data[dataset_name]['recall']
        pr_auc = curve_data[dataset_name]['pr_auc']
        if len(precision) > 1:  # Chỉ vẽ nếu có dữ liệu hợp lệ
            plt.plot(recall, precision, label=f'{dataset_name} (AUC = {pr_auc:.2f})')
        else:
            print(f"Skipping PR curve for {dataset_name} due to invalid data")

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves for Segmentation')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(pr_output, dpi=300, bbox_inches='tight')
    plt.close()

    # Vẽ ROC curves
    plt.figure(figsize=(8, 6))
    for dataset_name in datasets:
        fpr = curve_data[dataset_name]['fpr']
        tpr = curve_data[dataset_name]['tpr']
        roc_auc = curve_data[dataset_name]['roc_auc']
        if len(fpr) > 1:  # Chỉ vẽ nếu có dữ liệu hợp lệ
            plt.plot(fpr, tpr, label=f'{dataset_name} (AUC = {roc_auc:.2f})')
        else:
            print(f"Skipping ROC curve for {dataset_name} due to invalid data")

    plt.plot([0, 1], [0, 1], 'k--')  # Đường chéo cho ROC
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Segmentation')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(roc_output, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Ví dụ sử dụng
    dataset_configs = [
        {
            'name': 'CVC_ClinicDB',
            'config_path': 'data/configs/CVC-ClinicDB.py'
        },
        {
            'name': 'CVC_300',
            'config_path': 'data/configs/CVC_300.py'
        },
        {
            'name': 'Kvasir_SEG',
            'config_path': 'data/configs/kvasir-seg.py'
        },
        {
            'name': 'PolypGen',
            'config_path': 'data/configs/PolypGen.py'
        },
        {
            'name': 'ETIS_LaribPolypDB',
            'config_path': 'data/configs/ETIS-LaribPolypDB.py'
        },
        {
            'name': 'kvasir-sessile',
            'config_path': 'data/configs/kvasir-sessile.py'
        },
        {
            "name": "CVC-ColonDB",
            "config_path": "data/configs/CVC-ColonDB.py"
        }
    ]

    results = benchmark_model(
        model_path="best_gan_model_LRA.pth",
        dataset_configs=dataset_configs,
        phase="val",
        batch_size=8,
        verbose=True
    )

    # In lại tất cả kết quả
    print("\nSummary of all results:")
    for dataset_name, metrics in results.items():
        print(f"\n{dataset_name}:")
        for metric, value in metrics.items():
            print(f"  {metric.capitalize()}: {value:.4f}")