import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import DataBenchmark
from loss_functions import SegmentationMetrics
from models import DiscriminatorWithConvCRF, DiscriminatorWithLRA, GanModel, Generator
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

def benchmark_model(
    model_path,
    dataset_configs,
    phase="val",
    batch_size=8,
    device=None,
    datasets = None,
    model_class=None,
    threshold=0.5,
    verbose=True,
    output_csv="benchmark_results.csv",
    plot_output="benchmark_plot.png"
):
    """
    Đánh giá hiệu suất mô hình phân đoạn trên nhiều bộ dữ liệu, lưu kết quả vào CSV và vẽ biểu đồ.

    Args:
        model_path (str): Đường dẫn tới file checkpoint của mô hình.
        dataset_configs (list of dict): Danh sách các config dataset, mỗi config chứa:
            - 'config_path' (str): Đường dẫn tới file cấu hình dataset.
            - 'name' (str): Tên của bộ dữ liệu (dùng để hiển thị).
        phase (str): Giai đoạn dữ liệu ('train', 'val', hoặc 'test'). Mặc định là 'val'.
        batch_size (int): Kích thước batch cho DataLoader. Mặc định là 8.
        device (str): Thiết bị để chạy mô hình ('cuda' hoặc 'cpu'). Mặc định là 'cuda' nếu có GPU.
        model_class (torch.nn.Module, optional): Lớp mô hình. Nếu None, sử dụng GanModel mặc định.
        threshold (float): Ngưỡng để nhị phân hóa đầu ra dự đoán. Mặc định là 0.5.
        verbose (bool): In kết quả chi tiết nếu True. Mặc định là True.
        output_csv (str): Đường dẫn file CSV để lưu kết quả. Mặc định là 'benchmark_results.csv'.
        plot_output (str): Đường dẫn file PNG để lưu biểu đồ. Mặc định là 'benchmark_plot.png'.

    Returns:
        dict: Từ điển chứa kết quả đánh giá cho mỗi bộ dữ liệu, với key là tên dataset
              và value là dict chứa các chỉ số (Dice, IoU, Recall, Precision, Accuracy, F2).
    """
    # Xác định thiết bị
    device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

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

    # Khởi tạo metrics
    metrics = SegmentationMetrics(
        num_classes=2,
        device=device,
        include_background=True,
        iou_foreground_only=False,
        threshold=threshold
    )

    # Lưu kết quả cho tất cả dataset
    results_all = {}
    csv_data = []

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
                if not os.path.exists(config_path):
                    print(f"Warning: Config file not found for {dataset_name}: {config_path}. Skipping...")
                    continue
                dataset = DataBenchmark(config_path=config_path, phase=phase)
                print(f"Loaded dataset from config for {dataset_name}")
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {str(e)}. Skipping...")
            continue

        # Reset metrics
        metrics.reset()
        print(f"Evaluating on {dataset_name}...")
        # Đánh giá trên dataset
        with torch.no_grad():
            for images, masks in tqdm(dataloader, desc=f"Evaluating on {dataset_name} ({phase})"):
                images = images.to(device).to(torch.float32)
                masks = masks.to(device).to(torch.float32)
                preds = model(images)
                metrics.update(preds, masks)

        # Lưu kết quả
        results = metrics.compute()
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

    # Vẽ biểu đồ
    if csv_data:
        plot_metrics(results_all, plot_output)
        if verbose:
            print(f"Plot saved to {plot_output}")

    return results_all

def plot_metrics(results_all, output_path):
    """
    Vẽ biểu đồ cột so sánh các chỉ số giữa các dataset.

    Args:
        results_all (dict): Từ điển chứa kết quả đánh giá cho mỗi dataset.
        output_path (str): Đường dẫn để lưu file biểu đồ.
    """
    datasets = list(results_all.keys())
    metrics = ['dice', 'mean_iou', 'recall', 'precision', 'accuracy', 'f2']
    metric_labels = ['Dice', 'IoU', 'Recall', 'Precision', 'Accuracy', 'F2']

    # Chuẩn bị dữ liệu cho biểu đồ
    data = {metric: [results_all[ds][metric] for ds in datasets] for metric in metrics}

    # Thiết lập biểu đồ
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.15
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
    ax.set_xticks([x + bar_width * 2.5 for x in index])
    ax.set_xticklabels(datasets, rotation=45)
    ax.legend()
    ax.set_ylim(0, 1)  # Các chỉ số nằm trong khoảng [0, 1]
    plt.tight_layout()

    # Lưu biểu đồ
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# if __name__ == "__main__":
#     # Ví dụ sử dụng
#     dataset_configs = [
#         {
#             'name': 'CVC_ClinicDB',
#             'config_path': 'data/configs/CVC-ClinicDB.py'
#         },
#         {
#             'name': 'CVC_300',
#             'config_path': 'data/configs/CVC_300.py'
#         },
#         {
#             'name': 'Kvasir_SEG',
#             'config_path': 'data/configs/kvasir-seg.py'
#         },
#         {
#             'name': 'PolypGen',
#             'config_path': 'data/configs/PolypGen.py'
#         },
#         {
#             'name': 'ETIS_LaribPolypDB',
#             'config_path': 'data/configs/ETIS-LaribPolypDB.py'
#         },
#         {
#             'name': 'kvasir-sessile',
#             'config_path': 'data/configs/kvasir-sessile.py'
#         },
#         {
#             "name": "CVC-ColonDB",
#             "config_path": "data/configs/CVC-ColonDB.py"
#         }
#         # Thêm các dataset khác nếu có
#         # {
#         #     'name': 'Another_Dataset',
#         #     'config_path': 'data/configs/another_config.py'
#         # }
#     ]

#     results = benchmark_model(
#         model_path="best_gan_model_LRA.pth",
#         dataset_configs=dataset_configs,
#         phase="val",
#         batch_size=8,
#         verbose=True
#     )

#     # In lại tất cả kết quả
#     print("\nSummary of all results:")
#     for dataset_name, metrics in results.items():
#         print(f"\n{dataset_name}:")
#         for metric, value in metrics.items():
#             print(f"  {metric.capitalize()}: {value:.4f}")
def plot_pr_roc_curves(model, dataset, device='cpu', batch_size=32, save_path=None):
    """
    Vẽ PR curve và ROC curve cho model trên dataset.
    
    Args:
        model: Model PyTorch đã train.
        dataset: Dataset PyTorch chứa dữ liệu (images, labels).
        device (str): Thiết bị để chạy model ('cpu' hoặc 'cuda').
        batch_size (int): Kích thước batch cho DataLoader.
        save_path (str, optional): Đường dẫn để lưu figure, nếu None thì hiển thị.
    
    Returns:
        dict: Dictionary chứa AUC của PR và ROC curves.
    """
    # Chuyển model sang chế độ đánh giá
    model.eval()
    device = torch.device(device)
    model.to(device)
    
    # Tạo DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Thu thập tất cả dự đoán và nhãn
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            
            # Lấy dự đoán từ model (giả sử model trả về xác suất sigmoid)
            preds = model(images)
            if preds.shape[1] == 1:  # Binary segmentation
                preds = preds.squeeze(1)  # Shape: (bs, h, w)
            
            # Chuyển sang numpy và làm phẳng
            preds = preds.cpu().numpy().flatten()
            targets = targets.cpu().numpy().flatten()
            
            all_preds.extend(preds)
            all_targets.extend(targets)
    
    # Chuyển sang numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Tính Precision-Recall curve
    precision, recall, _ = precision_recall_curve(all_targets, all_preds)
    pr_auc = auc(recall, precision)
    
    # Tính ROC curve
    fpr, tpr, _ = roc_curve(all_targets, all_preds)
    roc_auc = auc(fpr, tpr)
    
    # Vẽ curves
    plt.figure(figsize=(12, 5))
    
    # PR Curve
    plt.subplot(1, 2, 1)
    plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True)
    
    # ROC Curve
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Đường chéo
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Lưu hoặc hiển thị
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    
    return {'pr_auc': pr_auc, 'roc_auc': roc_auc}

model = YourModel()
dataset = YourDataset()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Vẽ curves và lấy AUC
results = plot_pr_roc_curves(model, dataset, device=device, batch_size=32, save_path='curves.png')
print(f"PR AUC: {results['pr_auc']:.4f}")
print(f"ROC AUC: {results['roc_auc']:.4f}")