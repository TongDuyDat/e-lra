import argparse
import csv
import os
import cv2
import numpy as np
import torch
from pathlib import Path
from matplotlib import pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.general import increment_path, check_file, colorstr, LOGGER, Profile
from utils.torch_utils import select_device

class SegmentationInference:
    """
    Enhanced segmentation inference class similar to YOLOv5 detection.
    Supports batch processing, multiple input sources, and comprehensive result saving.
    """
    
    def __init__(self, model_path, model_class=None, transform=None, device=None):
        """
        Initialize the segmentation inference engine.
        
        Args:
            model_path (str): Path to model checkpoint
            model_class (torch.nn.Module, optional): Model class
            transform (albumentations.Compose, optional): Image transform pipeline
            device (str, optional): Device to run inference on
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path, model_class)
        self.transform = transform if transform else self._default_transform()
        
    def _load_model(self, model_path, model_class):
        """Load model from checkpoint."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
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
        """Create default image transformation pipeline."""
        return A.Compose([
            A.Resize(height=256, width=256),
            ToTensorV2(),
        ])

    def calculate_metrics(self, ground_truth, predicted, epsilon=1e-6):
        """
        Calculate segmentation metrics (Dice, IoU, Recall, Precision, Accuracy, F2).
        
        Args:
            ground_truth (torch.Tensor): Ground truth mask [1, 256, 256]
            predicted (torch.Tensor): Predicted mask [1, 256, 256]
            epsilon (float): Small constant to avoid division by zero
            
        Returns:
            dict: Dictionary containing calculated metrics
        """
        assert ground_truth.shape == predicted.shape, "Ground truth and predicted must have same shape"
        assert ground_truth.shape == (1, 256, 256), "Tensor shape must be [1, 256, 256]"

        ground_truth = (ground_truth > 0).float()
        predicted = (predicted > 0).float()

        TP = torch.sum((ground_truth == 1) & (predicted == 1))
        TN = torch.sum((ground_truth == 0) & (predicted == 0))
        FP = torch.sum((ground_truth == 0) & (predicted == 1))
        FN = torch.sum((ground_truth == 1) & (predicted == 0))

        TP, TN, FP, FN = TP.float(), TN.float(), FP.float(), FN.float()

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

    def process_single_image(self, image_path, mask_path=None, threshold=0.5):
        """
        Process a single image for segmentation.
        
        Args:
            image_path (str): Path to input image
            mask_path (str, optional): Path to ground truth mask
            threshold (float): Threshold for binary mask prediction
            
        Returns:
            tuple: (original_image, predicted_mask, ground_truth_mask, metrics)
        """
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=image_rgb)
        input_tensor = transformed["image"].unsqueeze(0).to(self.device)
        input_tensor = input_tensor.float() / 255.0

        # Inference
        with torch.no_grad():
            pred_mask = self.model(input_tensor)
        
        pred_mask = pred_mask.cpu().numpy()[0, 0]  # Shape: [256, 256]
        pred_mask = (pred_mask > threshold).astype(np.uint8)

        # Load ground truth mask if provided
        mask_gt = None
        metrics = None
        if mask_path and os.path.exists(mask_path):
            mask_gt = cv2.imread(mask_path, 0)
            mask_gt = cv2.resize(mask_gt, (256, 256))
            mask_gt = (mask_gt > 128).astype(np.uint8)

            # Calculate metrics
            mask_gt_tensor = torch.from_numpy(mask_gt).float().unsqueeze(0)
            pred_mask_tensor = torch.from_numpy(pred_mask).float().unsqueeze(0)
            metrics = self.calculate_metrics(mask_gt_tensor, pred_mask_tensor)

        return image_rgb, pred_mask, mask_gt, metrics

def run_segmentation(
    weights,
    source,
    imgsz=(256, 256),
    threshold=0.5,
    device="",
    view_img=False,
    save_txt=False,
    save_csv=False,
    save_img=True,
    nosave=False,
    project="runs/segment",
    name="exp",
    exist_ok=False,
    show_metrics=True,
    save_overlay=True
):
    """
    Run segmentation inference on images, similar to YOLOv5 detection.
    
    Args:
        weights (str): Path to model weights
        source (str): Input source (image, directory, glob pattern)
        imgsz (tuple): Input image size
        threshold (float): Segmentation threshold
        device (str): Device to run on
        view_img (bool): Display results
        save_txt (bool): Save results to text files
        save_csv (bool): Save results to CSV
        save_img (bool): Save result images
        nosave (bool): Don't save anything
        project (str): Project directory
        name (str): Experiment name
        exist_ok (bool): Don't increment run number
        show_metrics (bool): Calculate and display metrics
        save_overlay (bool): Save overlay images
    """
    # Setup
    device = select_device(device)
    save_img = not nosave and save_img
    source = str(source)
    
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
    if save_overlay:
        (save_dir / 'overlays').mkdir(exist_ok=True)
    
    # Initialize model
    segmentation_model = SegmentationInference(
        model_path=weights,
        device=device
    )
    
    # Get image files
    source_path = Path(source)
    if source_path.is_file():
        image_files = [source_path]
    elif source_path.is_dir():
        image_files = list(source_path.glob('*.jpg')) + list(source_path.glob('*.png')) + list(source_path.glob('*.jpeg'))
    else:
        image_files = list(Path().glob(source))
    
    # CSV file setup
    csv_path = save_dir / 'predictions.csv'
    csv_headers = ['Image', 'Pixels_Segmented', 'Confidence_Mean', 'Dice', 'IoU', 'Recall', 'Precision', 'Accuracy', 'F2']
    
    def write_to_csv(data):
        """Write prediction data to CSV file."""
        file_exists = csv_path.exists()
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)
    
    # Process images
    dt = Profile(device=device)
    results_summary = []
    
    LOGGER.info(f"Processing {len(image_files)} images...")
    
    for i, img_path in enumerate(image_files):
        with dt:
            # Try to find corresponding mask file
            mask_path = None
            if show_metrics:
                mask_dir = img_path.parent / 'masks' if (img_path.parent / 'masks').exists() else img_path.parent.parent / 'masks'
                potential_mask = mask_dir / img_path.name if mask_dir.exists() else None
                if potential_mask and potential_mask.exists():
                    mask_path = str(potential_mask)
            
            # Process image
            try:
                image_rgb, pred_mask, mask_gt, metrics = segmentation_model.process_single_image(
                    str(img_path), mask_path, threshold
                )
                
                # Calculate statistics
                pixels_segmented = np.sum(pred_mask > 0)
                confidence_mean = np.mean(pred_mask) if np.any(pred_mask) else 0.0
                
                # Prepare result data
                result_data = {
                    'Image': img_path.name,
                    'Pixels_Segmented': pixels_segmented,
                    'Confidence_Mean': f"{confidence_mean:.4f}",
                    'Dice': f"{metrics['Dice']:.4f}" if metrics else 'N/A',
                    'IoU': f"{metrics['IoU']:.4f}" if metrics else 'N/A',
                    'Recall': f"{metrics['Recall']:.4f}" if metrics else 'N/A',
                    'Precision': f"{metrics['Precision']:.4f}" if metrics else 'N/A',
                    'Accuracy': f"{metrics['Accuracy']:.4f}" if metrics else 'N/A',
                    'F2': f"{metrics['F2']:.4f}" if metrics else 'N/A'
                }
                
                results_summary.append(result_data)
                
                # Save results
                if save_csv:
                    write_to_csv(result_data)
                
                if save_txt:
                    txt_path = save_dir / 'labels' / f"{img_path.stem}.txt"
                    with open(txt_path, 'w') as f:
                        f.write(f"segmentation {pixels_segmented} {confidence_mean:.4f}\n")
                        if metrics:
                            f.write(f"metrics dice:{metrics['Dice']:.4f} iou:{metrics['IoU']:.4f} ")
                            f.write(f"recall:{metrics['Recall']:.4f} precision:{metrics['Precision']:.4f} ")
                            f.write(f"accuracy:{metrics['Accuracy']:.4f} f2:{metrics['F2']:.4f}\n")
                
                if save_img:
                    # Save predicted mask
                    mask_save_path = save_dir / f"{img_path.stem}_mask.png"
                    cv2.imwrite(str(mask_save_path), pred_mask * 255)
                    
                    # Save overlay if requested
                    if save_overlay:
                        overlay = image_rgb.copy()
                        overlay[pred_mask > 0] = [255, 0, 0]  # Red overlay
                        blended = cv2.addWeighted(image_rgb, 0.7, overlay, 0.3, 0)
                        overlay_path = save_dir / 'overlays' / f"{img_path.stem}_overlay.png"
                        cv2.imwrite(str(overlay_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
                
                # Display results
                if view_img:
                    plt.figure(figsize=(15, 5))
                    
                    plt.subplot(1, 3, 1)
                    plt.imshow(image_rgb)
                    plt.title('Original Image')
                    plt.axis('off')
                    
                    if mask_gt is not None:
                        plt.subplot(1, 3, 2)
                        plt.imshow(mask_gt, cmap='gray')
                        plt.title('Ground Truth')
                        plt.axis('off')
                        
                        plt.subplot(1, 3, 3)
                        plt.imshow(pred_mask, cmap='gray')
                        plt.title('Prediction')
                        plt.axis('off')
                        
                        if metrics:
                            metrics_text = '\n'.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                            plt.figtext(0.02, 0.02, metrics_text, fontsize=10, 
                                      bbox=dict(facecolor='white', alpha=0.8))
                    else:
                        plt.subplot(1, 3, 2)
                        plt.imshow(pred_mask, cmap='gray')
                        plt.title('Prediction')
                        plt.axis('off')
                    
                    plt.tight_layout()
                    plt.show()
                
                # Log progress
                status = f"({i+1}/{len(image_files)}) {img_path.name}: {pixels_segmented} pixels segmented"
                if metrics:
                    status += f", Dice: {metrics['Dice']:.4f}, IoU: {metrics['IoU']:.4f}"
                LOGGER.info(status)
                
            except Exception as e:
                LOGGER.error(f"Error processing {img_path.name}: {str(e)}")
                continue
    
    # Print summary
    LOGGER.info(f"Inference completed in {dt.t:.3f}s")
    if save_img or save_txt or save_csv:
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")
    
    return results_summary


def parse_opt():
    """Parse command line arguments for segmentation inference."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='model weights path')
    parser.add_argument('--source', type=str, default='data/images', help='source directory or image')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[256], help='inference size h,w')
    parser.add_argument('--threshold', type=float, default=0.5, help='segmentation threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-csv', action='store_true', help='save results in CSV format')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--project', default='runs/segment', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--show-metrics', action='store_true', help='calculate metrics if ground truth available')
    parser.add_argument('--save-overlay', action='store_true', help='save overlay images')
    
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


def main(opt):
    """Main function to run segmentation inference."""
    run_segmentation(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)