import importlib
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.processing_CVC import CVC_CliniCDBDataset
from loss_functions.loss import CombinedLoss
from models import GanModel, Generator, DiscriminatorWithConvCRF
from models.e_lra import DiscriminatorWithLRA
from utils import SegmentationMetrics, check_loss_nan
import logging
import csv
from datetime import datetime

class GANTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        config_path=None,
    ):
        self.load_config(config_path)
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self._setup_training()
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging and CSV writer for metrics"""
        # Create log directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(self.config.checkpoint_dir, f"logs_{timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)

        # Setup logger
        self.logger = logging.getLogger("GANTrainer")
        self.logger.setLevel(logging.INFO)
        log_file = os.path.join(self.log_dir, "training.log")
        file_handler = logging.FileHandler(log_file)
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

        # Setup CSV file for metrics
        self.csv_file = os.path.join(self.log_dir, "metrics.csv")
        self.csv_fields = [
            "epoch", "train_g_loss", "train_d_loss", "val_loss",
            "mean_iou", "recall", "precision", "accuracy", "dice"
        ]
        with open(self.csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fields)
            writer.writeheader()

    def train(self):
        """Main training loop"""
        self.model.to(self.config.device)
        self.logger.info("Starting training...")
        for epoch in range(self.config.num_epochs):
            train_g_loss, train_d_loss = self.train_one_epoch()
            val_loss, logs = self.validate()

            # Log to console and file
            log_message = (
                f"Epoch [{epoch + 1}/{self.config.num_epochs}] - "
                f"Train G Loss: {train_g_loss:.4f}, Train D Loss: {train_d_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Mean IoU: {logs['mean_iou']:.4f}, Dice: {logs['dice']:.4f}, "
                f"Recall: {logs['recall']:.4f}, Precision: {logs['precision']:.4f}, "
                f"Accuracy: {logs['accuracy']:.4f}"
            )
            self.logger.info(log_message)

            # Save metrics to CSV
            self._log_to_csv(epoch + 1, train_g_loss, train_d_loss, val_loss, logs)

            # Save best model
            is_best = False
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                is_best = True
            self.save_model(epoch=epoch, is_best=is_best)
            # Learning rate scheduling
            self.reduce_learning_rate(epoch)

    def _log_to_csv(self, epoch, train_g_loss, train_d_loss, val_loss, logs):
        """Save metrics to CSV file"""
        metrics = {
            "epoch": epoch,
            "train_g_loss": train_g_loss,
            "train_d_loss": train_d_loss,
            "val_loss": val_loss,
            "mean_iou": logs["mean_iou"],
            "recall": logs["recall"],
            "precision": logs["precision"],
            "accuracy": logs["accuracy"],
            "dice": logs["dice"],
        }
        with open(self.csv_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fields)
            writer.writerow(metrics)

    def train_one_epoch(self):
        """Training logic for one epoch"""
        self.model.train()
        total_g_loss = 0
        total_d_loss = 0

        with tqdm(self.train_loader, unit="batch") as tepoch:
            for data, targets in tepoch:
                data = data.to(self.config.device).to(torch.float32)
                targets = targets.to(self.config.device).to(torch.float32)
                # Train generator
                g_loss, fake_mask = self.train_generator(data, targets)
                if check_loss_nan(g_loss):
                    self.logger.error("NaN detected in generator loss. Stopping training.")
                    raise ValueError("NaN in generator loss")
                # Train discriminator
                d_loss = self.train_discriminator(data, fake_mask, targets)
                if check_loss_nan(d_loss):
                    self.logger.error("NaN detected in discriminator loss. Stopping training.")
                    raise ValueError("NaN in discriminator loss")

                total_d_loss += d_loss
                total_g_loss += g_loss
                tepoch.set_postfix(g_loss=g_loss, d_loss=d_loss)

        avg_g_loss = total_g_loss / len(self.train_loader)
        avg_d_loss = total_d_loss / len(self.train_loader)
        return avg_g_loss, avg_d_loss

    def train_discriminator(self, data, mask_fakes, targets):
        """Train discriminator one step"""
        self.optimizer_D.zero_grad()
        mask_fakes = mask_fakes.detach()
        real_output = self.model.discriminator(data, targets)
        fake_output = self.model.discriminator(data, mask_fakes)
        real_labels = torch.ones_like(real_output) * 0.9
        fake_labels = torch.zeros_like(fake_output)
        d_real_loss = self.discriminator_loss(real_output, real_labels)
        if check_loss_nan(d_real_loss):
            self.logger.error("NaN detected in d_real_loss")
            raise ValueError("NaN in d_real_loss")
        d_fake_loss = self.discriminator_loss(fake_output, fake_labels)
        if check_loss_nan(d_fake_loss):
            self.logger.error("NaN detected in d_fake_loss")
            raise ValueError("NaN in d_fake_loss")
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        self.optimizer_D.step()
        return d_loss.item()

    def train_generator(self, data, targets):
        """Train generator one step"""
        self.optimizer_G.zero_grad()
        fake_masks = self.model.generate(data)
        g_seg_loss = self.generator_loss(fake_masks, targets)
        if check_loss_nan(g_seg_loss):
            self.logger.error("NaN detected in g_seg_loss")
            raise ValueError("NaN in g_seg_loss")
        g_seg_loss.backward()
        self.optimizer_G.step()
        return g_seg_loss.item(), fake_masks

    @torch.no_grad()
    def validate(self):
        """Validation loop"""
        self.model.eval()
        total_val_loss = 0
        metrics = {
            "mean_iou": 0,
            "recall": 0,
            "precision": 0,
            "accuracy": 0,
            "dice": 0,
        }
        with tqdm(self.val_loader, desc="Validating", leave=False) as pbar:
            for data, targets in self.val_loader:
                data, targets = data.to(self.config.device), targets.to(self.config.device)
                data, targets = data.to(torch.float32), targets.to(torch.float32)
                outputs = self.model.generate(data)
                combined_loss = self.generator_loss(outputs, targets)
                if check_loss_nan(combined_loss):
                    self.logger.error("NaN detected in val_loss")
                    raise ValueError("NaN in val_loss")
                total_val_loss += combined_loss.item()
                metric = self.metrics.update(outputs, targets)
                metric = self.metrics.compute()
                for key in metrics:
                    if key in metric:
                        metrics[key] += metric[key]
                pbar.update(1)
                logs = {
                    "val_loss": combined_loss.item() / pbar.n,
                    "mean_iou": metrics["mean_iou"] / pbar.n,
                    "recall": metrics["recall"] / pbar.n,
                    "precision": metrics["precision"] / pbar.n,
                    "accuracy": metrics["accuracy"] / pbar.n,
                    "dice": metrics["dice"] / pbar.n,
                }
                pbar.set_postfix(logs)
        avg_val_loss = total_val_loss / len(self.val_loader)
        logs = {key: metrics[key] / len(self.val_loader) for key in metrics}
        logs["val_loss"] = avg_val_loss
        return avg_val_loss, logs

    def reduce_learning_rate(self, epoch):
        """Learning rate scheduling"""
        if epoch % 30 == 0 and epoch > 0:
            for param_group in self.optimizer_G.param_groups:
                param_group["lr"] *= 0.5
            for param_group in self.optimizer_D.param_groups:
                param_group["lr"] *= 0.5
            self.logger.info(f"Reduced learning rate to {param_group['lr']:.6f} at epoch {epoch}")

    def _setup_training(self):
        """Setup optimizers, schedulers, and loss functions"""
        self.best_val_loss = float("inf")
        self.optimizer_G = optim.Adam(
            self.model.generator.parameters(),
            lr=self.config.lr_generator,
            betas=(self.config.beta1, self.config.beta2),
        )
        self.optimizer_D = optim.Adam(
            self.model.discriminator.parameters(),
            lr=self.config.lr_discriminator,
            betas=(self.config.beta1, self.config.beta2),
        )
        self.scheduler_G = optim.lr_scheduler.StepLR(
            self.optimizer_G,
            step_size=self.config.lr_decay_step,
            gamma=self.config.lr_decay_gamma,
        )
        self.scheduler_D = optim.lr_scheduler.StepLR(
            self.optimizer_D,
            step_size=self.config.lr_decay_step,
            gamma=self.config.lr_decay_gamma,
        )
        self.generator_loss = CombinedLoss()
        self.discriminator_loss = nn.BCELoss()
        self.metrics = SegmentationMetrics(
            num_classes=2, device="cuda", iou_foreground_only=True
        )

    def load_config(self, config_path):
        module_name = config_path.split("/")[-1].replace(".py", "")
        spec = importlib.util.spec_from_file_location(module_name, config_path)
        if spec is None:
            raise ImportError(f"Không thể tải file từ {config_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        GANTrainingConfig = getattr(module, "GANTrainingConfig")
        self.config = GANTrainingConfig()

    def save_model(self, epoch, is_best=False):
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        last_model_path = f"{self.config.checkpoint_dir}/last_model.pth"
        self.model.save_checkpoint(last_model_path)
        self.logger.info(f"Saved model checkpoint at {last_model_path}")
        if is_best:
            best_model_path = f"{self.config.checkpoint_dir}/best_gan_model.pth"
            self.model.save_best_checkpoint(best_model_path)
            self.logger.info(f"Saved best model checkpoint at {best_model_path}")

if __name__ == "__main__":
    import argparse
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description="Train GAN model for image segmentation")
    parser.add_argument('--data', default="data/configs/CVC_ClinicDB_config.py")
    parser.add_argument('--config', default="configs/train_config.py")
    parser.add_argument('--train-batch-size', type=int, default=8)
    parser.add_argument('--val-batch-size', type=int, default=8)
    args = parser.parse_args()

    dataset_config_path = args.data
    trainer_config_path = args.config
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size

    model = GanModel(
        generator=Generator(input_shape=(3, 256, 256)),
        discriminator=DiscriminatorWithLRA(4),
        model_name="GAN",
        version="1.0",
        description="GAN for image segmentation",
    )

    train_dataset = CVC_CliniCDBDataset(config_path=dataset_config_path, phase="train")
    val_dataset = CVC_CliniCDBDataset(config_path=dataset_config_path, phase="val")
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size)

    trainer = GANTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config_path=trainer_config_path,
    )
    trainer.train()