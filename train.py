import importlib
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from data.processing_CVC import CVC_CliniCDBDataset
from loss_functions.loss import CombinedLoss
from models import GanModel, Generator, DiscriminatorWithConvCRF
from utils import SegmentationMetrics, check_loss_nan


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

    def train(self):
        """Main training loop"""
        self.model.to(self.config.device)
        print("Starting training...")
        for epoch in range(self.config.num_epochs):
            train_g_loss, train_d_loss = self.train_one_epoch()
            val_loss, logs = self.validate()

            print(f"Epoch [{epoch + 1}/{self.config.num_epochs}]")
            print(f"Train G Loss: {train_g_loss:.4f}, Train D Loss: {train_d_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            for key, value in logs.items():
                print(f"{key}: {value:.4f}")
            # Save best model
            is_best = False
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                is_best = True
            self.save_model(epoch=epoch, is_best = is_best)
            # Learning rate scheduling
            self.reduce_learning_rate(epoch)

    def train_one_epoch(self):
        """Training logic for one epoch"""
        self.model.train()
        total_g_loss = 0
        total_d_loss = 0

        with tqdm(self.train_loader, unit="batch") as tepoch:
            for data, targets in tepoch:
                data = data.to(self.config.device).to(torch.float32)
                targets = targets.to(self.config.device).to(torch.float32)
                # Then train generator
                g_loss, fake_mask = self.train_generator(data, targets)
                if check_loss_nan(g_loss):
                    print("NaN detected in generator loss. Stopping training.")
                # Train discriminator first
                d_loss = self.train_discriminator(data, fake_mask, targets)
                if check_loss_nan(d_loss):
                    print("NaN detected in discriminator loss. Stopping training.")

                total_d_loss += d_loss
                total_g_loss += g_loss

                tepoch.set_postfix(g_loss=g_loss, d_loss=d_loss)

        avg_g_loss = total_g_loss / len(self.train_loader)
        avg_d_loss = total_d_loss / len(self.train_loader)
        return avg_g_loss, avg_d_loss

    def train_discriminator(self, data, mask_fakes, targets):
        """Train discriminator one step"""
        self.optimizer_D.zero_grad()
        mask_fakes = mask_fakes.detach()  # Detach to avoid backprop through generator
        real_output = self.model.discriminator(data, targets)
        fake_output = self.model.discriminator(data, mask_fakes)
        real_labels = torch.ones_like(real_output) * 0.9  # Use label smoothing
        fake_labels = torch.zeros_like(fake_output)
        # Real loss
        d_real_loss = self.discriminator_loss(real_output, real_labels)
        if check_loss_nan(d_real_loss):
            print("NaN detected in d_real_loss")
        # Fake loss
        d_fake_loss = self.discriminator_loss(fake_output, fake_labels)
        if check_loss_nan(d_real_loss):
            print("NaN detected in d_real_loss")
        # Total loss
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        self.optimizer_D.step()
        return d_loss.item()

    def train_generator(self, data, targets):
        """Train generator one step"""
        self.optimizer_G.zero_grad()

        # Generate fake samples
        fake_masks = self.model.generate(data)
        g_seg_loss = self.generator_loss(fake_masks, targets)
        if check_loss_nan(g_seg_loss):
            print("NaN detected in g_seg_loss")
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
            "total_val_loss": 0,
        }
        with tqdm(self.val_loader, desc="Validating", leave=False) as pbar:
            for data, targets in self.val_loader:
                data, targets = data.to(self.config.device), targets.to(
                    self.config.device
                )
                data, targets = data.to(torch.float32), targets.to(torch.float32)
                outputs = self.model.generate(data)
                combined_loss = self.generator_loss(outputs, targets)
                if check_loss_nan(combined_loss):
                    print("NaN detected in val_loss")
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
        return total_val_loss / len(self.val_loader), logs

    def reduce_learning_rate(self, epoch):
        """Learning rate scheduling"""
        if epoch % 30 == 0 and epoch > 0:
            for param_group in self.optimizer_G.param_groups:
                param_group["lr"] *= 0.5
            for param_group in self.optimizer_D.param_groups:
                param_group["lr"] *= 0.5

    def _setup_training(self):
        """Setup optimizers, schedulers, and loss functions"""

        # Optimizers

        # Tracking metrics
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

        # Learning rate schedulers
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

        # Loss functions
        # Loss functions
        self.generator_loss = CombinedLoss()
        self.discriminator_loss = nn.BCELoss()
        self.metrics = SegmentationMetrics(num_classes=2, device="cuda", iou_foreground_only=True)

    def load_config(self, config_path):
        # Lấy tên module từ đường dẫn
        module_name = config_path.split("/")[-1].replace(".py", "")

        # Tải module từ đường dẫn
        spec = importlib.util.spec_from_file_location(module_name, config_path)
        if spec is None:
            raise ImportError(f"Không thể tải file từ {config_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        # Truy cập class GANTrainingConfig
        GANTrainingConfig = getattr(module, "GANTrainingConfig")
        # Tạo instance của class hoặc truy cập trực tiếp các thuộc tính
        self.config = GANTrainingConfig()

    def save_model(self, epoch, is_best=False):
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        self.model.save_checkpoint(
            f"{self.config.checkpoint_dir}/last_model.pth"
        )
        if is_best:
            self.model.save_best_checkpoint(
                f"{self.config.checkpoint_dir}/best_gan_model.pth"
            )


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = GanModel(
        generator=Generator(input_shape=(3, 256, 256)),
        discriminator=DiscriminatorWithConvCRF(4),
        model_name="GAN",
        version="1.0",
        description="GAN for image segmentation",
    )

    # Dataset
    dataset = CVC_CliniCDBDataset(config_path="data/configs/CVC_ClinicDB_config.py")

    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    print("Data loaded", len(train_loader), len(val_loader))
    # Trainer
    trainer = GANTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config_path="configs/train_config.py",
    )
    trainer.train()
