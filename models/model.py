from typing import Any, Dict, Optional
import torch
import torch.nn as nn


class GanModel(nn.Module):
    def __init__(
        self,
        generator: nn.Module,
        discriminator: Optional[nn.Module] = None,
        model_name: str = "GAN-Segmentation",
        version: str = "1.0",
        description: str = "",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Lớp bao bọc cho GAN bao gồm Generator và Discriminator.

        Args:
            generator: Mạng Generator (tạo mask)
            discriminator: Mạng Discriminator (phân biệt thật/giả)
            model_name: Tên mô hình
            version: Phiên bản
            description: Mô tả
            config: Thông số cấu hình mô hình
        """
        super(GanModel, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.model_name = model_name
        self.version = version
        self.description = description
        self.config = config or {}

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tạo ra đầu ra từ đầu vào.

        Args:
            x: Đầu vào cho Generator

        Returns:
            Đầu ra từ Generator
        """
        return self.generator(x)

    def discriminate_real(self, inputs: torch.Tensor, real_mask) -> torch.Tensor:
        """
        Phân biệt đầu vào thật.

        Args:
            inputs: Đầu vào cho Discriminator
            real_mask: Đầu ra thật từ Generator

        Returns:
            Đầu ra từ Discriminator
        """
        if self.discriminator is not None:
            return self.discriminator(inputs, real_mask)
        else:
            raise ValueError("Discriminator is not defined.")

    def discriminate_fake(
        self, inputs: torch.Tensor, fake_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Phân biệt đầu vào giả.
        Args:
            inputs: Đầu vào cho Discriminator
            fake_masks: Đầu ra giả từ Generator

        Returns:
            Đầu ra từ Discriminator
        """
        if self.discriminator is not None:
            return self.discriminator(inputs, fake_masks)
        else:
            raise ValueError("Discriminator is not defined.")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Hàm forward cho mô hình GAN.

        Args:
            x: Đầu vào cho Generator
        Returns:
            Đầu ra từ Generator
        """
        return self.generate(inputs) # bs, 3, 256, 256

    def save_best_checkpoint(
        self,
        path: str,
    ) -> None:
        state = {
            "model_name": self.model_name,
            "version": self.version,
            "description": self.description,
            "config": self.config,
            "generator_state_dict": self.generator.state_dict(),
        }
        torch.save(state, path)
        print(f"Best model saved to {path}")

    def save_checkpoint(self, path: str) -> None:
        state = {
            "model_name": self.model_name,
            "version": self.version,
            "description": self.description,
            "config": self.config,
            "generator_state_dict": self.generator.state_dict(),
        }
        if self.discriminator is not None:
            state["discriminator_state_dict"] = self.discriminator.state_dict()
        torch.save(state, path)
        print(f"Full GAN model saved at {path}")

    def load_checkpoint(self, path: str, train=False) -> None:
        
        state = torch.load(path, map_location=next(self.parameters()).device)
        self.model_name = state["model_name"]
        self.version = state["version"]
        self.description = state["description"]
        self.config = state["config"]
        self.generator.load_state_dict(state["generator_state_dict"])

        if train and self.discriminator is not None:
            if "discriminator_state_dict" in state:
                self.discriminator.load_state_dict(state["discriminator_state_dict"])
            else:
                raise ValueError("Discriminator state not found in checkpoint")
        print(f"Model loaded from {path}")
