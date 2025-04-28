import torch
import torch.nn as nn
import torchvision.models as models


class SqueezeExcitationBlock(nn.Module):
    """Squeeze and Excitation block to recalibrate feature maps."""

    def __init__(self, filters, reduction_ratio=16):
        super(SqueezeExcitationBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(filters, filters // reduction_ratio)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(filters // reduction_ratio, filters)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        se = self.global_avg_pool(inputs)
        se = se.view(se.size(0), -1)
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se)
        se = se.view(se.size(0), -1, 1, 1)
        return inputs * se


class ModifiedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ModifiedResidualBlock, self).__init__()
        self.filters = out_channels // 4

        # 1x1 convolution (bottleneck)
        self.conv1 = nn.Conv2d(in_channels, self.filters, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(self.filters)
        self.relu1 = nn.ReLU()

        # 3x3 convolution
        self.conv2 = nn.Conv2d(self.filters, self.filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.filters)

        # Shortcut adjustment
        self.shortcut_conv = (
            nn.Conv2d(in_channels, self.filters, kernel_size=1, padding=0)
            if in_channels != self.filters
            else None
        )
        self.shortcut_bn = (
            nn.BatchNorm2d(self.filters) if in_channels != self.filters else None
        )

        self.relu2 = nn.ReLU()
        self.se_block = SqueezeExcitationBlock(self.filters, out_channels)

    def forward(self, x):
        shortcut = x
        # Main path
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # Shortcut path
        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(shortcut)
            shortcut = self.shortcut_bn(shortcut)

        # Add and process
        x = x + shortcut
        x = self.relu2(x)
        x = self.se_block(x)
        return x


class Encoder(nn.Module):
    def __init__(self, input_shape):
        super(Encoder, self).__init__()
        base_model = models.mobilenet_v2(pretrained=True)
        self.features = base_model.features[:15]
        self.outputs = {}
        self.features[2].conv[0][2].register_forward_hook(
            lambda m, i, o: self.outputs.update({"skip1": o})
        )
        self.features[4].conv[0][2].register_forward_hook(
            lambda m, i, o: self.outputs.update({"skip2": o})
        )
        self.features[7].conv[0][2].register_forward_hook(
            lambda m, i, o: self.outputs.update({"skip3": o})
        )
        self.features[14].conv[0][2].register_forward_hook(
            lambda m, i, o: self.outputs.update({"encoder_output": o})
        )
        self.features[14].conv = nn.Sequential(self.features[14].conv[0])

    def forward(self, x):
        _ = self.features(x)
        return [
            self.outputs["encoder_output"],
            self.outputs["skip1"],
            self.outputs["skip2"],
            self.outputs["skip3"],
        ]


class Generator(nn.Module):
    def __init__(self, input_shape):
        super(Generator, self).__init__()
        self.encoder = Encoder(input_shape)

        # Bridge
        self.bottleneck = ModifiedResidualBlock(
            576, 576
        )  # 320 is MobileNetV2's last feature map channels

        # Decoder blocks
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv1 = ModifiedResidualBlock(336, 256)  # 320 + 128 (skip3 channels)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv2 = ModifiedResidualBlock(208, 128)  # 128 + 64 (skip2 channels)

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv3 = ModifiedResidualBlock(128, 64)  # 64 + 24 (skip1 channels)

        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        encoder_output, skip1, skip2, skip3 = self.encoder(x)
        # encoder_output: [2, 576, 16, 16] Skip1: [2, 96, 128, 128] Skip2: [2, 144, 64, 64] Skip3: [2, 192, 32, 32]
        # Bridge
        x = self.bottleneck(encoder_output)  # [2, 144, 16, 16]

        # Decoder
        x = self.up1(x)  # [2, 144, 32, 32]
        x = torch.cat([x, skip3], dim=1)  # [2, 336, 32, 32]
        x = self.conv1(x)  # [2, 64, 32, 32]
        x = self.up2(x)  # [2, 64, 64, 64]
        x = torch.cat([x, skip2], dim=1)  # [2, 208, 64, 64]
        x = self.conv2(x)  # [2, 32, 64, 64]
        x = self.up3(x)  # [2, 32, 128, 128]
        x = torch.cat([x, skip1], dim=1)  # [2, 128, 128, 128]
        x = self.conv3(x)  # [2, 16, 128, 128]
        x = self.up4(x)  # [2, 16, 256, 256]
        x = self.final_conv(x)  # [2, 1, 256, 256]
        x = self.sigmoid(x)  # [2, 1, 256, 256]

        return x


# Example usage
input_shape = (3, 256, 256)  # PyTorch uses (channels, height, width)
x = torch.randn(1, *input_shape)  # Batch size of 1
generator_model = Generator(input_shape)
# output = generator_model(x)
# print(output)
# To print model summary (requires torchsummary)
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator_model = generator_model.to(device)
summary(generator_model, input_shape)
