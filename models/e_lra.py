from torch import nn
import torch


class ConvBaseBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=2,
        padding=None,
        alpha=0.2,
    ):
        super(ConvBaseBlock, self).__init__()
        self.padding = kernel_size // 2 if padding is None else padding
        self.alpha = 0.2 if alpha is None else alpha
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
        )
        self.leaky_relu = nn.LeakyReLU(self.alpha, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.leaky_relu(x)
        return x


class SqueezeExcitationBlock(nn.Module):
    """Squeeze and Excitation block to recalibrate feature maps."""

    def __init__(self, in_channels, reduction_ratio=16):
        super(SqueezeExcitationBlock, self).__init__()
        reduced_channels = in_channels // reduction_ratio
        if reduced_channels == 0:
            print(
                f"Warning: reduction_ratio {reduction_ratio} is too large for in_channels {in_channels}. Setting reduced_channels to 1."
            )
        self.squeeze = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.excite = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        se = self.squeeze(x)  # [B, C, 1, 1]
        se = self.excite(se)  # [B, C, 1, 1]
        return x * se


class LRA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super(LRA, self).__init__()
        self.dw_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2 if padding is None else padding,
            groups=in_channels,
        )
        self.point_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1
        )
        self.se_block = SqueezeExcitationBlock(out_channels)

    def forward(self, x):
        shortcut = x
        x = self.dw_conv(x)
        x = self.point_conv(x)
        se = self.se_block(shortcut)
        x = x + se
        return x


class DiscriminatorWithLRA(nn.Module):
    def __init__(self, inchannels, num_filters=[64, 128, 256, 512, 512]):
        super(DiscriminatorWithLRA, self).__init__()
        self.conv = nn.Sequential()
        edg_input = inchannels - 1
        for i, out_channels in enumerate(num_filters):
            self.conv.add_module(
                f"conv_{i}",
                ConvBaseBlock(inchannels, out_channels, kernel_size=3, stride=2),
            )
            inchannels = out_channels
        self.edg_gui_attn = nn.Sequential(
            nn.Conv2d(edg_input, 1, kernel_size=3), nn.Sigmoid()
        )
        self.lra = LRA(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.refined = nn.Conv2d(inchannels, inchannels, kernel_size=1, stride=1)
        self.simoid = nn.Sigmoid()

    def forward(self, inputs, masks):
        x = torch.cat([inputs, masks], dim=1)
        f = self.conv(x)
        ae = self.edg_gui_attn(inputs)
        f = self.lra(f)
        f_shape = f.shape
        ae = torch.nn.functional.interpolate(
            ae, size=f_shape[2:], mode='bilinear', align_corners=False
        )
        refined = ae * f + f
        refined = self.refined(refined)
        x = self.simoid(refined)
        return x


model = DiscriminatorWithLRA(4)
image = torch.randn(1, 3, 256, 256)
mask = torch.randn(1, 1, 256, 256)
output = model(image, mask)
print(output.shape)  # Expected output shape: [1, 512, 16, 16]
