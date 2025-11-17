from torch import nn
import torch
class ConCRF(nn.Module):
    def __init__(self, in_chanels, num_classes, kernel_size=3, strides=1, padding=None):
        """
        Args:
            in_chanels (int): Number of input channels.
            out_channels (int): Number of output channels.
            num_classes (int): Number of classes.
            kernel_size (int): Size of the convolutional kernel.
        """
        super(ConCRF, self).__init__()
        self.in_channels = in_chanels
        self.num_classes = num_classes
        self.padding = kernel_size // 2 if padding is None else padding
        self.strides = strides
        self.conv = nn.Conv2d(
            in_chanels,
            num_classes,
            kernel_size=kernel_size,
            stride=self.strides,
            padding=self.padding,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.sigmoid(x)
        return x


class ConvBaseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=None):
        super(ConvBaseBlock, self).__init__()
        self.padding = kernel_size // 2 if padding is None else padding
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
        )
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.leaky_relu(x)
        return x


class DiscriminatorWithConvCRF(nn.Module):
    def __init__(self, inchannels, num_filters=[64, 128, 256, 512, 512]):
        super(DiscriminatorWithConvCRF, self).__init__()
        self.conv = nn.Sequential()
        self.refined = nn.Sequential()
        for i, out_channels in enumerate(num_filters):
            self.conv.add_module(
                f"conv_{i}",
                ConvBaseBlock(inchannels, out_channels, kernel_size=3, stride=2),
            )
            inchannels = out_channels
        for i in range(4):
            self.refined.add_module(
                f"refined_{i}",
                ConCRF(inchannels, inchannels, kernel_size=3, strides=1),
            )
        self.simoid = nn.Sigmoid()
        print(
            f"DiscriminatorWithConvCRF"
        )
    def forward(self, inputs, masks):

        x = torch.cat([inputs, masks], dim=1)
        x = self.conv(x)
        x = self.refined(x)
        x = self.simoid(x)

        return x
