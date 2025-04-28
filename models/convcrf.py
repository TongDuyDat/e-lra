from torch import nn

class ConCRF(nn.Module):
    def __init__(self, in_chanels, num_classes, kernel_size=3, strides=1, padding = None):
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
        self.conv = nn.Conv2d(in_chanels, num_classes, kernel_size=kernel_size, stride=self.strides, padding=self.padding)
        self.sigmoid = nn.Sigmoid()
    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.sigmoid(x)
        return x