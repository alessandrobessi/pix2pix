import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules import BatchNorm2d, Dropout2d


class Downsampling(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 batchnorm: bool = True,
                 bottleneck: bool = False):
        super(Downsampling, self).__init__()
        self.batchnorm = BatchNorm2d(out_channels) if batchnorm else None
        self.bottleneck = bottleneck
        self.kernel_size = (2, 2) if self.bottleneck else (4, 4)
        self.pad = (1, 1, 1, 1) if self.bottleneck else (2, 2, 2, 2)
        self.conv2d = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=self.kernel_size,
                                stride=(1, 1))

    def forward(self, x: Tensor) -> Tensor:
        x = F.max_pool2d(F.pad(self.conv2d(x), pad=self.pad), kernel_size=(2, 2))
        if self.batchnorm and not self.bottleneck:
            x = self.batchnorm(x)
        return F.relu(x)


class Upsampling(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, dropout: bool = False):
        super(Upsampling, self).__init__()
        self.dropout = Dropout2d(0.5) if dropout else None
        self.conv2d_transpose = nn.ConvTranspose2d(in_channels, out_channels,
                                                   kernel_size=(2, 2), stride=2)

    def forward(self, x, corresponding_layer):
        x = F.relu(self.conv2d_transpose(x))
        if self.dropout:
            x = self.dropout(x)
        # noinspection PyUnresolvedReferences
        x = torch.cat([corresponding_layer, x], dim=1)
        return x
