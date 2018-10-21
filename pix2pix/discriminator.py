import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pix2pix.ops import Downsampling, Upsampling
from pix2pix.logger import logger


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.downsample_1 = Downsampling(in_channels=3 * 2, out_channels=64, batchnorm=False)
        self.downsample_2 = Downsampling(in_channels=64, out_channels=128)
        self.downsample_3 = Downsampling(in_channels=128, out_channels=256)

        self.conv2d_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(4, 4), stride=1)
        self.conv2d_2 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=(4, 4), stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor, y: Tensor):
        # noinspection PyUnresolvedReferences
        xy = torch.cat([x, y], dim=1)

        x1 = F.relu(self.downsample_1(xy))
        x2 = F.relu(self.downsample_2(x1))
        x3 = F.relu(self.downsample_3(x2))
        x4 = F.pad(x3, pad=(1, 1, 1, 1))
        x5 = F.relu(self.conv2d_1(x4))
        x6 = F.pad(x5, pad=(1, 1, 1, 1))
        x7 = F.relu(self.conv2d_2(x6))
        return self.sigmoid(x7)
