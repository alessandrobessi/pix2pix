import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pix2pix.utils import Downsampling, Upsampling
from pix2pix.logger import logger


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.downsample_1 = Downsampling(in_channels=3, out_channels=64, batchnorm=False)
        self.downsample_2 = Downsampling(in_channels=64, out_channels=128)
        self.downsample_3 = Downsampling(in_channels=128, out_channels=256)

        self.conv2d_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(4, 4), stride=1)
        self.conv2d_2 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=(4, 4), stride=1)

    def forward(self, x: Tensor, y: Tensor):
        logger.debug(y.shape)
        y = y.permute(0, 3, 1, 2)
        logger.debug(y.shape)
        logger.debug(x.shape)
        # noinspection PyUnresolvedReferences
        xy = torch.cat([x, y], dim=1)

        x1 = self.downsample_1(x)
        logger.debug(x1.shape)
        x2 = self.downsample_2(x1)
        logger.debug(x2.shape)
        x3 = self.downsample_3(x2)
        logger.debug(x3.shape)
        x4 = F.pad(x3, pad=(1, 1, 1, 1))
        logger.debug(x4.shape)
        x5 = self.conv2d_1(x4)
        logger.debug(x5.shape)
        x6 = F.pad(x5, pad=(1, 1, 1, 1))
        logger.debug(x6.shape)
        x7 = self.conv2d_2(x6)
        logger.debug(x7.shape)
        pass
