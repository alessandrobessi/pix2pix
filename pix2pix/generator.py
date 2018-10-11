import torch
import torch.nn as nn
from torch import Tensor

from pix2pix.ops import Downsampling, Upsampling
from pix2pix.logger import logger


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.downsample_1 = Downsampling(in_channels=3, out_channels=64, batchnorm=False)
        self.downsample_2 = Downsampling(in_channels=64, out_channels=128)
        self.downsample_3 = Downsampling(in_channels=128, out_channels=256)
        self.downsample_4 = Downsampling(in_channels=256, out_channels=512)
        self.downsample_5 = Downsampling(in_channels=512, out_channels=512)
        self.downsample_6 = Downsampling(in_channels=512, out_channels=512)
        self.downsample_7 = Downsampling(in_channels=512, out_channels=512)

        self.upsample_1 = Upsampling(in_channels=512, out_channels=512, dropout=True)
        self.upsample_2 = Upsampling(in_channels=1024, out_channels=512, dropout=True)
        self.upsample_3 = Upsampling(in_channels=1024, out_channels=512, dropout=True)
        self.upsample_4 = Upsampling(in_channels=1024, out_channels=256)
        self.upsample_5 = Upsampling(in_channels=512, out_channels=128)
        self.upsample_6 = Upsampling(in_channels=256, out_channels=64)

        self.last = nn.ConvTranspose2d(128, 3, kernel_size=(2, 2), stride=2)

    def forward(self, x: Tensor):
        x1 = self.downsample_1(x)
        x2 = self.downsample_2(x1)
        x3 = self.downsample_3(x2)
        x4 = self.downsample_4(x3)
        x5 = self.downsample_5(x4)
        x6 = self.downsample_6(x5)
        x7 = self.downsample_7(x6)

        x8 = self.upsample_1(x7, x6)
        x9 = self.upsample_2(x8, x5)
        x10 = self.upsample_3(x9, x4)
        x11 = self.upsample_4(x10, x3)
        x12 = self.upsample_5(x11, x2)
        x13 = self.upsample_6(x12, x1)
        # noinspection PyUnresolvedReferences
        x14 = torch.tanh(self.last(x13))
        return x14
