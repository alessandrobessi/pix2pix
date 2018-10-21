import torch
import torch.nn as nn
from torch import Tensor

from pix2pix.ops import Downsampling, Upsampling
from pix2pix.logger import logger


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        f = 2
        self.downsample_1 = Downsampling(in_channels=3, out_channels=64 * f, batchnorm=False)
        self.downsample_2 = Downsampling(in_channels=64 * f, out_channels=128 * f)
        self.downsample_3 = Downsampling(in_channels=128 * f, out_channels=256 * f)
        self.downsample_4 = Downsampling(in_channels=256 * f, out_channels=512 * f)
        self.downsample_5 = Downsampling(in_channels=512 * f, out_channels=512 * f)
        self.downsample_6 = Downsampling(in_channels=512 * f, out_channels=512 * f)
        self.downsample_7 = Downsampling(in_channels=512 * f, out_channels=512 * f)
        self.downsample_8 = Downsampling(in_channels=512 * f, out_channels=512 * f, bottleneck=True)

        self.upsample_1 = Upsampling(in_channels=512 * f, out_channels=512 * f, dropout=True)
        self.upsample_2 = Upsampling(in_channels=1024 * f, out_channels=512 * f, dropout=True)
        self.upsample_3 = Upsampling(in_channels=1024 * f, out_channels=512 * f, dropout=True)
        self.upsample_4 = Upsampling(in_channels=1024 * f, out_channels=512 * f)
        self.upsample_5 = Upsampling(in_channels=1024 * f, out_channels=256 * f)
        self.upsample_6 = Upsampling(in_channels=512 * f, out_channels=128 * f)
        self.upsample_7 = Upsampling(in_channels=256 * f, out_channels=64 * f)

        self.last = nn.ConvTranspose2d(128 * f, 3, kernel_size=(2, 2), stride=2)

    def forward(self, x: Tensor):
        x1 = self.downsample_1(x)
        x2 = self.downsample_2(x1)
        x3 = self.downsample_3(x2)
        x4 = self.downsample_4(x3)
        x5 = self.downsample_5(x4)
        x6 = self.downsample_6(x5)
        x7 = self.downsample_7(x6)
        x8 = self.downsample_8(x7)

        x9 = self.upsample_1(x8, x7)
        x10 = self.upsample_2(x9, x6)
        x11 = self.upsample_3(x10, x5)
        x12 = self.upsample_4(x11, x4)
        x13 = self.upsample_5(x12, x3)
        x14 = self.upsample_6(x13, x2)
        x15 = self.upsample_7(x14, x1)
        # noinspection PyUnresolvedReferences
        x16 = torch.tanh(self.last(x15))
        return x16
