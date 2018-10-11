import numpy as np

import torch
import torch.nn as nn
from torch import Tensor

from pix2pix.logger import logger


def generator_loss(discriminator_judgement: Tensor,
                   generated_image: Tensor,
                   real_image: Tensor,
                   correction: int = 100) -> Tensor:
    bce = nn.BCELoss()
    l1 = nn.L1Loss()
    # noinspection PyUnresolvedReferences
    discriminator_judgement = discriminator_judgement.view(-1, 30 * 30)
    # noinspection PyUnresolvedReferences
    gan_loss = bce(discriminator_judgement,
                   torch.ones(discriminator_judgement.size()))
    l1_loss = l1(real_image, generated_image)

    return gan_loss + correction * l1_loss


def discriminator_loss(real_image: Tensor, generated_image: Tensor) -> Tensor:
    bce = nn.BCELoss()
    real_image = real_image.view(-1, 30 * 30)
    generated_image = generated_image.view(-1, 30 * 30)
    # noinspection PyUnresolvedReferences
    real_loss = bce(real_image, torch.ones(real_image.shape))
    # noinspection PyUnresolvedReferences
    generated_loss = bce(generated_image, torch.zeros(generated_image.shape))

    return real_loss + generated_loss


def loss(generator_loss: Tensor, discriminator_loss: Tensor) -> Tensor:
    return generator_loss + discriminator_loss
