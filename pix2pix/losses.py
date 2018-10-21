import torch
import torch.nn as nn
from torch import Tensor


def generator_loss_gan(discriminator_judgement: Tensor,
                       gan_weight: int = 1) -> Tensor:
    mse = nn.MSELoss()
    # noinspection PyUnresolvedReferences
    discriminator_judgement = discriminator_judgement.view(-1, 30 * 30)
    # noinspection PyUnresolvedReferences
    gan_loss = mse(discriminator_judgement, torch.ones(discriminator_judgement.shape))
    return gan_loss * gan_weight


def generator_loss_l1(real_image: Tensor,
                      generated_image: Tensor,
                      l1_weight: int = 100) -> Tensor:
    # noinspection PyUnresolvedReferences
    l1_loss = torch.mean(torch.abs(real_image - generated_image))
    return l1_loss * l1_weight


def discriminator_loss(real_image: Tensor, generated_image: Tensor) -> Tensor:
    mse = nn.MSELoss()
    real_image = real_image.view(-1, 30 * 30)
    generated_image = generated_image.view(-1, 30 * 30)
    # noinspection PyUnresolvedReferences
    real_loss = mse(real_image, torch.ones(real_image.shape))
    # noinspection PyUnresolvedReferences
    generated_loss = mse(generated_image, torch.zeros(generated_image.shape))

    return real_loss + generated_loss
