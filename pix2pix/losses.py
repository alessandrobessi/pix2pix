import torch
from torch import Tensor


def generator_loss_gan(discriminator_judgement: Tensor) -> Tensor:
    eps = 1e-12
    discriminator_judgement = discriminator_judgement.view(-1, 30 * 30)
    # noinspection PyUnresolvedReferences
    gan_loss = torch.mean(-torch.log(discriminator_judgement + eps))
    return gan_loss


def generator_loss_l1(real_image: Tensor,
                      generated_image: Tensor,
                      multiplier: int = 100) -> Tensor:
    # noinspection PyUnresolvedReferences
    l1_loss = torch.mean(torch.abs(real_image - generated_image))
    return l1_loss * multiplier


def discriminator_loss(real_image: Tensor, generated_image: Tensor) -> Tensor:
    eps = 1e-12
    # noinspection PyUnresolvedReferences
    loss = torch.mean(-(torch.log(real_image + eps) + torch.log(1 - generated_image + eps)))
    return loss


def loss(generator_loss_l1: Tensor,
         generator_loss_gan: Tensor,
         discriminator_loss: Tensor) -> Tensor:
    return generator_loss_l1 + generator_loss_gan + discriminator_loss
