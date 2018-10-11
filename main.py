import os

import torch
from torch.utils.data import DataLoader

from pix2pix.datasets import FacadesDataset
from pix2pix.transforms import Transform
from pix2pix.logger import logger
from pix2pix.generator import Generator
from pix2pix.discriminator import Discriminator
from pix2pix.losses import generator_loss, discriminator_loss, loss

if __name__ == '__main__':
    dataset = FacadesDataset(root_dir=os.path.join(os.getcwd(), 'data'),
                             split='train',
                             transform=Transform())

    dataloader = DataLoader(dataset,
                            batch_size=1,  # Todo: update
                            shuffle=True,
                            num_workers=1)

    generator = Generator()
    discriminator = Discriminator()

    g_optim = torch.optim.Adam(generator.parameters(), lr=0.001)
    d_optim = torch.optim.Adam(discriminator.parameters(), lr=0.001)

    total_loss = 0

    for input_img, real_img in dataloader:
        g_optim.zero_grad()
        d_optim.zero_grad()

        generated_img = generator(input_img)
        d_judge_generated = discriminator(generated_img, real_img)
        d_judge_real = discriminator(input_img, real_img)

        g_loss = generator_loss(d_judge_generated, generated_img, real_img)
        d_loss = discriminator_loss(d_judge_real, d_judge_generated)
        total_loss = loss(g_loss, d_loss)
        total_loss.backward()

        g_optim.step()
        d_optim.step()

        total_loss += float(total_loss.data)

        logger.debug("Loss: {}".format(total_loss))

        break
