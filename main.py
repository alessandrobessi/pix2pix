import os

from tqdm import tqdm

import tensorboardX
import torch
from torch.utils.data import DataLoader

from pix2pix.logger import logger
from pix2pix.datasets import FacadesDataset
from pix2pix.transforms import Transform
from pix2pix.generator import Generator
from pix2pix.discriminator import Discriminator
from pix2pix.losses import generator_loss, discriminator_loss, loss
from pix2pix.utils import create_working_env

if __name__ == '__main__':

    runs_dir, logs_dir = create_working_env()

    num_epochs = 20

    train_dataset = FacadesDataset(root_dir=os.path.join(os.getcwd(), 'data'),
                                   split='train',
                                   transform=Transform())
    val_dataset = FacadesDataset(root_dir=os.path.join(os.getcwd(), 'data'),
                                 split='val')

    train_loader = DataLoader(train_dataset,
                              batch_size=8,
                              shuffle=True,
                              num_workers=os.cpu_count())

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=True,
                            num_workers=os.cpu_count())

    generator = Generator()
    discriminator = Discriminator()

    g_optim = torch.optim.Adam(generator.parameters(), lr=0.001)
    d_optim = torch.optim.Adam(discriminator.parameters(), lr=0.001)

    train_loss = 0
    val_loss = 0

    writer = tensorboardX.SummaryWriter(logs_dir)

    for epoch in range(num_epochs):
        for input_img, real_img in tqdm(train_loader, desc='Epoch {}'.format(epoch)):
            g_optim.zero_grad()
            d_optim.zero_grad()

            generated_img = generator(input_img)
            d_judge_generated = discriminator(generated_img, real_img)
            d_judge_real = discriminator(input_img, real_img)

            g_loss = generator_loss(d_judge_generated, generated_img, real_img)
            d_loss = discriminator_loss(d_judge_real, d_judge_generated)
            train_loss = loss(g_loss, d_loss)
            train_loss.backward()

            g_optim.step()
            d_optim.step()

        print("Train Loss: {}".format(float(train_loss.data)))

        with torch.no_grad():
            count = 0
            for input_img, real_img in tqdm(val_loader, desc='Epoch {}'.format(epoch)):
                count += 1
                generated_img = generator(input_img)
                d_judge_generated = discriminator(generated_img, real_img)
                d_judge_real = discriminator(input_img, real_img)

                g_loss = generator_loss(d_judge_generated, generated_img, real_img)
                d_loss = discriminator_loss(d_judge_real, d_judge_generated)
                val_loss = loss(g_loss, d_loss)

        print("Val Loss: {}".format(float(val_loss.data)))

        writer.add_scalar('train_loss', float(train_loss.data), epoch)
        writer.add_scalar('val_loss', float(val_loss.data), epoch)

        checkpoint_path = os.path.join(runs_dir, 'checkpoint_{}'.format(epoch))
        torch.save(generator.state_dict(), checkpoint_path)
