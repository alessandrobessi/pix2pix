import os

import tensorboardX
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pix2pix.datasets import FacadesDataset
from pix2pix.discriminator import Discriminator
from pix2pix.generator import Generator
from pix2pix.losses import generator_loss_l1, generator_loss_gan, discriminator_loss, loss
from pix2pix.transforms import Transform
from pix2pix.utils import create_working_env
from pix2pix.view import save_image

if __name__ == '__main__':

    data_dir, runs_dir, logs_dir, examples_dir = create_working_env()

    num_epochs = 100
    lr = 0.0001

    train_dataset = FacadesDataset(data_dir=data_dir,
                                   split='train',
                                   transform=Transform())
    val_dataset = FacadesDataset(data_dir=data_dir,
                                 split='val')

    train_loader = DataLoader(train_dataset,
                              batch_size=1,
                              shuffle=True,
                              num_workers=os.cpu_count())

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=True,
                            num_workers=os.cpu_count())

    generator = Generator()
    discriminator = Discriminator()

    g_optim = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optim = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    g_scheduler = torch.optim.lr_scheduler.StepLR(g_optim, step_size=num_epochs // 20, gamma=0.8)
    d_scheduler = torch.optim.lr_scheduler.StepLR(d_optim, step_size=num_epochs // 20, gamma=0.8)

    writer = tensorboardX.SummaryWriter(logs_dir)

    step = 0
    for epoch in range(num_epochs):
        g_scheduler.step()
        d_scheduler.step()

        for input_img, real_img in tqdm(train_loader, desc='Epoch {}'.format(epoch)):
            step += 1

            generated_img = generator(input_img)
            d_judge_generated = discriminator(generated_img, real_img)
            d_judge_real = discriminator(input_img, real_img)

            # generator update
            g_optim.zero_grad()
            g_loss_l1 = generator_loss_l1(real_img, generated_img)
            g_loss_gan = generator_loss_gan(d_judge_generated)
            g_loss = g_loss_l1 + g_loss_gan
            g_loss.backward()
            g_optim.step()

            # discriminator update
            d_optim.zero_grad()
            d_loss = discriminator_loss(d_judge_real, d_judge_generated.detach())
            d_loss.backward()
            d_optim.step()

            writer.add_scalar('generator_loss_l1', float(g_loss_l1.data), step)
            writer.add_scalar('generator_loss_gan', float(g_loss_gan.data), step)
            writer.add_scalar('discriminator_loss', float(d_loss.data), step)

        with torch.no_grad():
            for input_img, real_img in tqdm(val_loader, desc='Epoch {}'.format(epoch)):
                generated_img = generator(input_img)
                d_judge_generated = discriminator(generated_img, real_img)
                d_judge_real = discriminator(input_img, real_img)

                g_loss_l1 = generator_loss_l1(real_img, generated_img)
                g_loss_gan = generator_loss_gan(d_judge_generated)
                d_loss = discriminator_loss(d_judge_real, d_judge_generated)
                val_loss = loss(g_loss_l1, g_loss_gan, d_loss)

        writer.add_scalar('val_loss', float(val_loss.data), step)

        writer.add_scalar('lr_generator', g_scheduler.get_lr()[0], step)
        writer.add_scalar('lr_discriminator', d_scheduler.get_lr()[0], step)

        save_image([input_img, real_img, generated_img], examples_dir, epoch)

        checkpoint_path = os.path.join(runs_dir, 'checkpoint_{}'.format(epoch))
        torch.save(generator.state_dict(), checkpoint_path)
