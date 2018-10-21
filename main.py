import os

import tensorboardX
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pix2pix.datasets import FacadesDataset
from pix2pix.discriminator import Discriminator
from pix2pix.generator import Generator
from pix2pix.losses import generator_loss_l1, generator_loss_gan, discriminator_loss
from pix2pix.transforms import Transform
from pix2pix.utils import create_working_env
from pix2pix.view import tensor_to_image

if __name__ == '__main__':

    data_dir, runs_dir, logs_dir, examples_dir = create_working_env()

    num_epochs = 100
    lr = 0.0002
    discriminator_steps = 10

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

    writer = tensorboardX.SummaryWriter(logs_dir)

    step = 0
    for epoch in range(num_epochs):

        for input_img, real_img in tqdm(train_loader, desc='Epoch {}'.format(epoch)):
            step += 1

            # generator update
            g_optim.zero_grad()
            generated_img = generator(input_img)
            d_judge_generated = discriminator(generated_img, real_img)
            g_loss_l1 = generator_loss_l1(real_img, generated_img)
            g_loss_gan = generator_loss_gan(d_judge_generated)
            g_loss = 0.5 * g_loss_l1 + 0.5 * g_loss_gan
            g_loss.backward()
            g_optim.step()

            writer.add_scalar('training/generator_loss_l1', float(g_loss_l1.item()), step)
            writer.add_scalar('training/generator_loss_gan', float(g_loss_gan.item()), step)
            writer.add_scalar('training/generator_loss_total', float(g_loss.item()), step)

            if step % discriminator_steps == 0:
                # discriminator update
                d_optim.zero_grad()
                d_judge_generated = discriminator(generated_img.detach(), real_img)
                d_judge_real = discriminator(input_img, real_img)
                d_loss = discriminator_loss(d_judge_real, d_judge_generated)
                d_loss.backward()
                d_optim.step()

                writer.add_scalar('training/discriminator_loss', float(d_loss.data), step)

        with torch.no_grad():
            for input_img, real_img in tqdm(val_loader, desc='Epoch {}'.format(epoch)):
                generated_img = generator(input_img)
                d_judge_generated = discriminator(generated_img, real_img)
                d_judge_real = discriminator(input_img, real_img)

                g_loss_l1 = generator_loss_l1(real_img, generated_img)
                g_loss_gan = generator_loss_gan(d_judge_generated)
                g_loss_total = 0.5 * g_loss_l1 + 0.5 * g_loss_gan
                d_loss = discriminator_loss(d_judge_real, d_judge_generated)

        writer.add_scalar('validation/generator_loss_l1', float(g_loss_l1.item()), step)
        writer.add_scalar('validation/generator_loss_gan', float(g_loss_gan.item()), step)
        writer.add_scalar('validation/generator_loss_total', float(g_loss_total.item()), step)
        writer.add_scalar('validation/discriminator_loss', float(d_loss.item()), step)

        writer.add_image('step_{}/input'.format(epoch), tensor_to_image(input_img), epoch)
        writer.add_image('step_{}/real'.format(epoch), tensor_to_image(real_img), epoch)
        writer.add_image('step_{}/generated'.format(epoch), tensor_to_image(generated_img), epoch)

        # save_image([input_img, real_img, generated_img], examples_dir, epoch)

        checkpoint_path = os.path.join(runs_dir, 'checkpoint_{}'.format(epoch))
        torch.save(generator.state_dict(), checkpoint_path)
