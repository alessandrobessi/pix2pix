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
from pix2pix.view import save_image

if __name__ == '__main__':

    runs_dir, logs_dir, examples_dir = create_working_env()

    num_epochs = 50

    train_dataset = FacadesDataset(root_dir=os.path.join(os.getcwd(), 'data'),
                                   split='train',
                                   transform=Transform())
    val_dataset = FacadesDataset(root_dir=os.path.join(os.getcwd(), 'data'),
                                 split='val')

    train_loader = DataLoader(train_dataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=os.cpu_count())

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=True,
                            num_workers=os.cpu_count())

    generator = Generator()
    discriminator = Discriminator()

    g_optim = torch.optim.Adam(generator.parameters(), lr=0.0001)
    d_optim = torch.optim.Adam(discriminator.parameters(), lr=0.0001)

    writer = tensorboardX.SummaryWriter(logs_dir)

    for epoch in range(num_epochs):
        step = 0
        total_train_loss = 0
        for input_img, real_img in tqdm(train_loader, desc='Epoch {}'.format(epoch)):
            step += 1
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

            total_train_loss += float(train_loss.data)

        logger.info("Train Loss: {}".format(total_train_loss / step))

        with torch.no_grad():
            step = 0
            total_val_loss = 0
            for input_img, real_img in tqdm(val_loader, desc='Epoch {}'.format(epoch)):
                step += 1
                generated_img = generator(input_img)
                d_judge_generated = discriminator(generated_img, real_img)
                d_judge_real = discriminator(input_img, real_img)

                g_loss = generator_loss(d_judge_generated, generated_img, real_img)
                d_loss = discriminator_loss(d_judge_real, d_judge_generated)
                val_loss = loss(g_loss, d_loss)

                total_val_loss += float(val_loss.data)

        save_image([input_img, real_img, generated_img], examples_dir, epoch)

        logger.info("Val Loss: {}".format(total_val_loss // step))

        writer.add_scalar('train_loss', total_train_loss // step, epoch)
        writer.add_scalar('val_loss', total_val_loss // step, epoch)

        checkpoint_path = os.path.join(runs_dir, 'checkpoint_{}'.format(epoch))
        torch.save(generator.state_dict(), checkpoint_path)
