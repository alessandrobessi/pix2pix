import os

from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
from torch.utils.data import DataLoader

from pix2pix.datasets import FacadesDataset
from pix2pix.transforms import Transform
from pix2pix.generator import Generator
from pix2pix.discriminator import Discriminator
from pix2pix.losses import generator_loss, discriminator_loss, loss

if __name__ == '__main__':

    num_epochs = 1

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

            for i, input_img, real_img in enumerate(tqdm(val_loader,
                                                         desc='Epoch {}'.format(epoch))):
                generated_img = generator(input_img)
                d_judge_generated = discriminator(generated_img, real_img)
                d_judge_real = discriminator(input_img, real_img)

                g_loss = generator_loss(d_judge_generated, generated_img, real_img)
                d_loss = discriminator_loss(d_judge_real, d_judge_generated)
                val_loss = loss(g_loss, d_loss)
                print("Val Loss: {}".format(float(val_loss.data)))

                if i % 10:
                    plt.imshow(input_img, interpolation='nearest')
                    plt.show()

                    plt.imshow(real_img, interpolation='nearest')
                    plt.show()

                    plt.imshow(generated_img, interpolation='nearest')
                    plt.show()
