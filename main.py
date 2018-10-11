import os

from torch.utils.data import DataLoader

from pix2pix.datasets import FacadesDataset
from pix2pix.transforms import Transform
from pix2pix.logger import logger
from pix2pix.generator import Generator
from pix2pix.discriminator import Discriminator

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

    for input_img, real_img in dataloader:
        logger.debug('main')
        logger.debug(input_img.shape)
        logger.debug(real_img.shape)

        x = generator(input_img)
        discriminator(x, real_img)

        break
