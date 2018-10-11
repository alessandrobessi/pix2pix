from typing import Tuple

import numpy as np
from PIL import Image
from torchvision.transforms import Resize, RandomCrop, RandomHorizontalFlip

from pix2pix.logger import logger


class Transform:

    def __call__(self,
                 input_image: np.ndarray,
                 real_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        resize = Resize(286)
        random_crop = RandomCrop(256)
        random_horizontal_flip = RandomHorizontalFlip(0.5)

        input_image = Image.fromarray(np.uint8(input_image))
        real_image = Image.fromarray(np.uint8(real_image))

        input_image = random_horizontal_flip(random_crop(resize(input_image)))
        real_image = random_horizontal_flip(random_crop(resize(real_image)))

        return np.array(input_image, dtype=np.float32), np.array(real_image, dtype=np.float32)
