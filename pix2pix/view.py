import os
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import Tensor


def tensor_to_array(t: Tensor) -> np.ndarray:
    t = t.permute(0, 2, 3, 1)
    # noinspection PyUnresolvedReferences
    t = torch.squeeze(t)
    t = t.view(256, 256, 3)
    img = (t.detach().numpy() + 1) * 127.5
    return np.uint8(img)


def save_image(images: Tuple, examples_dir: str, epoch: int) -> None:
    images = [tensor_to_array(image) for image in images]
    img = np.concatenate(images, axis=1)
    plt.imshow(img, interpolation='nearest')
    file_path = os.path.join(examples_dir, 'epoch_{}.png'.format(epoch))
    plt.savefig(file_path, dpi=300)
