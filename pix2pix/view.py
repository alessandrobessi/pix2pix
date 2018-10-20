import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor


def tensor_to_image(t: Tensor) -> Tensor:
    # noinspection PyUnresolvedReferences
    t = torch.squeeze(t)
    img = (t.detach() + 1) * 127.5
    # noinspection PyUnresolvedReferences
    return img.type(torch.ByteTensor)


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
    plt.xticks([])
    plt.yticks([])
    file_path = os.path.join(examples_dir, 'epoch_{}.png'.format(epoch))
    plt.savefig(file_path, dpi=300)
