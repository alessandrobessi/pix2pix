import os
import numpy as np
import matplotlib.pyplot as plt

from torch import Tensor


def tensor_to_array(t: Tensor) -> np.ndarray:
    t = t.permute(0, 2, 3, 1)
    # noinspection PyUnresolvedReferences
    t = torch.squeeze(t)
    t = t.view(256, 256, 3)
    img = (t.detach().numpy() + 1) * 127.5
    return np.uint8(img)


def save_image(t: Tensor, examples_dir: str, name: str, epoch: int) -> None:
    a = tensor_to_array(t)
    plt.imshow(a, interpolation='nearest')
    file_path = os.path.join(examples_dir, 'epoch_{}_{}.png'.format(epoch, name))
    plt.savefig(file_path, dpi=300)
