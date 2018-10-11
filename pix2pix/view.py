import numpy as np

from torch import Tensor


def tensor_to_array(t: Tensor) -> np.ndarray:
    t = t.permute(0, 2, 3, 1)
    # noinspection PyUnresolvedReferences
    t = torch.squeeze(t)
    t = t.view(256, 256, 3)
    img = (t.detach().numpy() + 1) * 127.5
    return np.uint8(img)
