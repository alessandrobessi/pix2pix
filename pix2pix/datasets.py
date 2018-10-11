import os
from typing import Tuple, Optional, Callable

import cv2
import numpy as np
from torch.utils.data import Dataset

from pix2pix.logger import logger


class FacadesDataset(Dataset):

    def __init__(self,
                 root_dir: str,
                 split: str,
                 transform: Optional[Callable[[np.ndarray, np.ndarray],
                                              Tuple[np.ndarray, np.ndarray]]] = None):
        self.root_dir = os.path.expanduser(os.path.normpath(root_dir))
        self.split = split
        self.transform = transform
        self.images_path = os.path.join(self.root_dir, self.split)
        self.images_list = [f for f in os.listdir(self.images_path)
                            if os.path.isfile(os.path.join(self.images_path, f))]

    def __len__(self) -> int:
        return len(self.images_list)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        image = cv2.imread(os.path.join(self.images_path, self.images_list[idx]))

        half_width = int(image.shape[1] / 2)

        real_image = image[:, :half_width, :].astype(np.float32)
        input_image = image[:, half_width:, :].astype(np.float32)

        if self.transform:
            input_image, real_image = self.transform(input_image, real_image)

        real_image = real_image / 127.5 - 1
        input_image = input_image / 127.5 - 1

        return input_image, real_image
