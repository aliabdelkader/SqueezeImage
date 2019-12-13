import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import cv2


class CityscapesDataset(Dataset):
    def __init__(self, dataset_root_dir: Path, filenames: list, image_transforms=None, has_labels=True, image_width=256,
                 image_height=1024):
        """
        constructor for dataset class for semantic kitti

        Args:
            dataset_root_dir: path to root directory of dataset
            filenames: names of scans
            image_transforms: data transforms for images such as cropping, resize
            has_labels: training set or test set
        """

        self.has_labels = has_labels
        self.dataset_root_dir = dataset_root_dir
        self.filenames = filenames
        self.image_transforms = image_transforms
        self.image_width = image_width
        self.image_height = image_height

    def __len__(self):
        return len(self.filenames)

    def load_input_image(self, filename):
        image_path = self.dataset_root_dir / "leftImg8bit" / (filename + "_leftImg8bit.png")

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(self.image_width, self.image_height))
        image = image / 255
        # if self.image_transforms:
        #     image = self.image_transforms(image)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        image = image.transpose((2, 0, 1))

        return image

    def load_target(self, filename):
        image_path = self.dataset_root_dir / "gtFine" / (filename + "_labelTrainIds.png")
        image = cv2.imread(str(image_path))
        image = cv2.resize(image, dsize=(self.image_width, self.image_height))

        # if self.image_transforms:
        #     image = self.image_transforms(image)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        image = image.transpose((2, 0, 1))
        return image

    def __getitem__(self, idx):

        filename = self.filenames[idx]
        image = self.load_input_image(filename).astype(np.float32)
        image = image.astype(np.float32)
        if self.has_labels:
            target = self.load_target(filename)
            target = target.astype(np.int64)
            return [torch.from_numpy(image), torch.from_numpy(target)]
        else:
            return [torch.from_numpy(image)]
