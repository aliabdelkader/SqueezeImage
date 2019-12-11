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

    def load_image(self, filename):
        image_path = self.dataset_root_dir / (filename + ".png")
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(self.image_width, self.image_height))
        # image = (image - image.mean()) / image.std()
        if self.image_transforms:
            image = self.image_transforms(image)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        image = image.transpose((2, 0, 1))
        return image

    def load_lidar(self, filename):
        lidar_path = self.dataset_root_dir / "lidar" / (filename + ".npy")
        data = np.load(str(lidar_path))

        if self.has_labels:
            lidar, label = data[:, :, :-1], data[:, :, -1]
        else:
            lidar = data[:]

        lidar_mask = np.reshape(
            (lidar[:, :, 4] > 0),
            [lidar.shape[0], lidar.shape[1], 1])

        lidar = np.concatenate([lidar, lidar_mask], axis=-1)

        lidar = lidar.transpose((2, 0, 1))

        return lidar, label

    def __getitem__(self, idx):

        filename = self.filenames[idx]
        image = self.load_image(filename).astype(np.float32)
        lidar, label = self.load_lidar(filename)
        lidar, label = lidar.astype(np.float32), label.astype(np.int64)
        if self.has_labels:
            return [torch.from_numpy(image), torch.from_numpy(lidar), torch.from_numpy(label)]
        else:
            return [torch.from_numpy(image), torch.from_numpy(lidar)]
