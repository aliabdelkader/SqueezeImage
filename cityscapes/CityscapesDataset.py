import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import cv2
from albumentations import Compose, RandomCrop, Normalize, HorizontalFlip, Resize, ShiftScaleRotate
from albumentations.pytorch import ToTensorV2

class CityscapesDataset(Dataset):
    def __init__(self, dataset_root_dir: Path, filenames: list, augmentation=False, has_labels=True, image_width=256,
                 image_height=1024):
        """
        constructor for dataset class for semantic kitti

        Args:
            dataset_root_dir: path to root directory of dataset
            filenames: names of scans
            augmentation: data transforms for images such as cropping, resize
            has_labels: training set or test set
        """

        self.has_labels = has_labels
        self.dataset_root_dir = dataset_root_dir
        self.filenames = filenames

        self.image_width = image_width
        self.image_height = image_height

        self.image_transforms = Compose([
            Resize(self.image_height, self.image_width),
            ToTensorV2()
        ])
        if augmentation:
            self.image_transforms = Compose([
                                    Resize(self.image_height, self.image_width),
                                    HorizontalFlip(),
                                    ShiftScaleRotate(),
                                    ToTensorV2()
                                ])



    def __len__(self):
        return len(self.filenames)

    def get_image_path(self, filename):
        # berlin_000108_000019_leftImg8bit
        split, city, frame, type = filename.split('-')
        return self.dataset_root_dir / "leftImg8bit" / split / city / "{c}_{f}_leftImg8bit.png".format(c=city, f=frame)

    def load_input_image(self, filename):
        image_path = self.get_image_path(filename)
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def get_target_path(self, filename):
        split, city, frame, type = filename.split('-')
        return self.dataset_root_dir / type / split / city / "{c}_{f}_{t}_labelTrainIds.png".format(c=city, f=frame, t=type)

    def load_target(self, filename):
        target_path = self.get_target_path(filename)
        target_path = cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)

        return target_path

    def __getitem__(self, idx):

        filename = self.filenames[idx]
        image = self.load_input_image(filename).astype(np.float32)
        if self.has_labels:
            target = self.load_target(filename)
            target = target.astype(np.int64)
            transformed = self.image_transforms(image=image, mask=target)
            image, target = transformed["image"], transformed["mask"]
            return [image, target]
        else:
            transformed = self.image_transforms(image=image)
            image, target = transformed["image"], transformed["mask"]
            return [image]
