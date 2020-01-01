
import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2
from albumentations import Compose, Resize
from albumentations.pytorch import ToTensorV2


class SemanticKittiDataset(Dataset):
    def __init__(self, dataset_root_dir: Path, filenames: list, normalize_image=True, image_crop=(256, 1024)):
        """
        constructor for dataset class for semantic kitti

        Args:
            dataset_root_dir: path to root directory of dataset
            filenames: names of scans
            augmentation: data transforms for images such as cropping, resize
            has_labels: training set or test set
        """

        self.dataset_root_dir = dataset_root_dir
        self.filenames = filenames
        self.normalize_image = normalize_image
        self.image_transforms = Compose([
            Resize(image_crop[0], image_crop[1]),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.filenames)

    def load_input_image(self, filename):
        image_path = Path(filename)
        image = cv2.imread(str(image_path))
        # print(image_path, image.shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.normalize_image:
            image = (image - image.mean()) / image.std()
        return image

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        # print("filename :", filename)
        image = self.load_input_image(filename)
        transformed = self.image_transforms(image=image)
        image = transformed["image"].type(torch.FloatTensor)
        return image, filename
