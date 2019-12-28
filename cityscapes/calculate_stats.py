from CityscapesDataset import CityscapesDataset
from pathlib import Path
from torch.utils.data import DataLoader
import torch
import yaml
import argparse
import torch.nn.functional as F


def get_filenames(filenames_path):
    with open(str(filenames_path), 'r') as f:
        content = [i.replace('\n', '') for i in f.readlines()]
    return content


parser = argparse.ArgumentParser(description='SqueezeImage')
# data
parser.add_argument('--dataset_root_path', default='data')
parser.add_argument('--imageset_path', default='imageset')
parser.add_argument('--dataset_config_path', default="cityscapes/cityscapes.yaml")

# training
parser.add_argument('--results_dir', default='results')

# input
parser.add_argument('--image_height', default='512')
parser.add_argument('--image_width', default='1024')

parser.add_argument('--output_classes', default='20')

args = parser.parse_args()

# data
dataset_root_path = Path(args.dataset_root_path)
imageset_path = Path(args.imageset_path)
dataset_config_path = Path(args.dataset_config_path)

# training
results_dir = Path(args.results_dir)
logging_dir = results_dir / "logs"

# input
image_width = int(args.image_width)
image_height = int(args.image_height)

# model
output_classes = int(args.output_classes)

# mkdirs
results_dir.mkdir(parents=True, exist_ok=True)
logging_dir.mkdir(parents=True, exist_ok=True)

trainset = get_filenames(imageset_path / "train.txt")

dataset_config = yaml.safe_load(open(str(dataset_config_path), 'r'))

train_dataset = CityscapesDataset(dataset_root_dir=dataset_root_path,
                                  filenames=trainset,
                                  has_labels=True,
                                  image_height=image_height,
                                  image_width=image_width,
                                  augmentation=False,
                                  normalize_image=False)

dataloader = DataLoader(train_dataset, batch_size=1)
stats = torch.ones((output_classes))

for sample in (dataloader):
    # with labels
    image, labels = sample

    uni, count = torch.unique(labels, return_counts=True)
    for i, c in zip(uni, count):
        stats[i] += c

for idx, c in enumerate(stats):
    print("class: {} counts {}".format(idx, c))

scores = stats / stats.max()
print("scores: ", scores)

print("inverse: ", 1 - scores)
