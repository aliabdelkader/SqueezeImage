from models.SqueezeImage import SqueezeImage
from semantickitti.SemanticKittiDataset import SemanticKittiDataset
from logger import Logger
from metric import MetricsCalculator
from pathlib import Path
from torch.utils.data import DataLoader
import torch
import yaml
import argparse
from tqdm import tqdm
import cv2
import numpy as np


def get_filenames(filenames_path):
    with open(str(filenames_path), 'r') as f:
        content = [i.replace('\n', '') for i in f.readlines()]
    return content


def get_save_path(save_root_path, file_path, save_dir="SqueezeImage_preds"):
    file_path = Path(file_path)
    frame_number = file_path.stem
    seq_number = file_path.parent.parent.stem
    save_path = save_root_path / seq_number / save_dir / "{}.png".format(frame_number)
    return str(save_path)


parser = argparse.ArgumentParser(description='semantic kitti infer')
# data
parser.add_argument('--dataset_root_path', default='data')
parser.add_argument('--imageset_path', default='imageset')
parser.add_argument('--image_width', default='1241')
parser.add_argument('--image_height', default='376')

# training
parser.add_argument('--results_dir', default='results')
parser.add_argument('--device', default='cuda')

# model
parser.add_argument('--model_name', default='SqueezeImage')
parser.add_argument('--model_path', default='resutls/SqueezeImage/model.pth')
parser.add_argument('--output_classes', default='20')

args = parser.parse_args()

# data
dataset_root_path = Path(args.dataset_root_path)
imageset_path = Path(args.imageset_path)
image_width = int(args.image_width)
image_height = int(args.image_height)

# training
results_dir = Path(args.results_dir)
device = args.device

# input
model_name = str(args.model_name)
output_classes = int(args.output_classes)

# model
model_path = Path(args.model_path)

if not dataset_root_path.exists():
    raise ("dataset does not exists")

if not model_path.exists():
    raise ("pretrained model does not exits")

# mkdirs
results_dir.mkdir(parents=True, exist_ok=True)


files_set = get_filenames(imageset_path / "all.txt")

dataset = SemanticKittiDataset(dataset_root_dir=dataset_root_path, filenames=files_set, normalize_image=True)

dataloader = DataLoader(dataset, batch_size=1)



print("evaluating model: ", model_name)

model = None
if model_name == "SqueezeImage":
    model = SqueezeImage(num_classes=output_classes)

if model_path.exists():
    print("loading saved model")
    model.load_state_dict(torch.load(str(model_path)))

model = model.to(device)

# testing loop
results = []
model.eval()
with torch.no_grad():
    for idx, sample in tqdm(enumerate(dataloader), "testing loop"):
        # with labels

        image, filename = sample
        image = image.to(device)

        output = model(image)

        predicted = output.argmax(dim=1)

        predicted_image = predicted.cpu().detach().numpy().transpose((1, 2, 0))

        cv2.resize(predicted_image, dsize=(image_height, image_width))
        save_path = get_save_path(results_dir, filename)
        cv2.imwrite(save_path, predicted_image)
