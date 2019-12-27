
from models.SqueezeImage import SqueezeImage
from cityscapes.CityscapesDataset import CityscapesDataset
from logger import Logger
from metric import MetricsCalculator
from pathlib import Path
from torch.utils.data import DataLoader
import torch
import yaml
import argparse
from tqdm import tqdm
import cv2

def get_filenames(filenames_path):
    with open(str(filenames_path), 'r') as f:
        content = [i.replace('\n', '') for i in f.readlines()]
    return content


parser = argparse.ArgumentParser(description='Middle fusion')
# data
parser.add_argument('--dataset_root_path', default='data')
parser.add_argument('--imageset_path', default='imageset/test.txt')
parser.add_argument('--dataset_config_path', default="cityscapes/cityscapes.yaml")

# training
parser.add_argument('--results_dir', default='results')
parser.add_argument('--device', default='cuda')

# input
parser.add_argument('--image_height', default='512')
parser.add_argument('--image_width', default='1024')

# model
parser.add_argument('--model_name', default='SqueezeImage')
parser.add_argument('--model_path', default='resutls/SqueezeImage/model.pth')
parser.add_argument('--output_classes', default='20')

args = parser.parse_args()

# data
dataset_root_path = Path(args.dataset_root_path)
imageset_path = Path(args.imageset_path)
dataset_config_path = Path(args.dataset_config_path)

# training
results_dir = Path(args.results_dir)
logging_dir = results_dir / "logs"
model_output = results_dir / "model_output"
prediction_dir = model_output / "preds"
ground_truth_dir = model_output / "ground_truth"
device = args.device

# input
model_name = str(args.model_name)
image_width = int(args.image_width)
image_height = int(args.image_height)

# model
model_path = Path(args.model_path)
output_classes = int(args.output_classes)

# mkdirs
results_dir.mkdir(parents=True, exist_ok=True)
logging_dir.mkdir(parents=True, exist_ok=True)
model_output.mkdir(parents=True, exist_ok=True)
prediction_dir.mkdir(parents=True, exist_ok=True)
ground_truth_dir.mkdir(parents=True, exist_ok=True)

files_set = get_filenames(imageset_path)

dataset_config = yaml.safe_load(open(str(dataset_config_path), 'r'))

dataset = CityscapesDataset(dataset_root_dir=dataset_root_path, filenames=files_set, has_labels=True,
                            image_height=image_height,
                            image_width=image_width)

dataloader = DataLoader(dataset, batch_size=1)

logger = Logger(logging_dir=str(logging_dir))

print("evaluating model: ", model_name)

model = None
if model_name == "SqueezeImage":
    model = SqueezeImage(num_classes=output_classes)

if model_path.exists():
    print("loading saved model")
    model.load_state_dict(torch.load(str(model_path)))

# log model
images, lidar, labels = next(iter(dataloader))
logger.add_graph(model=model, input=[images, lidar])

confusion_matrix = MetricsCalculator(class_map=class_map)

model = model.to(device)

# testing loop
results = []
model.eval()
confusion_matrix.reset_confusion_matrix()
with torch.no_grad():
    for idx, sample in tqdm(list(enumerate(dataloader)), "testing loop"):
        # with labels

        image, label = sample
        image, label = image.to(device), label.to(device)

        output = model(image)

        _, predicted = torch.max(output.data, 1)

        predicted_image = predicted.cpu().detach().numpy().squeeze().transpose((1, 2, 0))
        y_pred = predicted.cpu().detach().numpy().squeeze().reshape(-1)

        cv2.imwrite(str(prediction_dir/"{}.png".format(idx)), predicted_image)

        if labels is not None:
            y_true = labels.cpu().detach().numpy().squeeze().reshape(-1)

            confusion_matrix.update_confusion_matrix(y_true=y_true, y_pred=y_pred)

            ground_truth_image = labels.cpu().detach().numpy().squeeze().transpose((1, 2, 0))

            cv2.imwrite(str(ground_truth_dir / "{}.png".format(idx)), ground_truth_image)

iou = confusion_matrix.calculate_average_iou()
print("testing iou {}".format(iou))
with open(str(logging_dir/"miou.txt"), 'w') as f:
    f.write("miou {}".format(iou))
