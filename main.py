from torch import nn

from models.SqueezeImage import SqueezeImage
from Trainers.CitycsapesTrainer import CityscapesTrainer
from logger import Logger
from metric import MetricsCalculator
from cityscapes.CityscapesDataset import CityscapesDataset
from pathlib import Path
from torch.utils.data import DataLoader
import torch
import yaml
import argparse



def get_filenames(filenames_path):
    with open(str(filenames_path), 'r') as f:
        content = [i.replace('\n', '') for i in f.readlines()]
    return content


parser = argparse.ArgumentParser(description='SqueezeImage')
# data
parser.add_argument('--dataset_root_path', default='data')
parser.add_argument('--batch_size', default='1')
parser.add_argument('--imageset_path', default='imageset')
parser.add_argument('--dataset_config_path', default="cityscapes/cityscapes.yaml")

# training
parser.add_argument('--results_dir', default='results')
parser.add_argument('--learning_rate', default='0.001')
parser.add_argument('--weight_decay', default='1e-5')
parser.add_argument('--number_epochs', default='100')
parser.add_argument('--device', default='cuda')
parser.add_argument('--train_flag', default='1')

# input
parser.add_argument('--image_height', default='512')
parser.add_argument('--image_width', default='1024')

# model
parser.add_argument('--model_name', default='SqueezeImage')
parser.add_argument('--image_channels', default='3')
parser.add_argument('--output_classes', default='20')

args = parser.parse_args()

# data
dataset_root_path = Path(args.dataset_root_path)
batch_size = int(args.batch_size)
imageset_path = Path(args.imageset_path)
dataset_config_path = Path(args.dataset_config_path)

# training
results_dir = Path(args.results_dir)
logging_dir = results_dir / "logs"
learning_rate = float(args.learning_rate)
weight_decay = float(args.weight_decay)
number_epochs = int(args.number_epochs)
train_flag = int(args.train_flag)
device = args.device


# input
model_name = str(args.model_name)
image_width = int(args.image_width)
image_height = int(args.image_height)

# model
image_channels = int(args.image_channels)
output_classes = int(args.output_classes)

# mkdirs
results_dir.mkdir(parents=True, exist_ok=True)
logging_dir.mkdir(parents=True, exist_ok=True)


# trainset = get_filenames(imageset_path / "simple.txt")
# valset = get_filenames(imageset_path / "simple.txt")
# testset = get_filenames(imageset_path / "simple.txt")
trainset = get_filenames(imageset_path / "train.txt")
valset = get_filenames(imageset_path / "val.txt")
testset = get_filenames(imageset_path / "test.txt")

dataset_config = yaml.safe_load(open(str(dataset_config_path), 'r'))

train_dataset = CityscapesDataset(dataset_root_dir=dataset_root_path, filenames=trainset, has_labels=True,
                                  image_height=image_height, image_width=image_width)
val_dataset = CityscapesDataset(dataset_root_dir=dataset_root_path, filenames=valset, has_labels=True,
                                image_height=image_height, image_width=image_width)
test_dataset = CityscapesDataset(dataset_root_dir=dataset_root_path, filenames=testset, has_labels=True,
                                 image_height=image_height, image_width=image_width)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=1)

logger = Logger(logging_dir=str(logging_dir))

print("training model: ", model_name)
model = None
if model_name == "SqueezeImage":
    model = SqueezeImage(num_classes=output_classes)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_weights = torch.ones((len(dataset_config["class_map"].keys())))
loss_weights[0] = 0
loss_fn = nn.NLLLoss(weight=loss_weights)  # nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

trainer = CityscapesTrainer(model=model,
                            loss_fn=loss_fn,
                            optimizer=optimizer,
                            trainloader=train_dataloader,
                            validloader=val_dataloader,
                            num_epochs=number_epochs,
                            device=device,
                            results_dir=str(results_dir),
                            logger=logger,
                            class_map=dataset_config["class_map"])

trainer.train()

# for i in dataloader:
#     image, lidar, target = i
#     sample = (image, lidar)
#     y = model(sample)
