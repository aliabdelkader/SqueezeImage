import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from metric import MetricsCalculator
from pathlib import Path
import os

class CityscapesTrainer:

    def __init__(self, model, loss_fn, optimizer, trainloader, validloader, num_epochs, device="cpu",
                 results_dir="results", logger=None, class_map=None):
        """
        constructor for trainer class
        :param model: pytorch model to be trained
        :param loss_fn: loss function for training
        :param optimizer: optimizer for training
        :param trainloader: data loader for training set
        :param validloader: data loader for validation set
        :param num_epochs: number of training epochs
        :param device: device for pytorch tensors: cpu or gpu
        :param resutls_dir: directory to save models
        """
        self.trainloader = trainloader
        self.validloader = validloader
        self.num_epochs = num_epochs
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.results_dir = results_dir
        self.logger = logger
        self.class_map = class_map
        self.validation_confusion_matrix = MetricsCalculator(class_map=self.class_map)
        self.train_confusion_matrix = MetricsCalculator(class_map=self.class_map)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1, patience=1, verbose=True)

        # # log model once
        # images, labels = next(iter(trainloader))
        # logger.add_graph(model=model, input=images)

    def save_model(self, epoch, best=False):

        saving_path = os.path.join(self.results_dir, "{}_{}.pth")
        if best:
            torch.save(self.model.module.state_dict(), saving_path.format("best", epoch))
        else:
            torch.save(self.model.module.state_dict(), saving_path.format("model", epoch))


    def train_epoch(self):
        losses = []
        self.train_confusion_matrix.reset_confusion_matrix()

        for train_image, train_label in tqdm(self.trainloader, "training epoch"):
            train_image, train_label = train_image.to(self.device), train_label.to(self.device)

            self.optimizer.zero_grad()  # clear previous gradients

            output = self.model(image=train_image)
            # print("output shape: ", output.shape, "image shape: ", train_image.size(), "target shape: ", train_label.size())

            loss = self.loss_fn(output, train_label)

            losses.append(loss.cpu().detach().numpy())

            loss.backward()  # compute gradients of all variables wrt loss

            self.optimizer.step()

            # _, predicted = torch.max(output.data, 1)
            # y_true = train_label.cpu().detach().numpy().squeeze().reshape(-1)
            # y_pred = predicted.cpu().detach().numpy().squeeze().reshape(-1)
            # self.train_confusion_matrix.update_confusion_matrix(y_true=y_true, y_pred=y_pred)

        loss = np.mean(losses)
        return loss

    def val_epoch(self):
        losses = []
        self.validation_confusion_matrix.reset_confusion_matrix()
        with torch.no_grad():
            for val_image, val_label in tqdm(self.validloader, "validation epoch"):
                val_image, val_label = val_image.to(self.device), val_label.to(self.device)

                output = self.model(image=val_image)

                loss = self.loss_fn(output, val_label)

                losses.append(loss.cpu().detach().numpy())

                _, predicted = torch.max(output.data, 1)

                y_true = val_label.cpu().detach().numpy().squeeze().reshape(-1)
                y_pred = predicted.cpu().detach().numpy().squeeze().reshape(-1)

                self.validation_confusion_matrix.update_confusion_matrix(y_true=y_true, y_pred=y_pred)

        return loss

    def train(self):

        best_preformance = 0
        for epoch in range(self.num_epochs):

            self.model.train()
            train_loss = self.train_epoch()
            # train_miou = self.train_confusion_matrix.calculate_average_iou()

            self.model.eval()
            val_loss = self.val_epoch()
            val_miou = self.validation_confusion_matrix.calculate_average_iou()
            self.scheduler.step(val_miou)
            if val_miou > best_preformance:
                best_preformance = val_miou
                print("best performance {} saving model".format(best_preformance))
                self.save_model(epoch=epoch, best=True)
            else:
                self.save_model(epoch=epoch, best=False)

            print(
                "epoch {} train loss: {} val loss: {} val iou {}".format(epoch, train_loss, val_loss, val_miou))

            if self.logger:
                self.logger.add_metrics(self.validation_confusion_matrix, step=epoch)
                self.logger.add_scalers(scope="loss",
                                        scalers={" training loss": train_loss, " validation loss": val_loss}, step=epoch)
                self.logger.add_scalers(scope="miou",
                                        scalers={" validation miou": val_miou}, step=epoch)
