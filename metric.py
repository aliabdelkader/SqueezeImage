import numpy as np
from sklearn.metrics import multilabel_confusion_matrix


class MetricsCalculator:
    TN_INDEX = (0, 0)
    FP_INDEX = (0, 1)
    FN_INDEX = (1, 0)
    TP_INDEX = (1, 1)

    def __init__(self, class_map: dict):
        """
        constructor

        Args:
            class_map: dictnory mapping index to classes

        class keeps track of confusion matrix,
        its shape is number of classes, 2, 2
        confusion matix
        [ [TN, FP],
        [FN, TP]]
        """

        self.confusion_matrix = np.zeros((len(class_map), 2, 2))
        self.class_map = class_map
        self.class_indexs = sorted(list(self.class_map.values()))

    def update_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        function updates the stored confusion confusion matrix

        Args:
            y_true: numpy array of ground truth
            y_pred: numpy array of prediction
        """
        update = multilabel_confusion_matrix(y_pred=y_pred, y_true=y_true, labels=self.class_indexs)
        self.confusion_matrix += update

    def reset_confusion_matrix(self):
        del self.confusion_matrix
        self.confusion_matrix = np.zeros((len(self.class_map), 2, 2))

    def read_matrix_values(self, class_name: str):
        """
        function reads confusion matrix values
        """
        class_index = self.class_map[class_name]
        class_confusion_matrix = self.confusion_matrix[class_index, ...]
        TP = class_confusion_matrix[self.TP_INDEX]
        TN = class_confusion_matrix[self.TN_INDEX]
        FP = class_confusion_matrix[self.FP_INDEX]
        FN = class_confusion_matrix[self.FN_INDEX]
        return TP, TN, FP, FN

    def calculate_accuracy(self, class_name: str):
        """
        function calculates accuracy of given class based on stored confusion matrix

        Accuracy = ( TP + TN) / (TP + TN + FP + FN)

        """
        TP, TN, FP, FN = self.read_matrix_values(class_name)
        denom = TP + TN + FP + FN
        if denom == 0.0:
            return None
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        return accuracy

    def calculate_iou(self, class_name: str):
        """
        function calculates iou of given class based on stored confusion matrix

        iou = ( TP ) / (TP + FP + FN)

        """
        TP, TN, FP, FN = self.read_matrix_values(class_name)
        denom = TP + FP + FN
        if denom == 0.0:
            return None
        iou = TP / denom
        return iou

    def calculate_precision(self, class_name: str):
        """
        function calculates precision of given class based on stored confusion matrix

        precision = ( TP ) / (TP + FP)

        """
        TP, TN, FP, FN = self.read_matrix_values(class_name)
        denom = TP + FP
        if denom == 0.0:
            return None
        precision = (TP) / (TP + FP)
        return precision

    def calculate_recal(self, class_name: str):
        """
        function calculates recall of given class based on stored confusion matrix

        precision = ( TP ) / (TP + FN)

        """
        TP, TN, FP, FN = self.read_matrix_values(class_name)
        denom = TP + FN
        if denom == 0.0:
            return None
        recall = (TP) / (TP + FN)
        return recall

    def calculate_average_iou(self, include_unlabeled=False):
        """
        function calculates mean intersection of union for all classes

        """
        ious = []
        for i in list(self.class_map.keys()):
            iou = self.calculate_iou(i)
            if iou is not None:
                if (i == "unlabeled") and (include_unlabeled):
                    ious.append(iou)
                elif i != "unlabeled":
                    ious.append(iou)
        miou = np.mean(ious)
        return miou