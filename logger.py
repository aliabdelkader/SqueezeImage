from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from metric import MetricsCalculator


class Logger:
    def __init__(self, logging_dir="logs"):

        self.logging_dir = logging_dir
        self.writer = SummaryWriter(self.logging_dir, flush_secs=1)

    def add_scalers(self, scope, scalers, step):
        self.writer.add_scalars(scope, scalers, global_step=step)
        # self.writer.flush()

    def add_scaler(self, scope, tag, value, step):
        self.writer.add_scalar(tag=scope + '/' + tag, scalar_value=value, global_step=step)
        # self.writer.flush()

    def add_graph(self, model, input):
        self.writer.add_graph(model, input)
        # self.writer.flush()

    def add_image(self, tag, image_tensor):
        self.writer.add_image(tag=tag, img_tensor=image_tensor)
        # self.writer.flush()

    def add_metrics(self, metrics_calcuator: MetricsCalculator, step=0):
        """
        function logs metrics per class in tensorboard
        """
        for c in metrics_calcuator.class_map.keys():
            accuracy = metrics_calcuator.calculate_accuracy(c)
            if not accuracy:
                accuracy = 0

            iou = metrics_calcuator.calculate_iou(c)
            if not iou:
                iou = 0

            precision = metrics_calcuator.calculate_precision(c)
            if not precision:
                precision = 0

            recall = metrics_calcuator.calculate_recal(c)
            if not recall:
                recall = 0

            TP, TN, FP, FN = metrics_calcuator.read_matrix_values(c)

            # self.add_scalers(scope=c,
            # scalers={"true_positive": TP,
            #     "true_negative": TN,
            #     "false_positive": FP,
            #     "false_negative": FN,
            #     "accuracy": accuracy,
            #     "iou": iou,
            #     "precision": precision,
            #     "recall": recall
            #  },
            #  step=step)

            self.add_scaler(scope="true_positive", tag=c, value=TP, step=step)
            self.add_scaler(scope="true_negative", tag=c, value=TN, step=step)
            self.add_scaler(scope="false_positive", tag=c, value=FP, step=step)
            self.add_scaler(scope="false_negative", tag=c, value=FN, step=step)
            self.add_scaler(scope="accuracy", tag=c, value=accuracy, step=step)
            self.add_scaler(scope="iou", tag=c, value=iou, step=step)
            self.add_scaler(scope="precision", tag=c, value=precision, step=step)
            self.add_scaler(scope="recall", tag=c, value=recall, step=step)
