from typing import List
from flow_matching.path import PathSample

from src.flow_matching.view.logger import Logger

class SmartLogger:
    def __init__(self, verbose):
        super().__init__()
        self.logger = Logger(verbose=verbose)
        self.training_path_samples: List[PathSample] = []
        self.validation_path_samples: List[PathSample] = []

    def log_training_start(self):
        self.logger.log_training_start()

    def log_training_end(self):
        self.logger.log_training_end()

    def log_epoch(self, num_epoch):
        self.logger.log_epoch(num_epoch)

    def add_training_sample(self, sample: PathSample):
        self.training_path_samples.append(sample)

    def add_validation_sample(self, sample: PathSample):
        self.validation_path_samples.append(sample)

    def request_training_samples(self):
        return self.training_path_samples

    def request_validation_samples(self):
        return self.validation_path_samples

    def log_epoch_train_loss(self, loss):
        self.logger.log_epoch_train_loss(loss)

    def log_epoch_val_loss(self, loss):
        self.logger.log_epoch_validation_loss(loss)