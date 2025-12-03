import torch
from flow_matching.path import ProbPath
from torch.utils.data import DataLoader
from flow_matching.path.path_sample import PathSample

from src.flow_matching.controller.smart_logger import SmartLogger
from src.flow_matching.model.losses import ConditionalFMLoss

DEVICE_CPU = torch.device("cpu")

class TimelessTrainer:
    def __init__(self, model, optimizer, path: ProbPath, num_epochs, device=DEVICE_CPU, verbose=True, monitoring_int=50):
        self.model = model.to(device)
        self.path = path
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.device = device
        self.criterion = ConditionalFMLoss()
        self.logger = SmartLogger(verbose=verbose)
        self.monitoring_int = monitoring_int

    def training_loop(self, loader: DataLoader):
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.log_device_and_params(self.device, num_params)
        self.logger.log_training_start()
        for epoch in range(self.num_epochs):
            self.logger.log_epoch(epoch)
            self._train(loader)
            self._validate(loader)
        self.logger.log_training_end()

    # noinspection PyUnresolvedReferences
    def _train(self, loader: DataLoader):
        self.model.train()
        for batch_id, (x_0, x_1) in enumerate(loader):
            x_0 = x_0.to(self.device)
            x_1 = x_1.to(self.device)
            batch_size = x_1.shape[0]
            t = torch.full((batch_size,), 0, device=self.device)
            sample: PathSample = self.path.sample(t=t, x_0=x_0, x_1=x_1)
            if batch_id % self.monitoring_int == 0:
                self.logger.add_training_sample(sample)
            x_t = sample.x_t
            dx_t = sample.dx_t  # dX_t is ψ˙ t(X0 | X1).
            dx_t = x_1 - x_0
            # If D is the Euclidean distance, the CFM objective corresponds to the mean-squared error
            loss = self.criterion(self.model(x_t, t), dx_t) # Monte Carlo estimate
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        # calculate epoch train loss
        samples = self.logger.request_training_samples()
        epoch_train_loss = self._compute_epoch_loss(samples)
        self.logger.log_epoch_train_loss(epoch_train_loss)

    # noinspection PyUnresolvedReferences
    def _validate(self, loader: DataLoader):
        for batch_id, (x_0, x_1) in enumerate(loader):
            x_0 = x_0.to(self.device)
            x_1 = x_1.to(self.device)
            batch_size = x_1.shape[0]
            if batch_id % self.monitoring_int == 0: # only validate on limited samples
                t = torch.rand(batch_size, device=self.device)  # Randomize time t ∼ U[0, 1]
                sample: PathSample = self.path.sample(t=t, x_0=x_0, x_1=x_1)
                self.logger.add_validation_sample(sample)
        # calculate epoch validation loss
        samples = self.logger.request_validation_samples()
        epoch_val_loss = self._compute_epoch_loss(samples)
        self.logger.log_epoch_val_loss(epoch_val_loss)

    def _compute_epoch_loss(self, samples):
        self.model.eval()
        total_loss = 0
        for sample in samples:
            loss = self.criterion(self.model(sample.x_t, sample.t), sample.dx_t)
            total_loss += loss.item()
        return total_loss / len(samples)