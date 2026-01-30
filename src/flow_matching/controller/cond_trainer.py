import torch
import statistics

from flow_matching.path import ProbPath
from torch.utils.data import DataLoader
from flow_matching.path.path_sample import PathSample

from src.flow_matching.view.logger import Logger
from src.flow_matching.model.coupling import Coupler
from src.flow_matching.model.losses import ConditionalFMLoss, MACWeightedLoss, TensorCost
import torch.nn.functional as F

DEVICE_CPU = torch.device("cpu")

class CondTrainer:
    def __init__(self, model, optimizer, path: ProbPath, num_epochs, num_val_samples, device=DEVICE_CPU, verbose=True):
        self.model = model.to(device)
        self.path = path
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.num_val_samples = num_val_samples
        self.device = device
        self.criterion = ConditionalFMLoss()
        self.logger = Logger(verbose=verbose)
        self.train_loss_values = []
        self.val_x_t = None
        self.val_t = None
        self.val_dx_t = None

    def training_loop(self, loader: DataLoader):
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.log_device_and_params(self.device, num_params)
        self.logger.log_training_start()
        self._precompute_validation_samples(loader)
        for epoch in range(self.num_epochs):
            self.logger.log_epoch(epoch)
            self._train(loader)
            self._validate()
        self.logger.log_training_end()

    def _train(self, loader: DataLoader):
        self.train_loss_values.clear()
        self.model.train()
        for x_0, x_1 in loader:
            batch_size = x_1.shape[0]
            x_0 = x_0.to(self.device)
            x_1 = x_1.to(self.device)

            t = torch.rand(batch_size, device=self.device)  # Randomize time t ∼ U[0, 1]
            sample: PathSample = self.path.sample(t=t, x_0=x_0, x_1=x_1)
            x_t = sample.x_t
            dx_t = sample.dx_t  # dX_t is ψ˙ t(X0 | X1).
            # If D is the Euclidean distance, the CFM objective corresponds to the mean-squared error
            loss = self.criterion(self.model(x_t, t), dx_t) # Monte Carlo estimate
            self.train_loss_values.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        # calculate epoch train loss
        epoch_train_loss = statistics.mean(self.train_loss_values)
        self.logger.log_epoch_train_loss(epoch_train_loss)

    def _validate(self):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(self.val_x_t, self.val_t)
            loss = self.criterion(pred, self.val_dx_t)
        self.logger.log_epoch_validation_loss(loss.item())

    def _precompute_validation_samples(self, loader: DataLoader):
        x_t_list = []
        t_list = []
        dx_t_list = []
        num_collected = 0
        for x_0, x_1 in loader:
            batch_size = x_0.shape[0]
            if num_collected >= self.num_val_samples:
                break

            x_0 = x_0.to(self.device)
            x_1 = x_1.to(self.device)
            t = torch.rand(batch_size, device=self.device)
            sample: PathSample = self.path.sample(t=t, x_0=x_0, x_1=x_1)

            x_t_list.append(sample.x_t.detach())
            t_list.append(sample.t.detach())
            dx_t_list.append(sample.dx_t.detach())

            num_collected += batch_size

        self.val_x_t = torch.cat(x_t_list, dim=0).to(self.device)
        self.val_t = torch.cat(t_list, dim=0).to(self.device)
        self.val_dx_t = torch.cat(dx_t_list, dim=0).to(self.device)

class CondTrainerMAC(CondTrainer):
    def __init__(self, model, optimizer, path: ProbPath, num_epochs, num_val_samples,
                 top_k_percentage, mac_reg_coefficient, device=DEVICE_CPU):
        super().__init__(model, optimizer, path, num_epochs, num_val_samples, device=device)
        self.criterion_mac = MACWeightedLoss(mac_reg_coefficient)
        self.top_k_percentage = top_k_percentage

    def _train(self, loader: DataLoader):
        self.train_loss_values.clear()
        self.model.train()
        for x_0, x_1 in loader:
            batch_size = x_1.shape[0]
            x_0_train = x_0.to(self.device)
            x_1_train = x_1.to(self.device)
            threshold = int(self.top_k_percentage * batch_size)

            model_pred_error = self._mac_prediction_error(x_0_train, x_1_train)
            sorted_idx = torch.argsort(model_pred_error)
            error_ranks = torch.empty_like(sorted_idx, device=self.device)
            error_ranks[sorted_idx] = torch.arange(len(sorted_idx), device=self.device)

            t = torch.rand(batch_size, device=self.device)
            sample: PathSample = self.path.sample(t=t, x_0=x_0_train, x_1=x_1_train)
            x_t = sample.x_t
            dx_t = sample.dx_t
            loss = self.criterion_mac(self.model(x_t, t), dx_t, error_ranks, threshold)
            self.train_loss_values.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        # calculate epoch train loss
        epoch_train_loss = statistics.mean(self.train_loss_values)
        self.logger.log_epoch_train_loss(epoch_train_loss)

    def _mac_prediction_error(self, x_0, x_1):
        self.model.eval()
        with torch.no_grad():
            pred0 = self.model(x_0, torch.zeros(len(x_0), device=x_0.device))
            pred1 = self.model(x_1, torch.ones(len(x_1), device=x_1.device))

            target = x_1 - x_0

            err0 = F.mse_loss(pred0, target, reduction="none").mean(dim=1)
            err1 = F.mse_loss(pred1, target, reduction="none").mean(dim=1)

            pred_error = 0.5 * (err0 + err1)

        self.model.train()
        return pred_error

class CondTrainerBatchOT(CondTrainer):
    def __init__(self, model, optimizer, path: ProbPath, num_epochs, num_val_samples, device=DEVICE_CPU):
        super().__init__(model, optimizer, path, num_epochs, num_val_samples, device=device)
        self.ot_cost = TensorCost.quadratic_cost

    def _train(self, loader: DataLoader):
        self.train_loss_values.clear()
        self.model.train()
        for x_0, x_1 in loader:
            batch_size = x_1.shape[0]
            coupler_ot = Coupler(x_0, x_1)
            coupling_ot = coupler_ot.get_n_ot_coupling(batch_size, self.ot_cost)
            x_0_train = coupling_ot.x0.to(self.device)
            x_1_train = coupling_ot.x1.to(self.device)

            t = torch.rand(batch_size, device=self.device)  # Randomize time t ∼ U[0, 1]
            sample: PathSample = self.path.sample(t=t, x_0=x_0_train, x_1=x_1_train)
            x_t = sample.x_t
            dx_t = sample.dx_t  # dX_t is ψ˙ t(X0 | X1).
            # If D is the Euclidean distance, the CFM objective corresponds to the mean-squared error
            loss = self.criterion(self.model(x_t, t), dx_t)  # Monte Carlo estimate
            self.train_loss_values.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        # calculate epoch train loss
        epoch_train_loss = statistics.mean(self.train_loss_values)
        self.logger.log_epoch_train_loss(epoch_train_loss)

    def _precompute_validation_samples(self, loader: DataLoader):
        x_t_list = []
        t_list = []
        dx_t_list = []
        num_collected = 0

        for x_0, x_1 in loader:
            if num_collected >= self.num_val_samples:
                break
            batch_size = x_0.shape[0]
            coupler_ot = Coupler(x_0, x_1)
            coupling_ot = coupler_ot.get_n_ot_coupling(batch_size, self.ot_cost)

            x_0_ot = coupling_ot.x0.to(self.device)
            x_1_ot = coupling_ot.x1.to(self.device)
            t = torch.rand(batch_size, device=self.device)
            sample: PathSample = self.path.sample(t=t, x_0=x_0_ot, x_1=x_1_ot)

            x_t_list.append(sample.x_t.detach())
            t_list.append(sample.t.detach())
            dx_t_list.append(sample.dx_t.detach())

            num_collected += batch_size

        self.val_x_t = torch.cat(x_t_list, dim=0).to(self.device)
        self.val_t = torch.cat(t_list, dim=0).to(self.device)
        self.val_dx_t = torch.cat(dx_t_list, dim=0).to(self.device)
