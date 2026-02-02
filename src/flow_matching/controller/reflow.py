import torch
from flow_matching.solver import ODESolver
from torch.utils.data import DataLoader

from src.flow_matching.controller.cond_trainer import CondTrainer
from src.flow_matching.model.coupling import Coupling
from src.flow_matching.view.utils import plot_tensor_2d


class TrainerReflow:
    def __init__(self, trainer: CondTrainer, coupling: Coupling, batch_size, learning_rate, reflow_order):
        self.trainer = trainer
        self.coupling: Coupling = coupling
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.reflow_order = reflow_order
        self.logger = trainer.logger
        self.device = trainer.device
        self.solver_steps = 128.0

    def training_loop_reflow(self):
        current_model = self.trainer.model
        for i in range(self.reflow_order):
            self.logger.log_reflow_iteration(i)
            loader = DataLoader(
                self.coupling,
                self.batch_size,
                shuffle=True,
            )
            self.trainer.precompute_validation_samples(loader)
            self.trainer.training_loop(loader)
            solver = ODESolver(velocity_model=self.trainer.model)
            x0 = self.coupling.x0.to(self.device)
            x_1_reflow = solver.sample(x_init=x0, method='midpoint',
                                       step_size=1.0 / self.solver_steps)
            plot_tensor_2d(x_1_reflow) # to do remove
            coupling_reflow = Coupling(x0, x_1_reflow)
            self.coupling = coupling_reflow
            current_model = self.trainer.model
            # update instances of the trainer
            new_model = type(current_model)(device=self.device)
            new_optimizer = torch.optim.Adam(new_model.parameters(), self.learning_rate)
            self.trainer.model = new_model
            self.trainer.optimizer = new_optimizer
        return current_model