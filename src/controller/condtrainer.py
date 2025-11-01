import torch
from flow_matching.path import ProbPath
from torch.utils.data import DataLoader
from flow_matching.path.path_sample import PathSample

from src.model.losses import ConditionalFMLoss

class CondTrainer:
    def __init__(self, model, path: ProbPath, optimizer):
        self.model = model
        self.path = path
        self.optimizer = optimizer
        self.criterion = ConditionalFMLoss()

    def train(self, loader: DataLoader):
        batch_size = loader.batch_size
        for x_0, x_1 in loader:
            t = torch.rand(batch_size)  # Randomize time t ∼ U[0, 1]
            sample: PathSample = self.path.sample(t=t, x_0=x_0, x_1=x_1)
            x_t = sample.x_t
            dx_t = sample.dx_t  # dX_t is ψ˙ t(X0 | X1).
            # If D is the Euclidean distance, the CFM objective corresponds to the mean-squared error
            loss = self.criterion(self.model(x_t, t), dx_t) # Monte Carlo estimate
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()