import torch
from torch import Tensor

class Distribution:
    def __init__(self, base_distribution: Tensor, device):
        super().__init__()
        self.tensor = base_distribution.to(device)

    def add_uniform_noise(self, noise_bound=0.05):
        # uniform noise in [-noise_bound, +noise_bound]
        uniform_noise = (torch.rand_like(self.tensor) * 2 - 1) * noise_bound
        self.tensor += uniform_noise

    def add_gaussian_noise(self, variance=0.05):
        # Gaussian noise with mean 0 and specified variance
        std = variance ** 0.5
        gaussian_noise = torch.randn_like(self.tensor) * std
        self.tensor += gaussian_noise

    @staticmethod
    def get_uni_distribution(center, n_samples, device):
        center_tensor = torch.tensor(center, dtype=torch.float32)
        tensor = center_tensor.unsqueeze(0).repeat(n_samples, 1)
        return Distribution(tensor, device)
