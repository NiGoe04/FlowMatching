import torch
from torch.utils.data import Dataset

class Coupling(Dataset):
    def __init__(self, x0_tensor, x1_tensor):
        self.x0 = x0_tensor
        self.x1 = x1_tensor

    def __len__(self):
        return len(self.x0)

    def __getitem__(self, idx):
        return self.x0[idx], self.x1[idx]

class Coupler:
    def __init__(self, x0_tensor, x1_tensor):
        super().__init__()
        assert len(x0_tensor) == len(x1_tensor)
        self.x0 = x0_tensor
        self.x1 = x1_tensor

    def get_independent_coupling(self):
        perm = torch.randperm(len(self.x1))
        x1_shuffled = self.x1[perm]
        return Coupling(self.x0, x1_shuffled)