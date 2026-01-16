import torch
from torch.utils.data import Dataset
from scipy.optimize import linear_sum_assignment


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

    def get_multi_cond_coupling(self, n):
        coupled_x0 = []
        coupled_x1 = []
        for _ in range(n):
            perm = torch.randperm(len(self.x1))
            x1_shuffled = self.x1[perm]
            coupled_x0.append(self.x0)
            coupled_x1.append(x1_shuffled)

        x0_coupled = torch.cat(coupled_x0, dim=0)
        x1_coupled = torch.cat(coupled_x1, dim=0)

        perm = torch.randperm(len(x1_coupled))
        x0_coupled = x0_coupled[perm]
        x1_coupled = x1_coupled[perm]

        return Coupling(x0_coupled, x1_coupled)

    def get_n_ot_coupling(self, n, cost_fn):
        N = len(self.x0)

        perm = torch.randperm(len(self.x1))
        x1_shuffled = self.x1[perm]

        coupled_x0 = []
        coupled_x1 = []

        for start in range(0, N, n):
            end = min(start + n, N)

            x0_block = self.x0[start:end]
            x1_block = x1_shuffled[start:end]

            # Compute cost matrix (block_size x block_size)
            cost = cost_fn(x0_block, x1_block)

            # Solve OT via Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(
                cost.detach().cpu().numpy()
            )

            coupled_x0.append(x0_block[row_ind])
            coupled_x1.append(x1_block[col_ind])

        x0_coupled = torch.cat(coupled_x0, dim=0)
        x1_coupled = torch.cat(coupled_x1, dim=0)

        return Coupling(x0_coupled, x1_coupled)
