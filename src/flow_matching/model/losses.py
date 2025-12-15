import torch
from torch import nn
import torch.nn.functional as F


class ConditionalFMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_velocity, gt_velocity):
        return F.mse_loss(pred_velocity, gt_velocity)

class TensorCost:
    @staticmethod
    def quadratic_cost(x0, x1):
        x0 = x0.view(x0.shape[0], -1)
        x1 = x1.view(x1.shape[0], -1)
        return torch.cdist(x0, x1, p=2) ** 2
