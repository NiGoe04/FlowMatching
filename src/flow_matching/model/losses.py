import torch
from torch import nn

class ConditionalFMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(pred_velocity, gt_velocity):
        return torch.pow(pred_velocity - gt_velocity, 2).mean()