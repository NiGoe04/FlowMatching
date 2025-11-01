import torch
from torch import nn


class ConditionalFMLoss(nn.Module):
    def __init__(self):
        super(ConditionalFMLoss, self).__init__()

    def forward(self, pred_velocity, gt_velocity):
        return torch.pow(pred_velocity - gt_velocity, 2).mean()