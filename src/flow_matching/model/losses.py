from torch import nn
import torch.nn.functional as F

class ConditionalFMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_velocity, gt_velocity):
        return F.mse_loss(pred_velocity, gt_velocity)