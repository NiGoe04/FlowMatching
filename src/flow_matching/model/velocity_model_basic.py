from flow_matching.utils import ModelWrapper
import torch
import torch.nn as nn

class SimpleVelocityField(nn.Module):
    def __init__(self, dim, h):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, h),  # +1 for time t
            nn.ELU(),
            nn.Linear(h, h),
            nn.ELU(),
            nn.Linear(h, h),
            nn.ELU(),
            nn.Linear(h, dim)  # output velocity same shape as x
        )

    def forward(self, x, t, **extras):
        t = t.view(-1, 1).expand(*x.shape[:-1], -1)  # ensure shape is (batch, 1)
        xt_input = torch.cat((x, t), dim=-1)
        return self.net(xt_input)

class SimpleVelocityModel(ModelWrapper):
    def __init__(self, device, dim: int = 2, h: int = 64):
        model = SimpleVelocityField(dim, h).to(device)
        super().__init__(model)