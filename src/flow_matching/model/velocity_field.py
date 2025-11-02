from flow_matching.utils import ModelWrapper
import torch
from torch import nn
class SimpleVelocityField(ModelWrapper):
    def __init___(self, dim: int = 2, h: int = 64):
        super().__init__(self)
        self.net = nn.Sequential(
            nn.Linear(dim + 1, h),  # +1 for time t
            nn.ELU(),
            nn.Linear(h, h),
            nn.ELU(),
            nn.Linear(h, dim)  # output velocity same shape as x
        )

    def forward(self, x_t, t, **extras):
        t = t.view(-1, 1)  # ensure shape is (batch, 1)
        xt_input = torch.cat((x_t, t), dim=-1)
        return self.net(xt_input)