import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class HardConcreteGate(nn.Module):
    def __init__(self, shape, log_a=0.0, temperature=0.5, adjust_range=(-0.1, 1.1)):
        super(HardConcreteGate, self).__init__()

        self.log_a = nn.Parameter(torch.Tensor([log_a]))

        self.register_buffer("temperature", torch.Tensor([temperature]))
        self.register_buffer("adjust_range", torch.Tensor(adjust_range))

        self.register_buffer("random_buffer", torch.rand(shape), persistent=False) # https://github.com/pytorch/pytorch/issues/237
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs:torch.Tensor) -> torch.Tensor:

        if self.training:
            torch.rand(self.random_buffer.size(), out=self.random_buffer) # avoid extra allocations

            one_minus_rand_log = (1 - self.random_buffer).log_()
            concrete = self.sigmoid((self.log_a + self.random_buffer.log() + one_minus_rand_log) / self.temperature)
        else:
            concrete = self.sigmoid(self.log_a)

        concrete = concrete * (self.adjust_range[1] - self.adjust_range[0]) + self.adjust_range[0]
        concrete = torch.clip(concrete, min=0, max=1)

        return inputs * concrete