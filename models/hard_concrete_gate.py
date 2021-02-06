import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class HardConcreteGate(nn.Module):
    def __init__(self,
                 num_heads,
                 log_a=0.0,
                 temperature=0.5,
                 adjust_range=(-0.1, 1.1),
                 l0_penalty_lambda=0.0,
                 l2_penalty_lambda=0.0,
                 eps=1e-9,
                 ):
        super(HardConcreteGate, self).__init__()

        self.log_a = nn.Parameter(torch.full((num_heads,), log_a))
        self.eps = eps

        self.register_buffer("temperature", torch.Tensor([temperature]))
        self.register_buffer("adjust_range", torch.Tensor(adjust_range))

        self.register_buffer("random_buffer", torch.rand(num_heads), persistent=False)
        self.l0_penalty = torch.zeros(1)
        self.l2_penalty = torch.zeros(1)

        self.sigmoid = nn.Sigmoid()

        self.l0_penalty_lambda = l0_penalty_lambda
        self.l2_penalty_lambda = l2_penalty_lambda

        return

    # batch_size, seq_len, num_heads * v_dim
    def forward(self, inputs:torch.Tensor) -> torch.Tensor:

        assert inputs.size(-1) % self.log_a.size(0) == 0

        if self.training:
            torch.rand(self.random_buffer.size(), out=self.random_buffer) # avoid extra allocations

            one_minus_rand_log = (1 - self.random_buffer).log_()
            concrete = self.sigmoid((self.log_a + self.random_buffer.log() + one_minus_rand_log) / self.temperature)
        else:
            concrete = self.sigmoid(self.log_a)

        concrete = concrete * (self.adjust_range[1] - self.adjust_range[0]) + self.adjust_range[0]
        concrete = torch.clip(concrete, min=0, max=1)

        if self.training and (self.l0_penalty_lambda > 0 or self.l2_penalty_lambda > 0):
            p_open = self.sigmoid(self.log_a - self.temperature * torch.log(- self.adjust_range[0] / self.adjust_range[1]) )
            p_open = torch.clip(p_open, min=self.eps, max=1-self.eps)

            if self.l0_penalty_lambda > 0:
                self.l0_penalty = self.l0_penalty_lambda * p_open

            if self.l2_penalty_lambda > 0:
                self.l2_penalty = self.l2_penalty_lambda * p_open * torch.pow(inputs, 2).sum()

        repeat_cnt = inputs.size(-1) // self.log_a.size(0)
        concrete = torch.repeat_interleave(concrete, repeat_cnt, dim=-1)

        return inputs * concrete