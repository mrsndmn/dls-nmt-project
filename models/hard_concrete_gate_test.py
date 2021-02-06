import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hard_concrete_gate import HardConcreteGate

import pytest

def test_hard_concrete_gate():

    num_heads = 8
    hcg_shape = (2, 3, num_heads * 3);
    hcg = HardConcreteGate(num_heads, log_a=0.0)

    hcg_input = torch.rand(hcg_shape)
    hcg_out = hcg.forward(hcg_input)

    assert hcg_out.size() == hcg_input.size()

    learned_params_count = sum( p.numel() for p in hcg.parameters() if p.requires_grad)
    hgc_state = hcg.state_dict()
    print(hgc_state)
    params_and_buffers_elems_count = sum( hgc_state[state].numel() for state in hgc_state )

    assert learned_params_count == num_heads
    assert params_and_buffers_elems_count == num_heads + 3 # log_a.numel() + len([temp, adjust_range_start, adjust_range_end])

