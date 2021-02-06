import torch
import torch.nn as nn

import math

from models.hard_concrete_gate import HardConcreteGate

class Attention(nn.Module):
    def __init__(self, hidden_dim: int, key_and_query_dim: int = 64, value_dim: int = 64):
        super(Attention, self).__init__()

        self.query_weights = nn.Linear(hidden_dim, key_and_query_dim)
        self.key_weights = nn.Linear(hidden_dim, key_and_query_dim)
        self.value_weights = nn.Linear(hidden_dim, value_dim)

        self.kq_dim_root = math.sqrt(key_and_query_dim)

        self.softmax = nn.Softmax(dim=-1)
        self.neg_inf = -1e9

        self.attention: torch.Tensor = None

    def forward(self, q_hidden_inputs: torch.Tensor, k_hidden_inputs: torch.Tensor, v_hidden_inputs: torch.Tensor, mask=None, save_attention=False):
        """
        {kqv}_hidden_inputs: batch_size, seq_len, model_hid_dim
        """

        queries: torch.Tensor = self.query_weights(q_hidden_inputs) # bs, q_seq_len, q_dim
        keys: torch.Tensor = self.key_weights(k_hidden_inputs)      # bs, k_seq_len, k_dim
        values: torch.Tensor = self.value_weights(v_hidden_inputs)  # bs, v_seq_len, v_dim

        batch_size = queries.size(0)
        q_seq_len = queries.size(1)
        k_seq_len = keys.size(1)
        v_seq_len = values.size(1)
        # print("q_seq_len", q_seq_len)
        # print("k_seq_len", k_seq_len)
        # print("v_seq_len", v_seq_len)
        assert k_seq_len == v_seq_len, f'k_seq_len{k_seq_len} == v_seq_len{v_seq_len}'

        keys_transposed = keys.permute(0, 2, 1) # bs, k_dim, k_seq_len
        scaled_kv = torch.matmul(queries, keys_transposed) / self.kq_dim_root # # bs, q_seq_len, k_seq_len
        assert scaled_kv.size() == torch.Size((batch_size, q_seq_len, k_seq_len))

        if mask is not None:
            # print("scaled_kv.size()", scaled_kv.size())
            # print("mask.size()", mask.size())
            scaled_kv.masked_fill_(mask == False, self.neg_inf)

        scaled_kv = self.softmax(scaled_kv)

        if save_attention:
            self.attention = scaled_kv

        return torch.matmul(scaled_kv, values) # bs, q_seq_len, v_dim


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, key_and_query_dim: int = 64, value_dim: int = 64, num_heads=8, with_hard_concrete_gate=False, hcg_l0_penalty_lambda=0.0):
        super(MultiHeadAttention, self).__init__()

        if hidden_dim // num_heads != key_and_query_dim:
            raise ValueError(f"hidden_dim must be equal to num_heads * key_and_query_dim. Got: hidden_dim={hidden_dim} // num_heads={num_heads} != key_and_query_dim={key_and_query_dim}")

        self.with_hard_concrete_gate = with_hard_concrete_gate

        attentions = []
        concrete_gates = []
        for _ in range(num_heads):
            attentions.append(Attention(hidden_dim, key_and_query_dim=key_and_query_dim, value_dim=value_dim))
            if with_hard_concrete_gate:
                concrete_gates.append( HardConcreteGate(1, l0_penalty_lambda=hcg_l0_penalty_lambda) )

        self.attention_heads = nn.ModuleList(attentions)

        self.hard_concrete_gates = nn.ModuleList(concrete_gates)

        self.heads_weights = nn.Linear(num_heads * value_dim, hidden_dim)

    @property
    def num_heads(self):
        return len(self.attention_heads)

    def forward(self, q_hidden_inputs: torch.Tensor, k_hidden_inputs: torch.Tensor, v_hidden_inputs: torch.Tensor, mask=None):

        attention_outputs = []
        for attention in self.attention_heads:
            attention_output = attention(q_hidden_inputs, k_hidden_inputs, v_hidden_inputs, mask=mask)
            attention_outputs.append(attention_output)

        if self.with_hard_concrete_gate:
            for i, hcg in enumerate(self.hard_concrete_gates):
                attention_outputs[i] = hcg(attention_outputs[i])

        # bs, seq_len, v_dim * num_heads
        attention_outputs = torch.cat(attention_outputs, dim=-1)

        return self.heads_weights(attention_outputs) # bs, seq_len, hidd_dim


class SimpleMultiHeadAttention(MultiHeadAttention):
    '''
    The same as MultiHeadAttention but all the query key and value inputs are the same
    '''

    def __init__(self, hidden_dim: int, key_query_value_dim: int = 64, num_heads=8):
        super(SimpleMultiHeadAttention, self).__init__(hidden_dim, key_and_query_dim=key_query_value_dim, value_dim=key_query_value_dim, num_heads=num_heads)
        return

    def forward(self, hidden_inputs, mask=None):
        return super(SimpleMultiHeadAttention, self).forward(q_hidden_inputs=hidden_inputs, k_hidden_inputs=hidden_inputs, v_hidden_inputs=hidden_inputs, mask=mask)


class AddAndNorm(nn.Module):
    def __init__(self, hidden_dim: int, sublayer: nn.Module):
        super(AddAndNorm, self).__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.sublayer = sublayer
        return

    def forward(self, inputs, **kwargs):
        return self.norm(inputs + self.sublayer(inputs, **kwargs))
