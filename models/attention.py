import torch
import torch.nn as nn

import math

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

    def forward(self, k_hidden_inputs: torch.Tensor, q_hidden_inputs: torch.Tensor, v_hidden_inputs: torch.Tensor, mask=None, save_attention=False):
        """
        {kqv}_hidden_inputs: batch_size, seq_len, model_hid_dim
        """

        keys: torch.Tensor = self.key_weights(k_hidden_inputs)      # bs, seq_len, k_dim
        queries: torch.Tensor = self.query_weights(q_hidden_inputs) # bs, seq_len, q_dim
        values: torch.Tensor = self.value_weights(v_hidden_inputs)  # bs, seq_len, v_dim

        keys_transposed = keys.permute(0, 2, 1) # bs, k_dim, seq_len
        scaled_kv = torch.matmul(queries, keys_transposed) / self.kq_dim_root # # bs, seq_len, seq_len
        if mask is not None:
            scaled_kv[mask == 0] = self.neg_inf

        scaled_kv = self.softmax(scaled_kv)

        if save_attention:
            self.attention = scaled_kv

        return torch.matmul(scaled_kv, values) # bs, seq_len, v_dim


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, key_and_query_dim: int = 64, value_dim: int = 64, num_heads=8):
        super(MultiHeadAttention, self).__init__()

        if hidden_dim // num_heads != key_and_query_dim:
            raise ValueError(f"hidden_dim must be equal to num_heads * key_and_query_dim. Got: hidden_dim={hidden_dim} // num_heads={num_heads} != key_and_query_dim={key_and_query_dim}")

        attentions = [Attention(hidden_dim, key_and_query_dim=key_and_query_dim, value_dim=value_dim)
                      for _ in range(num_heads)]
        self.attention_heads = nn.ModuleList(attentions)

        self.heads_weights = nn.Linear(num_heads * value_dim, hidden_dim)

    def forward(self, k_hidden_inputs: torch.Tensor, q_hidden_inputs: torch.Tensor, v_hidden_inputs: torch.Tensor, mask=None):

        # bs, seq_len, v_dim * num_heads
        attention_outputs = torch.cat([attention(
            k_hidden_inputs, q_hidden_inputs, v_hidden_inputs, mask=mask) for attention in self.attention_heads], dim=-1)

        return self.heads_weights(attention_outputs) # bs, seq_len, hidd_dim


class SimpleMultiHeadAttention(MultiHeadAttention):
    '''
    The same as MultiHeadAttention but all the query key and value inputs are the same
    '''

    def __init__(self, hidden_dim: int, key_query_value_dim: int = 64, num_heads=8):
        super(SimpleMultiHeadAttention, self).__init__(hidden_dim, key_and_query_dim=key_query_value_dim, value_dim=key_query_value_dim, num_heads=num_heads)
        return

    def forward(self, hidden_inputs, mask=None):
        return super(SimpleMultiHeadAttention, self).forward(k_hidden_inputs=hidden_inputs, q_hidden_inputs=hidden_inputs, v_hidden_inputs=hidden_inputs, mask=mask)


class AddAndNorm(nn.Module):
    def __init__(self, hidden_dim: int, sublayer: nn.Module):
        super(AddAndNorm, self).__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.sublayer = sublayer
        return

    def forward(self, inputs, **kwargs):
        return self.norm(inputs + self.sublayer(inputs, **kwargs))
