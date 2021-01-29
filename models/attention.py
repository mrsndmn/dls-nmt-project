import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_dim: int, key_and_query_dim: int = 64, value_dim: int = 64):
        super(Attention, self).__init__()

        self.query_weights = nn.Linear(hidden_dim, key_and_query_dim)
        self.key_weights = nn.Linear(hidden_dim, key_and_query_dim)
        self.value_weights = nn.Linear(hidden_dim, value_dim)

        self.kq_dim_root = torch.sqrt(key_and_query_dim)

        self.softmax = nn.Softmax()

    def forward(self, k_hidden_inputs: torch.Tensor, q_hidden_inputs: torch.Tensor, v_hidden_inputs: torch.Tensor):

        keys: torch.Tensor = self.key_weights(k_hidden_inputs)
        queries: torch.Tensor = self.query_weights(q_hidden_inputs)
        values: torch.Tensor = self.value_weights(v_hidden_inputs)

        keys_transposed = keys.permute(0, 2, 1)
        scaled_kv = torch.mm(queries, keys_transposed) / self.kq_dim_root

        return torch.mm(self.softmax(scaled_kv), values)


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, key_and_query_dim: int = 64, value_dim: int = 64, num_heads=8):
        super(MultiHeadAttention, self).__init__()

        attentions = [Attention(hidden_dim, key_and_query_dim=key_and_query_dim, value_dim=value_dim)
                      for _ in range(num_heads)]
        self.attention_heads = nn.ModuleList(attentions)

        self.heads_weights = nn.Linear(num_heads * value_dim, hidden_dim)

    def forward(self, k_hidden_inputs: torch.Tensor, q_hidden_inputs: torch.Tensor, v_hidden_inputs: torch.Tensor):

        attention_outputs = torch.cat([ attention(k_hidden_inputs, q_hidden_inputs, v_hidden_inputs) for attention in self.attention_heads ], dim=1)

        return self.heads_weights(attention_outputs)

class SimpleMultiHeadAttention(MultiHeadAttention):
    '''
    The same as MultiHeadAttention but all the query key and value inputs are the same
    '''
    def __init__(self, hidden_dim: int, key_query_value_dim: int = 64, num_heads=8):
        super(SimpleMultiHeadAttention, self).__init__(hidden_dim, key_and_query_dim=key_query_value_dim, value_dim=key_query_value_dim, num_heads=num_heads=)
        return

    def forward(self, hidden_inputs):
        return super(SimpleMultiHeadAttention, self).forward(k_hidden_inputs=hidden_inputs, q_hidden_inputs=hidden_inputs, v_hidden_inputs=hidden_inputs)


class AddAndNorm(nn.Module):
    def __init__(self, hidden_dim: int, sublayer: nn.Module):
        super(AddAndNorm, self).__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.sublayer = sublayer
        return

    def forward(self, inputs):
        return self.norm(inputs + self.sublayer(inputs))

