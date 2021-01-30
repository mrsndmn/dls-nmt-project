import torch
import torch.nn as nn

from models.attention import AddAndNorm, SimpleMultiHeadAttention


class TransformerEncoderBlock(nn.Module):
    '''
    param: hidden_dim - embedding hidden dim
    '''

    def __init__(self, hidden_dim: int, key_query_value_dim: int = 64, num_attention_heads=8, feed_forward_hidden_dim: int = 2048, num_heads=8):
        super(TransformerEncoderBlock, self).__init__()

        multihead_attention = SimpleMultiHeadAttention(
            hidden_dim, key_query_value_dim=key_query_value_dim, num_heads=num_heads)

        feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, feed_forward_hidden_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_hidden_dim, hidden_dim),
        )

        self.model = nn.Sequential([
            AddAndNorm(hidden_dim, multihead_attention),
            AddAndNorm(hidden_dim, feed_forward),
        ])

        return

    def forward(self, inputs):
        return self.model(inputs)
