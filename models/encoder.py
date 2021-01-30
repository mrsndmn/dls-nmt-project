import torch
import torch.nn as nn
from typing import List

from models.attention import AddAndNorm, SimpleMultiHeadAttention


class TransformerEncoderBlock(nn.Module):
    '''
    param: hidden_dim - embedding hidden dim
    '''

    def __init__(self, hidden_dim: int, key_query_value_dim: int = 64, feed_forward_hidden_dim: int = 2048, num_heads=8):
        super(TransformerEncoderBlock, self).__init__()

        multihead_attention = SimpleMultiHeadAttention(
            hidden_dim, key_query_value_dim=key_query_value_dim, num_heads=num_heads)

        feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, feed_forward_hidden_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_hidden_dim, hidden_dim),
        )

        self.self_attention = AddAndNorm(hidden_dim, multihead_attention)
        self.feed_forward = AddAndNorm(hidden_dim, feed_forward)

        return

    def forward(self, inputs, src_mask=None):
        outputs = self.self_attention(inputs, mask=src_mask)
        outputs = self.feed_forward(outputs)
        return outputs


class EncoderBlocksSequential(nn.Module):
    def __init__(self, encoder_blocks: List[TransformerEncoderBlock]):
        super(EncoderBlocksSequential, self).__init__()
        self.encoder_blocks = nn.ModuleList(encoder_blocks)

    def forward(self, inputs, src_mask=None, trg_mask=None):
        encoder_output = inputs
        for encoder_block in self.encoder_blocks:
            encoder_output = encoder_block(encoder_output, src_mask=src_mask)

        return encoder_output
