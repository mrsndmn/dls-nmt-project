import torch
import torch.nn as nn

from models.encoder import TransformerEncoderBlock
from models.decoder import TransformerDecoderBlock


class Transformer(nn.Module):
    def __init__(self, hidden_dim: int, num_blocks: int = 6):

        self.num_blocks = num_blocks

        encoder_blocks = [TransformerEncoderBlock(
            hidden_dim) for _ in range(num_blocks)]
        decoder_blocks = [TransformerDecoderBlock(
            hidden_dim) for _ in range(num_blocks)]

        self.encoder_blocks = nn.Sequential(encoder_blocks)
        self.decoder_blocks = nn.ModuleList(decoder_blocks)

    def forward(self, inputs_tokens, target_tokens):
        inputs_encoded = self.encoder_blocks(inputs_tokens)
        decoder_output = target_tokens
        for decoder_block in range(self.decoder_blocks):
            decoder_output = decoder_block(decoder_output, inputs_encoded)

        return decoder_output
