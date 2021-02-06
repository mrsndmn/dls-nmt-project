import math

import torch
import torch.nn as nn
from torchnlp.utils import lengths_to_mask

from models.encoder import TransformerEncoderBlock, EncoderBlocksSequential
from models.decoder import TransformerDecoderBlock, DecoderBlocksSequential

from datamodules.wmt import TransformerBatchedSequencesWithMasks

# copy-paste from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class NormedEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super(NormedEmbeddings, self).__init__()
        self.emb = nn.Embedding(vocab_size, hidden_dim)
        self.hidden_dim_sqrt = math.sqrt(hidden_dim)

    def forward(self, x):
        return self.emb(x) * self.hidden_dim_sqrt

class Transformer(nn.Module):
    def __init__(self, src_vocab_size: int, trg_vocab_size: int, hidden_dim: int, num_blocks: int = 6, key_query_value_dim=64, num_heads: int = None):
        super(Transformer, self).__init__()

        self.trg_vocab_size = trg_vocab_size
        self.src_vocab_size = src_vocab_size
        self.hidden_dim = hidden_dim

        if num_heads is None:
            num_heads = hidden_dim // key_query_value_dim

        if num_heads <= 0:
            raise ValueError('num_heads must be positive')

        self.input_embeddings = nn.Sequential(
            NormedEmbeddings(src_vocab_size, hidden_dim),
            PositionalEncoding(hidden_dim),
        )
        self.target_embeddings = nn.Sequential(
            NormedEmbeddings(trg_vocab_size, hidden_dim),
            PositionalEncoding(hidden_dim),
        )

        self.num_blocks = num_blocks

        encoder_blocks = [TransformerEncoderBlock(
            hidden_dim, key_query_value_dim=key_query_value_dim, num_heads=num_heads) for _ in range(num_blocks)]
        decoder_blocks = [TransformerDecoderBlock(
            hidden_dim, key_query_value_dim=key_query_value_dim, num_heads=num_heads) for _ in range(num_blocks)]

        self.encoder_blocks = EncoderBlocksSequential(encoder_blocks)
        self.decoder_blocks = DecoderBlocksSequential(decoder_blocks, hidden_dim=hidden_dim)

        self.generator = TransformerGenerator(hidden_dim, trg_vocab_size)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_tokens, trg_tokens, src_mask=None, trg_mask=None):
        # todo можно обучать так, чтобы один выход энкодера соответствовал
        # нескольким прогонам в декодере, на каждый токен
        # так данные будут более эффективно исползованы и уменьшится кол-во вызовов энкодера

        src_embeddings = self.input_embeddings(src_tokens)
        trg_embeddings = self.target_embeddings(trg_tokens)

        encoder_outputs = self.encoder_blocks.forward(
            src_embeddings, src_mask=src_mask)
        decoder_output = self.decoder_blocks.forward(
            trg_embeddings, encoder_outputs, src_mask=src_mask, trg_mask=trg_mask)

        return decoder_output # batch_size x seq_len x hidden_dim

    def _build_seq_eos_mask(self, tokens: torch.Tensor, eos_id=3, curr_pos_in_seq=0):
        """
        маскирует токены, которые идут после eos токена
        """

        current_max_seq_len = tokens.size(1)
        lengths = []
        for seq in tokens:
            eos_indexes = torch.nonzero(seq == eos_id)
            if eos_indexes.size(0) == 0:
                lengths.append(current_max_seq_len)
            else:
                current_len = eos_indexes[0,0]
                lengths.append(current_len)
        assert len(lengths) == tokens.size(0)

        mask: torch.Tensor = lengths_to_mask(lengths, device=tokens.device)
        return mask


    def encode_decode(self, src_padded: torch.LongTensor, src_mask = torch.BoolTensor, src_pad_idx=0, trg_pad_idx=0, trg_bos_id=2, trg_eos_id=3, max_len=None):
        """
        Greedy decoder
        """

        # print(src_padded.dtype, src_padded.size())
        src_embeddings = self.input_embeddings(src_padded)
        encoder_outputs = self.encoder_blocks.forward(src_embeddings, src_mask=src_mask)

        # batch_size x max_len
        trg_tokens = torch.full_like(src_padded, trg_pad_idx, device=src_padded.device, dtype=src_padded.dtype)
        trg_tokens[:, 0] = trg_bos_id

        batch_size = trg_tokens.size(0)
        seq_len = trg_tokens.size(1)

        trui_tensor = (torch.triu(torch.ones((trg_tokens.size(0), seq_len, seq_len), device=trg_tokens.device), diagonal=1) == 0)
        # trg_mask = torch.full_like(src_mask, False)
        for i in range(1, trg_tokens.size(1)):
            curr_trg = trg_tokens[:, :i]
            curr_trui_tensor = trui_tensor[:, :i, :i]

            pad_idx = 0
            trg_mask = (curr_trg != pad_idx).unsqueeze(-2) & curr_trui_tensor

            # print("trg_tokens", trg_tokens[0, i-1])
            trg_embeddings = self.target_embeddings(curr_trg)

            # print("curr_trg", curr_trg.size())
            # print("trg_embeddings.size()", trg_embeddings.size())
            # print('trg_mask', trg_mask.size())
            # print("encoder_outputs", encoder_outputs.size())
            # print("src_mask", src_mask.size())
            # есть предположение, что на этот шаг вообще не влияет trg_embeddings
            # надо смотреть в аттеншн, что там происходит, возможно там все зануляется
            decoder_output = self.decoder_blocks.forward(
                trg_embeddings, encoder_outputs, src_mask=src_mask, trg_mask=trg_mask)

            assert decoder_output.size() == torch.Size((batch_size, i, self.hidden_dim)), f"decoder_output.size(){decoder_output.size()} == torch.Size((batch_size, i, self.trg_vocab_size)){torch.Size((batch_size, i, self.trg_vocab_size))}"

            last_current_token_decoded = decoder_output[:, -1, :]
            # print("last_current_token_decoded", last_current_token_decoded[0, :2])
            # batch_size x trg_vocab_size
            trg_tokens_probabilities = self.generator.forward(last_current_token_decoded)
            _, next_tokens = torch.max(trg_tokens_probabilities, dim=1)

            print("next_token", next_tokens)

            trg_tokens[:, i] = next_tokens

        return trg_tokens




class TransformerGenerator(nn.Module):
    def __init__(self, hidden_dim, trg_vocab_size):
        super(TransformerGenerator, self).__init__()

        self.out_linear = nn.Sequential(
            nn.Linear(hidden_dim, trg_vocab_size),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, transformer_output):
        # transformer_output: # batch_size, seq_len, hidden_dim
        return self.out_linear(transformer_output)  # batch_size, seq_len, trg_vocab_size


# copy-paste http://nlp.seas.harvard.edu/2018/04/03/attention.html#decoder
class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, trg_vocab_size, padding_token_idx: int = 0, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_token_idx = padding_token_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.trg_vocab_size = trg_vocab_size
        self.true_dist = None

    def forward(self, trg_tokens_logprobas, target_token_idxs, save_true_dist=False):
        # trg_tokens_logprobas: batch_size x seq_len x vocab_dim
        # target_token_idxs: batch_size x seq_len
        # print(trg_tokens_logprobas.shape, target_token_idxs.shape)
        assert trg_tokens_logprobas.size(2) == self.trg_vocab_size # vocab size
        assert trg_tokens_logprobas.size(1) == target_token_idxs.size(1) # seq len
        assert trg_tokens_logprobas.size(0) == target_token_idxs.size(0) # batch size

        smooth_value = self.smoothing / (self.trg_vocab_size - 1) # -2 for padding
        # batch_size x seq_len x vocab_dim
        true_dist = torch.full_like(trg_tokens_logprobas, smooth_value, device=trg_tokens_logprobas.device, dtype=trg_tokens_logprobas.dtype)
        #                       # bs, seq_len, 1
        # print('scatter_indexes', scatter_indexes.size())
        true_dist.scatter_(2, target_token_idxs.unsqueeze(2), self.confidence)
        true_dist[:, :, self.padding_token_idx] = 0

        mask = target_token_idxs == self.padding_token_idx
        true_dist[mask] = 0

        if save_true_dist:
            self.true_dist = true_dist
        return self.criterion(trg_tokens_logprobas, torch.autograd.Variable(true_dist, requires_grad=False))
