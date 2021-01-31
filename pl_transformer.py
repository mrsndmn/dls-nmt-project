import os, math, random
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from torchnlp.encoders.text.text_encoder import BatchedSequences
from torchnlp.utils import lengths_to_mask

from models import transformer


class TransformerLightningModule(pl.LightningModule):
    def __init__(
        self,
        src_vocab_size: int,
        trg_vocab_size: int,
        hidden_dim: int=256,
        num_blocks: int=6,
        key_query_value_dim: int=32,
        padding_token_idx: int = 0,
        smoothing: float = 0.1,
        lr: float = 1e-4,
        noam_opt_warmup_steps: int= 4000,
    ):

        super(TransformerLightningModule, self).__init__()

        self.save_hyperparameters()

        self.transformer = transformer.Transformer(src_vocab_size, trg_vocab_size, hidden_dim,
                                                   num_blocks=num_blocks,
                                                   key_query_value_dim=key_query_value_dim)

        self.criterion = transformer.LabelSmoothing(trg_vocab_size, padding_token_idx=padding_token_idx, smoothing=smoothing)

        return


    def training_step(self, batch, batch_idx):
        src_padded_tokens: BatchedSequences
        trg_padded_tokens: BatchedSequences
        src_padded_tokens, trg_padded_tokens = batch

        src_tokens: torch.Tensor = src_padded_tokens.tensor
        src_mask: torch.Tensor = lengths_to_mask(src_padded_tokens.lengths, device=src_tokens.device)

        assert src_mask.size() == src_padded_tokens.tensor.size()

        trg_tokens: torch.Tensor = trg_padded_tokens.tensor
        trg_mask: torch.Tensor = lengths_to_mask(trg_padded_tokens.lengths, device=src_tokens.device)
        # целевые токены будем определять так:
        # просто рандомное число, которое будет меньше seq_len'а каждого предложения.
        # с учетом этого и будем формировать маску
        target_token_positions = []
        for i, seq_len in enumerate(list(trg_padded_tokens.lengths.numpy())):
            assert seq_len > 1
            visible_length = random.randint(1, seq_len-1)
            trg_mask[i, visible_length:] = False

        # todo how to determine target token?
        target_token_idxs: torch.Tensor = trg_tokens[torch.arange(trg_tokens.size(0)), target_token_positions]

        transformer_output = self.transformer.forward(src_tokens, trg_tokens, src_mask=src_mask, trg_mask=trg_mask)
        trg_tokens_probabilities = self.transformer.generator.forward(transformer_output)

        loss = self.criterion(trg_tokens_probabilities, target_token_idxs)

        self.log("loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        pass


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--src_vocab_size", type=int, help="")
        parser.add_argument("--trg_vocab_size", type=int, help="")
        parser.add_argument("--hidden_dim", type=int, help="")
        parser.add_argument("--num_blocks", type=int, help="")
        parser.add_argument("--key_query_value_dim", type=int, help="")

        parser.add_argument("--lr", type=float)

        # parser.add_argument("--num_workers", type=int, default=8)
        # parser.add_argument("--data_dir", type=str, default=".")

        return parser

    def noam_opt(self, current_step: int):
        min_inv_sqrt = min(1/math.sqrt(current_step+1), current_step / math.sqrt(self.hparams.noam_opt_warmup_steps))
        return min_inv_sqrt / math.sqrt(self.hparams.hidden_dim)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.transformer.parameters(), lr=self.hparams.lr)
        opt_sched = torch.optim.lr_scheduler.LambdaLR(opt, self.noam_opt)
        return [opt], [{"scheduler": opt_sched}]

# copy-paste https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/master/pl_bolts/models/autoencoders/basic_vae/basic_vae_module.py
def cli_main(args=None):
    from datamodules.wmt import WMTDataModule

    pl.seed_everything()

    parser = ArgumentParser()

    # todo support other datamodules
    dm_cls = WMTDataModule()

    parser = TransformerLightningModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args(args)

    dm = dm_cls.from_argparse_args(args)
    dm.download_dataset()
    dm.bpe_tokenize()


    if args.max_steps == -1:
        args.max_steps = None

    transformer_model = TransformerLightningModule(dm.src_bpe.vocab_size(), dm.trg_bpe.vocab_size(), 512)

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(transformer_model, datamodule=dm)
    return dm, transformer_model, trainer


if __name__ == "__main__":
    dm, model, trainer = cli_main()