
import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch

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
    ):

        super(TransformerLightningModule, self).__init__()

        self.transformer = transformer.Transformer(src_vocab_size, trg_vocab_size, hidden_dim,
                                                   num_blocks=num_blocks,
                                                   key_query_value_dim=key_query_value_dim)

        self.criterion = transformer.LabelSmoothing(trg_vocab_size, padding_token_idx=padding_token_idx, smoothing=smoothing)

        return


    def training_step(self, batch, batch_idx):
        src_tokens: torch.Tensor = batch.src_tokens
        src_mask: torch.Tensor = batch.src_mask

        trg_tokens: torch.Tensor = batch.trg_tokens
        trg_mask: torch.Tensor = batch.trg_mask

        target_token_idxs: torch.Tensor = batch.target_token_idxs

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

        parser.add_argument("--lr", type=float, default=1e-4)

        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--data_dir", type=str, default=".")

        return parser


# copy-paste https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/master/pl_bolts/models/autoencoders/basic_vae/basic_vae_module.py
def cli_main(args=None):
    from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule, STL10DataModule
    from datamodules import WMT14DataModule

    pl.seed_everything()

    parser = ArgumentParser()
    # parser.add_argument("--dataset", default="cifar10", type=str, choices=["cifar10", "stl10", "imagenet"])
    script_args, _ = parser.parse_known_args(args)

    # todo
    if script_args.dataset == "cifar10":
        dm_cls = CIFAR10DataModule
    elif script_args.dataset == "stl10":
        dm_cls = STL10DataModule
    elif script_args.dataset == "imagenet":
        dm_cls = ImagenetDataModule
    else:
        raise ValueError(f"undefined dataset {script_args.dataset}")

    parser = TransformerLightningModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args(args)

    dm = dm_cls.from_argparse_args(args)
    args.input_height = dm.size()[-1]

    if args.max_steps == -1:
        args.max_steps = None

    model = TransformerLightningModule(**vars(args))

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule=dm)
    return dm, model, trainer


if __name__ == "__main__":
    dm, model, trainer = cli_main()