
import pytorch_lightning as pl

from torchtext.datasets.translation import WMT14


class YaCorpusDataModule(pl.LightningDataModule):
    def __init__(self):
        super(YaCorpusDataModule, self).__init__()
        return

    def setup(self, stage):
        raise NotImplementedError