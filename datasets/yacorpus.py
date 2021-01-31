

import torchtext.datasets.translation as translation


class YaCorpusDataset(translation.TranslationDataset):
    urls = []
    name = 'yacorpus'
    dirname = ''

    @classmethod
    def splits(cls, exts, fields, root='.data'):
        raise NotImplementedError
