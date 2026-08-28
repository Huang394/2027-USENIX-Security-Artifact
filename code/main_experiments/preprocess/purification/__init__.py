from .datasets import FolderBackedDataset, SplitCLeanDataset, SplitDataset, nonSplitDataset
from .router import build_purifier


class Purify:
    def __init__(self, args, config, type, dataset):
        self.impl = build_purifier(args, config, type, dataset)

    def pur(self):
        return self.impl.pur()


__all__ = ["Purify", "FolderBackedDataset", "SplitCLeanDataset", "SplitDataset", "nonSplitDataset", "build_purifier"]
