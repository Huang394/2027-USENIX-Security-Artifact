from __future__ import annotations

import os
from typing import Any

import torch
from torch.utils.data import Dataset

from preprocess.purification.class_layout import ClassLayout, FolderBackedDataset, image_root, save_tensor_image


class Purify:
    """Export the input dataset unchanged and return it as a folder-backed dataset."""

    def __init__(self, args: Any, config: Any, type: str, dataset: Dataset) -> None:
        del config
        self.args = args
        self.type = type
        self.dataset = dataset
        self.classes = list(getattr(dataset, "classes", []))
        self.class_to_idx = dict(getattr(dataset, "class_to_idx", {}))
        self.layout = ClassLayout.from_dataset(dataset, getattr(args, "purified_label_style", "class_name"))
        if not self.classes:
            raise AttributeError("none backend requires the input dataset to expose a classes attribute")
        if not self.class_to_idx:
            self.class_to_idx = {class_name: index for index, class_name in enumerate(self.classes)}

    def _output_root(self) -> str:
        return image_root(self.args, self.type)

    def _save_sample(self, image: torch.Tensor, label: Any, index: int) -> None:
        sample_path = self.layout.sample_path(self._output_root(), label, index)
        save_tensor_image(image, sample_path.path)

    def pur(self) -> Dataset:
        root = self._output_root()
        os.makedirs(root, exist_ok=True)
        for index in range(len(self.dataset)):
            image, label = self.dataset[index]
            self._save_sample(image, label, index)

        poisoned = self.type in {"train", "test_pois"}
        return FolderBackedDataset(root, poisoned=poisoned, dataset=self.dataset)


__all__ = ["Purify"]
