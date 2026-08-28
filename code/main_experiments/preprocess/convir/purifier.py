from __future__ import annotations

import os
from typing import Any

import torch
from torch.utils.data import Dataset

from .degradation import Degrader
from .restorer import ConvIRRestorer
from preprocess.purification.class_layout import (
    ClassLayout,
    SplitCLeanDataset,
    SplitDataset,
    image_root,
    nonSplitDataset,
    save_tensor_image,
)


class Purify:
    """Orchestrate degradation, ConvIR restoration, and image saving."""

    def __init__(self, args: Any, config: Any, type: str, dataset: Any) -> None:
        del config
        self.args = args
        self.type = type
        self.dataset = dataset
        self.layout = ClassLayout.from_dataset(dataset, getattr(args, "purified_label_style", "class_name"))
        degradation_type = getattr(args, "degradation_type", args.deg)
        degradation_strength = getattr(args, "degradation_strength", args.deg_scale)
        self.degrader = Degrader(degradation_type, degradation_strength, args.seed)
        self.restorer = None
        if self.args.use_conv_ir:
            self.restorer = ConvIRRestorer(
                version=getattr(args, "restorer_version", "base"),
                ckpt_path=getattr(args, "restorer_ckpt", ""),
                device=torch.device(args.gpu if torch.cuda.is_available() else "cpu"),
            )

    def _output_root(self) -> str:
        return image_root(self.args, self.type)

    def _save_sample(self, image: torch.Tensor, label: Any, index: int) -> None:
        sample_path = self.layout.sample_path(self._output_root(), label, index)
        os.makedirs(sample_path.directory, exist_ok=True)
        save_tensor_image(image, sample_path.path)

    def _purify_sample(self, image: torch.Tensor) -> torch.Tensor:
        degraded = self.degrader.degrade(image)
        if self.restorer is None:
            return degraded
        return self.restorer.restore(degraded)

    def pur(self) -> Dataset:
        root = self._output_root()
        os.makedirs(root, exist_ok=True)
        for index in range(len(self.dataset)):
            image, label = self.dataset[index]
            purified = self._purify_sample(image)
            self._save_sample(purified, label, index)

        if not self.args.concat:
            return nonSplitDataset(self.args, self.type)
        if self.type == "test":
            return SplitCLeanDataset(self.dataset, self.args, self.type)
        if self.type == "train":
            return SplitDataset(self.dataset, self.args, self.type)
        if self.type == "test_pois":
            return SplitDataset(self.dataset, self.args, self.type)
        raise ValueError(f"Unsupported purification type: {self.type}")
