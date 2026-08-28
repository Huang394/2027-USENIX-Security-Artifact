"""Dataset loading and paired sampling."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sized, cast

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


@dataclass(frozen=True)
class SampleBatch:
    clean: torch.Tensor
    poisoned: torch.Tensor
    labels: torch.Tensor
    paths: list[str]
    indices: list[int]


class FlatImageDataset(Dataset[tuple[torch.Tensor, int, str]]):
    """Flat image dataset with labels read from labels.csv."""

    def __init__(self, root: Path, transform: Callable[[Image.Image], torch.Tensor]) -> None:
        labels_path = root / "labels.csv"
        if not labels_path.exists():
            raise FileNotFoundError(f"Expected labels.csv in flat dataset: {labels_path}")
        labels: dict[str, int] = {}
        with labels_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            if "filename" not in fieldnames or "label" not in fieldnames:
                raise ValueError("labels.csv must contain filename,label columns")
            for row in reader:
                labels[row["filename"]] = int(row["label"])
        self.items = [
            (path, labels[path.name])
            for path in sorted(root.iterdir())
            if path.suffix.lower() in IMAGE_EXTENSIONS and path.name in labels
        ]
        if not self.items:
            raise ValueError(f"No labelled images found in {root}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        path, label = self.items[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label, str(path)

def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ],
    )


def load_image_dataset(path: Path, image_size: int) -> Dataset[tuple[torch.Tensor, int, str]]:
    transform = build_transform(image_size)
    has_class_dirs = any(child.is_dir() for child in path.iterdir())
    if has_class_dirs:
        image_folder = datasets.ImageFolder(path, transform=transform)

        class WrappedImageFolder(Dataset[tuple[torch.Tensor, int, str]]):
            def __len__(self) -> int:
                return len(image_folder)

            def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
                image, label = image_folder[index]
                return image, int(label), image_folder.samples[index][0]

        return WrappedImageFolder()
    return FlatImageDataset(path, transform)


def sample_paired_batch(
    clean_dataset: Dataset[tuple[torch.Tensor, int, str]],
    poisoned_dataset: Dataset[tuple[torch.Tensor, int, str]],
    num_samples: int,
    true_class_filter: int | None,
) -> SampleBatch:
    limit = min(len(cast(Sized, clean_dataset)), len(cast(Sized, poisoned_dataset)))
    clean_images: list[torch.Tensor] = []
    poisoned_images: list[torch.Tensor] = []
    labels: list[int] = []
    paths: list[str] = []
    indices: list[int] = []
    for index in range(limit):
        clean, label, path = clean_dataset[index]
        poisoned, poison_label, _ = poisoned_dataset[index]
        if label != poison_label:
            raise ValueError(
                f"Paired datasets disagree at index {index}: clean={label}, poisoned={poison_label}"
            )
        if true_class_filter is not None and label != true_class_filter:
            continue
        clean_images.append(clean)
        poisoned_images.append(poisoned)
        labels.append(label)
        paths.append(path)
        indices.append(index)
        if len(labels) >= num_samples:
            break
    if not labels:
        raise ValueError("No paired samples selected; check dataset paths and filters")
    return SampleBatch(
        clean=torch.stack(clean_images),
        poisoned=torch.stack(poisoned_images),
        labels=torch.tensor(labels, dtype=torch.long),
        paths=paths,
        indices=indices,
    )


def sample_dataset_by_indices(
    dataset: Dataset[tuple[torch.Tensor, int, str]],
    indices: list[int],
    expected_labels: torch.Tensor,
    name: str,
) -> torch.Tensor:
    """Load a dataset using the primary clean/poisoned selected indices."""

    images: list[torch.Tensor] = []
    for position, index in enumerate(indices):
        image, label, _path = dataset[index]
        expected = int(expected_labels[position].item())
        if label != expected:
            raise ValueError(
                f"{name} disagrees at selected position {position}: expected={expected}, got={label}"
            )
        images.append(image)
    return torch.stack(images)
