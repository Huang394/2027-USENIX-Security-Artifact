from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF  # type: ignore[import-untyped]

IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg")


@dataclass(frozen=True)
class PurifiedSample:
    path: str
    label: Any


@dataclass(frozen=True)
class SamplePath:
    directory: str
    filename: str

    @property
    def path(self) -> str:
        return os.path.join(self.directory, self.filename)


def label_value(label: Any) -> int:
    if isinstance(label, torch.Tensor):
        return int(label.item())
    return int(label)


def extract_label_pair(label: Any) -> Tuple[int, int]:
    if not isinstance(label, dict):
        value = label_value(label)
        return value, value
    orig = label.get("label_orig", label.get("orig"))
    pois = label.get("label_pois", label.get("pois"))
    if orig is None or pois is None:
        raise KeyError("Poisoned labels must contain label_orig and label_pois")
    return label_value(orig), label_value(pois)


def dataset_classes(dataset: Any) -> list[str]:
    classes = list(getattr(dataset, "classes", []))
    if classes:
        return classes
    class_to_idx = getattr(dataset, "class_to_idx", {})
    if class_to_idx:
        return [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
    return []


def dataset_class_to_idx(dataset: Any, root_dir: str, poisoned: bool) -> dict[str, int]:
    class_to_idx = dict(getattr(dataset, "class_to_idx", {}))
    label_names: set[str] = set()
    root_path = Path(root_dir)
    if root_path.exists():
        for class_dir in sorted(path for path in root_path.iterdir() if path.is_dir()):
            label_names.add(class_dir.name)
            if poisoned:
                label_names.update(path.name for path in class_dir.iterdir() if path.is_dir())
    non_numeric = sorted(name for name in label_names if not name.isdigit())
    if not class_to_idx:
        return {name: index for index, name in enumerate(non_numeric)}

    missing_labels = [name for name in non_numeric if name not in class_to_idx]
    if not missing_labels:
        return class_to_idx

    root_class_to_idx = {name: index for index, name in enumerate(non_numeric)}
    if not any(name in class_to_idx for name in non_numeric):
        return root_class_to_idx
    return {**root_class_to_idx, **class_to_idx}


def class_index(label_name: str, class_to_idx: dict[str, int]) -> int:
    if label_name in class_to_idx:
        return class_to_idx[label_name]
    if label_name.isdigit():
        return int(label_name)
    raise KeyError(f"Class folder is not present in class_to_idx: {label_name}")


def label_folder_name(label: int, classes: list[str], style: str) -> str:
    if style == "numeric" or not classes:
        return str(label)
    return classes[label]


class ClassLayout:
    def __init__(self, classes: list[str], style: str) -> None:
        self.classes = classes
        self.style = style

    @classmethod
    def from_dataset(cls, dataset: Any, style: str) -> ClassLayout:
        return cls(dataset_classes(dataset), style)

    def folder_name(self, label: int) -> str:
        return label_folder_name(label, self.classes, self.style)

    def sample_path(self, root: str, label: Any, index: int) -> SamplePath:
        if isinstance(label, dict):
            orig_label, pois_label = extract_label_pair(label)
            orig_folder = self.folder_name(orig_label)
            pois_folder = self.folder_name(pois_label)
            return SamplePath(
                directory=os.path.join(root, orig_folder, pois_folder),
                filename=f"{orig_folder}_{pois_folder}_{index}.png",
            )

        class_folder = self.folder_name(label_value(label))
        return SamplePath(
            directory=os.path.join(root, class_folder),
            filename=f"{class_folder}_{index}.png",
        )


def image_root(args: Any, split: str) -> str:
    if split == "train":
        return args.image_folder
    if split == "test":
        return args.test_image_folder
    if split == "test_pois":
        return args.test_image_folder_pois
    raise ValueError(f"Unsupported split: {split}")


def save_tensor_image(tensor: torch.Tensor, path: str) -> None:
    image = TF.to_pil_image(torch.clamp(tensor.detach().cpu(), 0.0, 1.0))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)


def _image_paths(root_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for extension in IMAGE_EXTENSIONS:
        paths.extend(sorted(root_dir.rglob(extension)))
    return paths


def collect_clean_records(root_dir: str, class_to_idx: dict[str, int]) -> list[PurifiedSample]:
    records: list[PurifiedSample] = []
    root_path = Path(root_dir)
    for class_dir in sorted(path for path in root_path.iterdir() if path.is_dir()):
        for image_path in _image_paths(class_dir):
            records.append(PurifiedSample(path=str(image_path), label=class_index(class_dir.name, class_to_idx)))
    return records


def collect_poisoned_records(root_dir: str, class_to_idx: dict[str, int]) -> list[PurifiedSample]:
    records: list[PurifiedSample] = []
    root_path = Path(root_dir)
    for orig_dir in sorted(path for path in root_path.iterdir() if path.is_dir()):
        for pois_dir in sorted(path for path in orig_dir.iterdir() if path.is_dir()):
            label = {
                "label_orig": class_index(orig_dir.name, class_to_idx),
                "label_pois": class_index(pois_dir.name, class_to_idx),
            }
            for image_path in _image_paths(pois_dir):
                records.append(PurifiedSample(path=str(image_path), label=label))
    return records


def inferred_classes(root_dir: str, class_to_idx: dict[str, int]) -> list[str]:
    if class_to_idx:
        return [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
    root_path = Path(root_dir)
    if not root_path.exists():
        return []
    return sorted(path.name for path in root_path.iterdir() if path.is_dir())


class FolderBackedDataset(Dataset):
    def __init__(self, root_dir: str, poisoned: bool, dataset: Any = None) -> None:
        self.root_dir = root_dir
        self.poisoned = poisoned
        self.class_to_idx = dataset_class_to_idx(dataset, root_dir, poisoned)
        self.classes = dataset_classes(dataset) or inferred_classes(root_dir, self.class_to_idx)
        self.records = (
            collect_poisoned_records(root_dir, self.class_to_idx)
            if poisoned
            else collect_clean_records(root_dir, self.class_to_idx)
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Any]:
        record = self.records[idx]
        image = Image.open(record.path).convert("RGB")
        return TF.to_tensor(image), record.label


class SplitDataset(FolderBackedDataset):
    def __init__(self, dataset: Any, args: Any, type: str = "train", transform: Optional[Any] = None) -> None:
        del transform
        super().__init__(image_root(args, type), poisoned=True, dataset=dataset)


class SplitCLeanDataset(FolderBackedDataset):
    def __init__(self, dataset: Any, args: Any, type: str = "train", transform: Optional[Any] = None) -> None:
        del transform
        super().__init__(image_root(args, type), poisoned=False, dataset=dataset)


class nonSplitDataset(FolderBackedDataset):
    def __init__(self, args: Any, type: str = "train") -> None:
        root_dir = image_root(args, type)
        poisoned = False
        root_path = Path(root_dir)
        if root_path.exists():
            for child in root_path.iterdir():
                if child.is_dir() and any(grandchild.is_dir() for grandchild in child.iterdir()):
                    poisoned = True
                    break
            if type == "test_pois":
                poisoned = True
        super().__init__(root_dir, poisoned=poisoned)


__all__ = [
    "ClassLayout",
    "FolderBackedDataset",
    "PurifiedSample",
    "SamplePath",
    "SplitCLeanDataset",
    "SplitDataset",
    "class_index",
    "collect_clean_records",
    "collect_poisoned_records",
    "dataset_class_to_idx",
    "dataset_classes",
    "extract_label_pair",
    "image_root",
    "inferred_classes",
    "label_folder_name",
    "label_value",
    "nonSplitDataset",
    "save_tensor_image",
]
