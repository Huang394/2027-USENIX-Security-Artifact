from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class ExportedPoisonedDataset(Dataset):
    """Read exported poisoned images stored as split/orig_label/pois_label/*.png."""

    def __init__(self, root_dir: str) -> None:
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise RuntimeError(f"Exported poisoned dataset not found: {self.root_dir}")
        self.records: list[tuple[Path, dict[str, int]]] = []
        class_names: set[str] = set()
        for orig_dir in sorted(path for path in self.root_dir.iterdir() if path.is_dir()):
            for pois_dir in sorted(path for path in orig_dir.iterdir() if path.is_dir()):
                label = {"label_orig": int(orig_dir.name), "label_pois": int(pois_dir.name)}
                class_names.add(orig_dir.name)
                class_names.add(pois_dir.name)
                image_paths = (
                    sorted(pois_dir.rglob("*.png"))
                    + sorted(pois_dir.rglob("*.jpg"))
                    + sorted(pois_dir.rglob("*.jpeg"))
                )
                self.records.extend((image_path, label) for image_path in image_paths)
        if not self.records:
            raise RuntimeError(f"No exported poisoned images found under: {self.root_dir}")
        self.classes = sorted(class_names, key=int)
        self.class_to_idx = {name: int(name) for name in self.classes}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, Any]:
        image_path, label = self.records[index]
        image = Image.open(image_path).convert("RGB")
        return TF.to_tensor(image), label
