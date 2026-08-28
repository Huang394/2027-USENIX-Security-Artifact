import argparse
import shutil
from pathlib import Path
from typing import Sequence

from PIL import Image
from torchvision.datasets import CIFAR10  # type: ignore[import-untyped]

CIFAR10_FILES = (
    "batches.meta",
    "data_batch_1",
    "data_batch_2",
    "data_batch_3",
    "data_batch_4",
    "data_batch_5",
    "test_batch",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CIFAR-10 batches to ImageFolder layout.")
    parser.add_argument("--source-root", type=Path, default=Path("datasets"))
    parser.add_argument("--output-root", type=Path, default=Path("datasets") / "CIFAR10ImageFolder")
    parser.add_argument("--clean", action="store_true", help="Remove output-root before exporting.")
    parser.add_argument("--download", action="store_true", help="Download CIFAR-10 if source files are missing.")
    return parser.parse_args()


def normalize_source_root(source_root: Path) -> Path:
    source_root = source_root.expanduser()
    if source_root.name == CIFAR10.base_folder:
        return source_root.parent
    return source_root


def missing_files(source_root: Path) -> list[Path]:
    cifar_dir = source_root / CIFAR10.base_folder
    return [cifar_dir / filename for filename in CIFAR10_FILES if not (cifar_dir / filename).exists()]


def format_paths(paths: Sequence[Path]) -> str:
    return "\n".join(f"  - {path}" for path in paths)


def export_split(dataset: CIFAR10, split_dir: Path) -> None:
    for class_name in dataset.classes:
        (split_dir / class_name).mkdir(parents=True, exist_ok=True)

    for index, (array, target) in enumerate(zip(dataset.data, dataset.targets)):
        class_name = dataset.classes[int(target)]
        image = Image.fromarray(array)
        image.save(split_dir / class_name / f"{index:05d}.png")


def main() -> int:
    args = parse_args()
    source_root = normalize_source_root(args.source_root)
    if args.clean and args.output_root.exists():
        shutil.rmtree(args.output_root)

    missing = missing_files(source_root)
    if missing and not args.download:
        expected_dir = source_root / CIFAR10.base_folder
        raise FileNotFoundError(
            "CIFAR-10 batch files were not found in the expected torchvision layout.\n"
            f"Expected directory: {expected_dir}\n"
            f"Missing files:\n{format_paths(missing)}\n"
            "Pass --source-root as the parent directory of cifar-10-batches-py, "
            "or pass --download if this machine can download the dataset."
        )

    train_dataset = CIFAR10(root=str(source_root), train=True, download=args.download)
    test_dataset = CIFAR10(root=str(source_root), train=False, download=args.download)

    export_split(train_dataset, args.output_root / "train")
    export_split(test_dataset, args.output_root / "val")

    print(f"Exported {len(train_dataset)} train images and {len(test_dataset)} val images to {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
