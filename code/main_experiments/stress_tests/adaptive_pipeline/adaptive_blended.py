from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.convir.degradation import Degrader  # noqa: E402
from preprocess.convir.network import build_net  # noqa: E402
from preprocess.purification import Purify  # noqa: E402
from preprocess.purification.datasets import FolderBackedDataset  # noqa: E402


DIFFERENTIABLE_DEGRADATIONS = {"none", "gaussian_noise", "haze", "blur", "motion_blur", "resize"}


@dataclass(frozen=True)
class StressConfig:
    dataset: str
    datasets_root_dir: str
    classes: int
    img_size: int
    attack_target: int
    poisoned_rate: float
    deterministic: bool
    classifier_ckpt: str
    degradation_type: str
    degradation_strength: float
    use_conv_ir: bool
    restorer_ckpt: str
    restorer_version: str
    alpha: float
    lambda_reg: float
    lr: float
    steps: int
    batch_size: int
    num_workers: int
    search_samples: int
    eval_samples: int
    eval_offset: int
    best_eval_samples: int
    best_eval_interval: int
    seed: int
    attack_seed: int
    optim_seed: int
    device: str
    trigger_channels: int
    output_dir: str
    save_pattern: bool
    adaptive_init: str
    saved_original_purified_pois_root: str
    saved_clean_purified_root: str
    require_main_asr_baseline: bool
    final_eval_via_main_pipeline: bool
    main_pipeline_output_dir: str
    purified_label_style: str


class DifferentiableConvIRRestorer(nn.Module):
    def __init__(self, version: str, ckpt_path: str, device: torch.device, min_size: int = 256) -> None:
        super().__init__()
        self.model = build_net(version).to(device)
        self.model.eval()
        self.min_size = min_size
        if ckpt_path:
            checkpoint = torch.load(ckpt_path, map_location=device)
            state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
            self.model.load_state_dict(state_dict, strict=False)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _padded_size(size: int, factor: int, min_size: int) -> int:
        target = max(size, min_size)
        return target + (factor - target % factor) % factor

    @classmethod
    def _pad_to_factor(
        cls,
        x: torch.Tensor,
        factor: int = 32,
        min_size: int = 256,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        height, width = x.shape[-2:]
        target_height = cls._padded_size(height, factor, min_size)
        target_width = cls._padded_size(width, factor, min_size)
        padded = x
        while padded.shape[-2] < target_height or padded.shape[-1] < target_width:
            current_height, current_width = padded.shape[-2:]
            remaining_height = target_height - current_height
            remaining_width = target_width - current_width
            pad_height = min(remaining_height, max(0, current_height - 1))
            pad_width = min(remaining_width, max(0, current_width - 1))
            padded = F.pad(padded, (0, pad_width, 0, pad_height), mode="reflect")
        return padded, (height, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padded, original_shape = self._pad_to_factor(x, min_size=self.min_size)
        restored = self.model(padded)[-1]
        restored = restored[..., : original_shape[0], : original_shape[1]]
        return torch.clamp(restored, 0.0, 1.0)


class PipelinePurifier(nn.Module):
    def __init__(self, config: StressConfig, device: torch.device) -> None:
        super().__init__()
        self.degrader = Degrader(config.degradation_type, config.degradation_strength, config.optim_seed)
        self.restorer: DifferentiableConvIRRestorer | None = None
        if config.use_conv_ir:
            self.restorer = DifferentiableConvIRRestorer(
                version=config.restorer_version,
                ckpt_path=config.restorer_ckpt,
                device=device,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        degraded = self.degrader.degrade(x)
        if self.restorer is None:
            return degraded
        return self.restorer(degraded)


def parse_args() -> StressConfig:
    parser = argparse.ArgumentParser(description="Adaptive blended trigger stress test against a fixed purifier.")
    parser.add_argument("--dataset", default="Imagenette2")
    parser.add_argument("--datasets_root_dir", default="datasets")
    parser.add_argument("--classes", type=int, default=20)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--attack_target", type=int, default=1)
    parser.add_argument("--poisoned_rate", type=float, default=0.05)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--classifier_ckpt", default="Imagenette2_pretrain/Blended/Res_34_256/ckpt.pth")
    parser.add_argument("--degradation_type", default="haze")
    parser.add_argument("--degradation_strength", type=float, default=0.5)
    parser.add_argument("--use_conv_ir", action="store_true")
    parser.add_argument("--restorer_ckpt", default="pretrain/ots-base.pkl")
    parser.add_argument("--restorer_version", default="base")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--lambda_reg", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--search_samples", type=int, default=1000, help="Use -1 for all eligible train samples.")
    parser.add_argument("--eval_samples", type=int, default=1000, help="Use -1 for all eligible eval samples.")
    parser.add_argument("--eval_offset", type=int, default=0)
    parser.add_argument("--best_eval_samples", type=int, default=256)
    parser.add_argument("--best_eval_interval", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1234, help="Backward-compatible default for both seeds.")
    parser.add_argument("--attack_seed", type=int, default=-1, help="Seed for original Blended attack generation.")
    parser.add_argument("--optim_seed", type=int, default=-1, help="Seed for adaptive search sampling/init.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trigger_channels", type=int, choices=[1, 3], default=1)
    parser.add_argument("--output_dir", default="stress_tests/adaptive_pipeline/results")
    parser.add_argument("--save_pattern", action="store_true")
    parser.add_argument(
        "--adaptive_init",
        choices=["original_blended", "random"],
        default="original_blended",
        help="Initial pattern for adaptive search.",
    )
    parser.add_argument(
        "--saved_original_purified_pois_root",
        default="",
        help="Optional existing purified poisoned-image folder from the main CA/ASR/PA pipeline.",
    )
    parser.add_argument(
        "--saved_clean_purified_root",
        default="",
        help="Optional existing purified clean-image folder from the main CA/ASR/PA pipeline.",
    )
    parser.add_argument(
        "--require_main_asr_baseline",
        action="store_true",
        help="Fail unless --saved_original_purified_pois_root is provided for a main-flow comparable baseline.",
    )
    parser.add_argument(
        "--final_eval_via_main_pipeline",
        action="store_true",
        help="Evaluate the learned adaptive trigger through Purify(...).pur() image output.",
    )
    parser.add_argument(
        "--main_pipeline_output_dir",
        default="stress_tests/adaptive_pipeline/main_pipeline_outputs",
        help="Output root for adaptive final evaluation via the main purification pipeline.",
    )
    parser.add_argument(
        "--purified_label_style",
        choices=["class_name", "numeric"],
        default="class_name",
        help="Folder label style used when exporting adaptive purified images through Purify.",
    )
    args = parser.parse_args()
    if args.attack_seed < 0:
        args.attack_seed = args.seed
    if args.optim_seed < 0:
        args.optim_seed = args.seed
    return StressConfig(**vars(args))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def import_process_dataset(config: StressConfig) -> Any:
    old_argv = sys.argv[:]
    sys.argv = [
        old_argv[0],
        "--dataset",
        config.dataset,
        "--attack_method",
        "Blended",
        "--datasets_root_dir",
        config.datasets_root_dir,
        "--classes",
        str(config.classes),
        "--img_size",
        str(config.img_size),
        "--attack_target",
        str(config.attack_target),
        "--poisoned_rate",
        str(config.poisoned_rate),
        "--seed",
        str(config.attack_seed),
        "--deterministic",
        str(config.deterministic),
        "--degradation_type",
        config.degradation_type,
        "--degradation_strength",
        str(config.degradation_strength),
        "--deg",
        config.degradation_type,
        "--deg_scale",
        str(config.degradation_strength),
        "--disable_run_log",
    ]
    try:
        module = importlib.import_module("attack.backdoorattack")
    finally:
        sys.argv = old_argv
    return module.process_dataset


def load_module(module_path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_classifier(config: StressConfig, device: torch.device) -> nn.Module:
    resnet_module = load_module(
        Path("attack") / "BackdoorBox" / "core" / "models" / "resnet.py",
        "stress_test_resnet",
    )
    model = resnet_module.ResNet(34, num_classes=config.classes)
    state_dict = torch.load(config.classifier_ckpt, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        model_keys = model.state_dict().keys()
        model_uses_module = all(key.startswith("module.") for key in model_keys)
        ckpt_uses_module = all(key.startswith("module.") for key in state_dict.keys())
        if model_uses_module and not ckpt_uses_module:
            state_dict = {f"module.{key}": value for key, value in state_dict.items()}
        elif ckpt_uses_module and not model_uses_module:
            state_dict = {
                key[len("module.") :] if key.startswith("module.") else key: value
                for key, value in state_dict.items()
            }
        model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def build_dataset(config: StressConfig, split: str) -> Dataset[Any]:
    transform = transforms.Compose(
        [
            transforms.Resize((config.img_size, config.img_size)),
            transforms.ToTensor(),
        ]
    )
    dataset_name = config.dataset.strip().lower()
    if dataset_name in {"cifar10", "cifar-10", "cifar"}:
        return datasets.CIFAR10(
            config.datasets_root_dir,
            train=split == "train",
            transform=transform,
            download=False,
        )

    split_name = "train" if split == "train" else "val"
    root = Path(config.datasets_root_dir) / config.dataset / split_name
    return datasets.ImageFolder(str(root), transform=transform)


def dataset_targets(dataset: Dataset[Any]) -> list[int]:
    targets = getattr(dataset, "targets", None)
    if targets is not None:
        return [int(target) for target in targets]
    samples = getattr(dataset, "samples", None)
    if samples is not None:
        return [int(sample[1]) for sample in samples]
    raise TypeError("Dataset does not expose targets or samples; cannot build eligible subset.")


def bounded_selection(indices: list[int], limit: int, offset: int) -> list[int]:
    if offset < 0:
        raise ValueError("--eval_offset must be non-negative.")
    if limit == -1:
        return indices[offset:]
    return indices[offset : offset + limit]


def eligible_subset(dataset: Dataset[Any], target_label: int, limit: int, offset: int = 0) -> Subset[Any]:
    eligible = [index for index, label in enumerate(dataset_targets(dataset)) if label != target_label]
    selected = bounded_selection(eligible, limit, offset)
    if not selected:
        raise ValueError(
            "No eligible non-target samples selected. "
            f"eligible={len(eligible)}, requested_limit={limit}, requested_offset={offset}. "
            "Lower --eval_offset or choose a different --attack_target."
        )
    return Subset(dataset, selected)


def saved_poisoned_subset(dataset: FolderBackedDataset, target_label: int, limit: int, offset: int = 0) -> Subset[Any]:
    eligible = []
    for index, record in enumerate(dataset.records):
        label = record.label
        if not isinstance(label, dict):
            continue
        label_orig = int(label["label_orig"])
        label_pois = int(label["label_pois"])
        if label_orig != label_pois and label_pois == target_label:
            eligible.append(index)
    selected = bounded_selection(eligible, limit, offset)
    if not selected:
        raise ValueError(
            "No eligible saved purified poisoned samples selected. "
            f"eligible={len(eligible)}, requested_limit={limit}, requested_offset={offset}, "
            f"target_label={target_label}."
        )
    return Subset(dataset, selected)


def clean_subset(dataset: Dataset[Any], limit: int, offset: int = 0) -> Subset[Any]:
    selected = bounded_selection(list(range(len(dataset))), limit, offset)
    if not selected:
        raise ValueError(
            "No clean samples selected. "
            f"dataset_len={len(dataset)}, requested_limit={limit}, requested_offset={offset}."
        )
    return Subset(dataset, selected)


def dataset_source(dataset: Dataset[Any]) -> Any:
    if isinstance(dataset, Subset):
        return dataset_source(dataset.dataset)
    return dataset


class AdaptiveBlendedEvalDataset(Dataset[Any]):
    def __init__(self, dataset: Dataset[Any], pattern: torch.Tensor, alpha: float, target_label: int) -> None:
        self.dataset = dataset
        self.pattern = pattern.detach().cpu()
        self.alpha = float(alpha)
        self.target_label = int(target_label)
        source = dataset_source(dataset)
        self.classes = list(getattr(source, "classes", []))
        self.class_to_idx = dict(getattr(source, "class_to_idx", {}))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, int]]:
        image, label = self.dataset[index]
        label_orig = int(label.item()) if isinstance(label, torch.Tensor) else int(label)
        triggered = torch.clamp((1.0 - self.alpha) * image + self.alpha * self.pattern, 0.0, 1.0)
        return triggered, {"label_orig": label_orig, "label_pois": self.target_label}


class DatasetSubsetWithAttrs(Subset[Any]):
    def __init__(self, dataset: Dataset[Any], indices: list[int]) -> None:
        super().__init__(dataset, indices)
        self.classes = list(getattr(dataset_source(dataset), "classes", []))
        self.class_to_idx = dict(getattr(dataset_source(dataset), "class_to_idx", {}))


def make_loader(dataset: Dataset[Any], config: StressConfig, shuffle: bool) -> DataLoader[Any]:
    generator = torch.Generator()
    generator.manual_seed(config.optim_seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def initial_pattern(config: StressConfig, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(config.optim_seed)
    shape = (config.trigger_channels, config.img_size, config.img_size)
    return torch.rand(shape, generator=generator, device=device)


def original_blended_pattern(config: StressConfig, device: torch.device) -> torch.Tensor:
    """Recreate the BackdoorBox Blended pattern used by attack.backdoorattack.

    The project builds Blended as a single-channel uint8 random pattern after
    seeding torch with attack_seed. BackdoorBox applies it before ToTensor, so the
    equivalent tensor-space pattern is pattern / 255.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.attack_seed)
    pattern = torch.randint(
        0,
        255,
        size=(1, config.img_size, config.img_size),
        dtype=torch.uint8,
        generator=generator,
    )
    pattern = pattern.float().div(255.0)
    if config.trigger_channels == 3:
        pattern = pattern.expand(3, -1, -1).clone()
    return pattern.to(device)


def adaptive_initial_pattern(config: StressConfig, device: torch.device) -> torch.Tensor:
    if config.adaptive_init == "original_blended":
        return original_blended_pattern(config, device)
    if config.adaptive_init == "random":
        return initial_pattern(config, device)
    raise ValueError(f"Unsupported adaptive initialization: {config.adaptive_init}")


def pattern_to_blended_uint8(pattern: torch.Tensor) -> torch.Tensor:
    return pattern.detach().cpu().mul(255.0).round().clamp(0, 255).to(torch.uint8)


def blended_weight_like(pattern_uint8: torch.Tensor, alpha: float) -> torch.Tensor:
    return torch.full(pattern_uint8.shape, float(alpha), dtype=torch.float32)


def build_adaptive_blended_poisoned_test(
    config: StressConfig,
    original_backdoor_instance: Any,
    clean_train: Dataset[Any],
    clean_test: Dataset[Any],
    adaptive_pattern: torch.Tensor,
) -> Dataset[Any]:
    pattern_uint8 = pattern_to_blended_uint8(adaptive_pattern)
    weight = blended_weight_like(pattern_uint8, config.alpha)
    blended_cls = type(original_backdoor_instance)
    adaptive_instance = blended_cls(
        train_dataset=clean_train,
        test_dataset=clean_test,
        model=original_backdoor_instance.model,
        loss=nn.CrossEntropyLoss(),
        y_target=config.attack_target,
        poisoned_rate=config.poisoned_rate,
        pattern=pattern_uint8,
        weight=weight,
        poisoned_transform_train_index=1,
        poisoned_transform_test_index=1,
        seed=config.attack_seed,
        deterministic=config.deterministic,
    )
    return adaptive_instance.poisoned_test_dataset


def apply_blended_trigger(images: torch.Tensor, pattern: torch.Tensor, alpha: float) -> torch.Tensor:
    return torch.clamp((1.0 - alpha) * images + alpha * pattern.unsqueeze(0), 0.0, 1.0)


def target_rate(logits: torch.Tensor, target_label: int) -> tuple[int, float]:
    predictions = logits.argmax(dim=1)
    hits = int((predictions == target_label).sum().item())
    confidence = torch.softmax(logits, dim=1)[:, target_label].mean().item()
    return hits, confidence


def optimize_pattern(
    config: StressConfig,
    model: nn.Module,
    purifier: PipelinePurifier,
    loader: DataLoader[Any],
    best_loader: DataLoader[Any],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    pattern = adaptive_initial_pattern(config, device).detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([pattern], lr=config.lr)
    target = torch.full((config.batch_size,), config.attack_target, dtype=torch.long, device=device)
    model.eval()
    purifier.eval()
    best_pattern = pattern.detach().clone()
    best_rate = -1.0
    best_confidence = 0.0
    best_step = 0

    data_iter = iter(loader)
    for step in range(1, config.steps + 1):
        try:
            images, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            images, _ = next(data_iter)
        images = images.to(device)
        batch_target = target[: images.shape[0]]

        optimizer.zero_grad(set_to_none=True)
        constrained_pattern = torch.clamp(pattern, 0.0, 1.0)
        triggered = apply_blended_trigger(images, constrained_pattern, config.alpha)
        purified = purifier(triggered)
        logits = model(purified)
        ce_loss = F.cross_entropy(logits, batch_target)
        reg_loss = constrained_pattern.square().mean()
        loss = ce_loss + config.lambda_reg * reg_loss
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            pattern.clamp_(0.0, 1.0)

        if step == 1 or step % max(1, config.steps // 10) == 0 or step == config.steps:
            hits, confidence = target_rate(logits.detach(), config.attack_target)
            rate = hits / float(images.shape[0])
            print(
                f"step={step}/{config.steps} loss={loss.item():.4f} "
                f"batch_target_rate={rate:.4f} target_conf={confidence:.4f}"
            )

        if step == 1 or step % config.best_eval_interval == 0 or step == config.steps:
            candidate = pattern.detach().clamp(0.0, 1.0)
            metrics = evaluate_trigger_pipeline(config, model, purifier, best_loader, candidate, device)
            candidate_rate = float(metrics["target_rate"])
            candidate_confidence = float(metrics["mean_target_confidence"])
            if candidate_rate > best_rate or (
                candidate_rate == best_rate and candidate_confidence > best_confidence
            ):
                best_pattern = candidate.clone()
                best_rate = candidate_rate
                best_confidence = candidate_confidence
                best_step = step
                print(
                    f"best_update step={best_step} "
                    f"best_target_rate={best_rate:.4f} best_target_conf={best_confidence:.4f}"
                )

    return best_pattern.clamp(0.0, 1.0), {
        "best_step": best_step,
        "best_eval_target_rate": best_rate,
        "best_eval_mean_target_confidence": best_confidence,
        "best_eval_samples": len(best_loader.dataset),
    }


@torch.no_grad()
def evaluate_clean_pipeline(
    model: nn.Module,
    purifier: PipelinePurifier,
    loader: DataLoader[Any],
    device: torch.device,
) -> tuple[int, int]:
    correct = 0
    total = 0
    model.eval()
    purifier.eval()
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(purifier(images))
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += int(labels.numel())
    return correct, total


@torch.no_grad()
def evaluate_trigger_without_pipeline(
    config: StressConfig,
    model: nn.Module,
    loader: DataLoader[Any],
    pattern: torch.Tensor,
    device: torch.device,
) -> dict[str, float | int]:
    hits = 0
    total = 0
    confidence_sum = 0.0
    model.eval()
    for images, _ in loader:
        images = images.to(device)
        triggered = apply_blended_trigger(images, pattern.to(device), config.alpha)
        logits = model(triggered)
        batch_hits, batch_confidence = target_rate(logits, config.attack_target)
        hits += batch_hits
        total += int(images.shape[0])
        confidence_sum += batch_confidence * float(images.shape[0])
    if total == 0:
        raise RuntimeError("Before-pipeline evaluation loader produced no samples.")
    return {
        "target_hits": hits,
        "total": total,
        "target_rate": hits / float(total),
        "mean_target_confidence": confidence_sum / float(total),
    }


@torch.no_grad()
def evaluate_trigger_pipeline(
    config: StressConfig,
    model: nn.Module,
    purifier: PipelinePurifier,
    loader: DataLoader[Any],
    pattern: torch.Tensor,
    device: torch.device,
) -> dict[str, float | int]:
    hits = 0
    total = 0
    confidence_sum = 0.0
    model.eval()
    purifier.eval()
    for images, _ in loader:
        images = images.to(device)
        triggered = apply_blended_trigger(images, pattern.to(device), config.alpha)
        logits = model(purifier(triggered))
        batch_hits, batch_confidence = target_rate(logits, config.attack_target)
        hits += batch_hits
        total += int(images.shape[0])
        confidence_sum += batch_confidence * float(images.shape[0])
    if total == 0:
        raise RuntimeError("Evaluation loader produced no samples.")
    return {
        "target_hits": hits,
        "total": total,
        "target_rate": hits / float(total),
        "mean_target_confidence": confidence_sum / float(total),
    }


@torch.no_grad()
def evaluate_saved_poisoned_output(
    config: StressConfig,
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
) -> dict[str, float | int]:
    hits = 0
    total = 0
    confidence_sum = 0.0
    model.eval()
    for images, labels in loader:
        del labels
        images = images.to(device)
        logits = model(images)
        batch_hits, batch_confidence = target_rate(logits, config.attack_target)
        hits += batch_hits
        total += int(images.shape[0])
        confidence_sum += batch_confidence * float(images.shape[0])
    if total == 0:
        raise RuntimeError("Saved poisoned output loader produced no samples.")
    return {
        "target_hits": hits,
        "total": total,
        "target_rate": hits / float(total),
        "mean_target_confidence": confidence_sum / float(total),
    }


@torch.no_grad()
def evaluate_saved_clean_output(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
) -> tuple[int, int]:
    correct = 0
    total = 0
    model.eval()
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += int(labels.numel())
    if total == 0:
        raise RuntimeError("Saved clean output loader produced no samples.")
    return correct, total


def evaluate_optional_saved_outputs(
    config: StressConfig,
    model: nn.Module,
    eval_dataset: Dataset[Any],
    device: torch.device,
) -> dict[str, float | int | str]:
    results: dict[str, float | int | str] = {}
    if config.saved_original_purified_pois_root:
        saved_pois = FolderBackedDataset(
            config.saved_original_purified_pois_root,
            poisoned=True,
            dataset=eval_dataset,
        )
        saved_subset = saved_poisoned_subset(saved_pois, config.attack_target, config.eval_samples, config.eval_offset)
        saved_loader = make_loader(saved_subset, config, shuffle=False)
        saved_results = evaluate_saved_poisoned_output(config, model, saved_loader, device)
        results.update(
            {
                "saved_original_purified_pois_root": config.saved_original_purified_pois_root,
                "saved_original_trigger_target_rate_after_pipeline": saved_results["target_rate"],
                "saved_original_trigger_mean_target_confidence": saved_results["mean_target_confidence"],
                "saved_original_eval_total": saved_results["total"],
            }
        )

    if config.saved_clean_purified_root:
        saved_clean = FolderBackedDataset(
            config.saved_clean_purified_root,
            poisoned=False,
            dataset=eval_dataset,
        )
        saved_clean_subset = clean_subset(saved_clean, config.eval_samples, config.eval_offset)
        saved_clean_loader = make_loader(saved_clean_subset, config, shuffle=False)
        clean_correct, clean_total = evaluate_saved_clean_output(model, saved_clean_loader, device)
        results.update(
            {
                "saved_clean_purified_root": config.saved_clean_purified_root,
                "saved_clean_pipeline_accuracy": clean_correct / float(clean_total),
                "saved_clean_pipeline_correct": clean_correct,
                "saved_clean_pipeline_total": clean_total,
            }
        )
    return results


def main_pipeline_args(config: StressConfig, output_root: Path) -> SimpleNamespace:
    val_pois_root = output_root / "val_pois"
    return SimpleNamespace(
        pur_folder=str(output_root),
        image_folder=str(output_root / "train"),
        test_image_folder=str(output_root / "val"),
        test_image_folder_pois=str(val_pois_root),
        splited_image_folder=str(output_root / "split_train"),
        splited_test_image_folder=str(output_root / "split_val"),
        splited_test_image_folder_pois=str(output_root / "split_val_pois"),
        purified_label_style=config.purified_label_style,
        purifier_backend="convir_zip",
        degradation_type=config.degradation_type,
        degradation_strength=config.degradation_strength,
        deg=config.degradation_type,
        deg_scale=config.degradation_strength,
        seed=config.attack_seed,
        gpu=config.device,
        use_conv_ir=config.use_conv_ir,
        disable_conv_ir=not config.use_conv_ir,
        restorer_ckpt=config.restorer_ckpt,
        restorer_version=config.restorer_version,
        concat=True,
    )


def export_test_pois_with_main_purifier(
    config: StressConfig,
    dataset: Dataset[Any],
    output_root: Path,
) -> FolderBackedDataset:
    args = main_pipeline_args(config, output_root)
    purifier = Purify(args, SimpleNamespace(), "test_pois", dataset)
    purified = purifier.pur()
    if isinstance(purified, FolderBackedDataset):
        return purified
    return FolderBackedDataset(args.test_image_folder_pois, poisoned=True, dataset=dataset)


def evaluate_dataset_as_saved_poisoned(
    config: StressConfig,
    model: nn.Module,
    dataset: FolderBackedDataset,
    device: torch.device,
    limit: int | None = None,
    offset: int = 0,
) -> dict[str, float | int]:
    subset = saved_poisoned_subset(
        dataset,
        config.attack_target,
        config.eval_samples if limit is None else limit,
        offset,
    )
    loader = make_loader(subset, config, shuffle=False)
    return evaluate_saved_poisoned_output(config, model, loader, device)


def evaluate_main_pipeline_poisoned_dataset(
    config: StressConfig,
    model: nn.Module,
    dataset: Dataset[Any],
    tag: str,
    device: torch.device,
) -> dict[str, float | int | str]:
    run_id = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    output_root = Path(config.main_pipeline_output_dir) / f"{run_id}_{config.dataset}_{tag}"
    purified_dataset = export_test_pois_with_main_purifier(config, dataset, output_root)
    metrics = evaluate_dataset_as_saved_poisoned(config, model, purified_dataset, device, limit=-1, offset=0)
    return {
        "output_root": str(output_root),
        "target_rate": metrics["target_rate"],
        "mean_target_confidence": metrics["mean_target_confidence"],
        "eval_total": metrics["total"],
    }


def validate_config(config: StressConfig) -> None:
    normalized_degradation = config.degradation_type.strip().lower().replace("-", "_")
    if normalized_degradation not in DIFFERENTIABLE_DEGRADATIONS:
        raise ValueError(
            f"{config.degradation_type} is not supported for gradient-based adaptive search. "
            f"Use one of {sorted(DIFFERENTIABLE_DEGRADATIONS)}."
        )
    if not 0.0 <= config.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1].")
    if config.search_samples == 0 or config.search_samples < -1:
        raise ValueError("--search_samples must be positive, or -1 for all eligible samples.")
    if config.eval_samples == 0 or config.eval_samples < -1:
        raise ValueError("--eval_samples must be positive, or -1 for all eligible samples.")
    if config.best_eval_samples <= 0:
        raise ValueError("--best_eval_samples must be positive.")
    if config.best_eval_interval <= 0:
        raise ValueError("--best_eval_interval must be positive.")
    if config.require_main_asr_baseline and not config.saved_original_purified_pois_root:
        raise ValueError(
            "--require_main_asr_baseline needs --saved_original_purified_pois_root. "
            "Pass the main CA/ASR/PA pipeline's purified val_pois folder."
        )


def add_canonical_baseline(results: dict[str, Any]) -> None:
    saved_rate = results.get("saved_original_trigger_target_rate_after_pipeline")
    saved_confidence = results.get("saved_original_trigger_mean_target_confidence")
    if saved_rate is not None:
        source = "saved_main_pipeline_output"
        canonical_rate = saved_rate
        canonical_confidence = saved_confidence
        comparable_to_main_asr = True
    else:
        source = "in_memory_unverified"
        canonical_rate = results["in_memory_original_trigger_target_rate_after_pipeline"]
        canonical_confidence = results["in_memory_original_trigger_mean_target_confidence"]
        comparable_to_main_asr = False

    adaptive_rate = results["adaptive_pipeline_break_rate"]
    results.update(
        {
            "canonical_original_baseline_source": source,
            "canonical_original_trigger_target_rate_after_pipeline": canonical_rate,
            "canonical_original_trigger_mean_target_confidence": canonical_confidence,
            "adaptive_minus_canonical_original": float(adaptive_rate) - float(canonical_rate),
            "comparable_to_main_asr": comparable_to_main_asr,
        }
    )


def write_results(config: StressConfig, results: dict[str, Any], adaptive_pattern: torch.Tensor) -> Path:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    result_path = output_dir / f"{run_id}_{config.dataset}_Blended_PABR.json"
    payload = {"config": asdict(config), "results": results}
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if config.save_pattern:
        torch.save(adaptive_pattern.cpu(), output_dir / f"{run_id}_{config.dataset}_adaptive_pattern.pt")
    return result_path


def main() -> int:
    config = parse_args()
    validate_config(config)
    seed_everything(config.optim_seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = build_dataset(config, "train")
    eval_dataset = build_dataset(config, "val")
    search_subset = eligible_subset(train_dataset, config.attack_target, config.search_samples)
    eval_subset = eligible_subset(eval_dataset, config.attack_target, config.eval_samples, config.eval_offset)
    best_eval_subset = eligible_subset(eval_dataset, config.attack_target, config.best_eval_samples, config.eval_offset)
    search_loader = make_loader(search_subset, config, shuffle=True)
    eval_loader = make_loader(eval_subset, config, shuffle=False)
    best_eval_loader = make_loader(best_eval_subset, config, shuffle=False)

    model = load_classifier(config, device)
    original_pattern = original_blended_pattern(config, device)
    original_before_pipeline_results = evaluate_trigger_without_pipeline(
        config,
        model,
        eval_loader,
        original_pattern,
        device,
    )
    print(
        "original_asr_before_pipeline="
        f"{original_before_pipeline_results['target_rate']:.6f} "
        f"target_conf={original_before_pipeline_results['mean_target_confidence']:.6f}"
    )

    purifier = PipelinePurifier(config, device).to(device)
    original_results = evaluate_trigger_pipeline(config, model, purifier, eval_loader, original_pattern, device)
    clean_correct, clean_total = evaluate_clean_pipeline(model, purifier, eval_loader, device)

    adaptive_pattern, best_metadata = optimize_pattern(config, model, purifier, search_loader, best_eval_loader, device)
    adaptive_results = evaluate_trigger_pipeline(config, model, purifier, eval_loader, adaptive_pattern, device)

    results = {
        "metric": "PABR",
        "original_asr_before_pipeline": original_before_pipeline_results["target_rate"],
        "original_trigger_target_rate_before_pipeline": original_before_pipeline_results["target_rate"],
        "original_trigger_mean_target_confidence_before_pipeline": original_before_pipeline_results[
            "mean_target_confidence"
        ],
        "original_trigger_target_rate_after_pipeline": original_results["target_rate"],
        "original_trigger_mean_target_confidence": original_results["mean_target_confidence"],
        "in_memory_original_trigger_target_rate_after_pipeline": original_results["target_rate"],
        "in_memory_original_trigger_mean_target_confidence": original_results["mean_target_confidence"],
        "adaptive_pipeline_break_rate": adaptive_results["target_rate"],
        "adaptive_mean_target_confidence": adaptive_results["mean_target_confidence"],
        "adaptive_best_step": best_metadata["best_step"],
        "adaptive_best_eval_target_rate": best_metadata["best_eval_target_rate"],
        "adaptive_best_eval_mean_target_confidence": best_metadata["best_eval_mean_target_confidence"],
        "adaptive_best_eval_samples": best_metadata["best_eval_samples"],
        "clean_pipeline_accuracy_on_eval_subset": clean_correct / float(clean_total),
        "clean_pipeline_correct": clean_correct,
        "clean_pipeline_total": clean_total,
        "eval_eligible_total": adaptive_results["total"],
    }
    results.update(evaluate_optional_saved_outputs(config, model, eval_dataset, device))

    if config.final_eval_via_main_pipeline:
        process_dataset = import_process_dataset(config)
        clean_train, clean_test, _, original_poisoned_test, original_backdoor_instance = process_dataset(
            SimpleNamespace(
                dataset=config.dataset,
                attack_method="Blended",
                datasets_root_dir=config.datasets_root_dir,
                classes=config.classes,
                img_size=config.img_size,
                attack_target=config.attack_target,
                poisoned_rate=config.poisoned_rate,
                seed=config.attack_seed,
                deterministic=config.deterministic,
            )
        )
        eval_indices = [int(index) for index in eval_subset.indices]
        original_poisoned_eval = DatasetSubsetWithAttrs(original_poisoned_test, eval_indices)
        original_main = evaluate_main_pipeline_poisoned_dataset(
            config,
            model,
            original_poisoned_eval,
            "original_blended",
            device,
        )
        adaptive_poisoned_test = build_adaptive_blended_poisoned_test(
            config,
            original_backdoor_instance,
            clean_train,
            clean_test,
            adaptive_pattern,
        )
        adaptive_poisoned_eval = DatasetSubsetWithAttrs(adaptive_poisoned_test, eval_indices)
        adaptive_main = evaluate_main_pipeline_poisoned_dataset(
            config,
            model,
            adaptive_poisoned_eval,
            "adaptive_blended",
            device,
        )
        results.update(
            {
                "main_original_output_root": original_main["output_root"],
                "main_original_asr_after_pipeline": original_main["target_rate"],
                "main_original_mean_target_confidence": original_main["mean_target_confidence"],
                "main_original_eval_total": original_main["eval_total"],
                "main_adaptive_output_root": adaptive_main["output_root"],
                "main_adaptive_asr_after_pipeline": adaptive_main["target_rate"],
                "main_adaptive_mean_target_confidence": adaptive_main["mean_target_confidence"],
                "main_adaptive_eval_total": adaptive_main["eval_total"],
                "main_adaptive_minus_original": float(adaptive_main["target_rate"])
                - float(original_main["target_rate"]),
                "comparable_to_main_pipeline": True,
            }
        )

    add_canonical_baseline(results)
    if not results["comparable_to_main_asr"]:
        print(
            "WARNING: no saved main-pipeline val_pois baseline was provided. "
            "Use --saved_original_purified_pois_root for a result comparable to main ASR."
        )
    result_path = write_results(config, results, adaptive_pattern)

    print(json.dumps(results, indent=2, sort_keys=True))
    print(f"Wrote results to {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
