"""Model loading and feature extraction helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models


class BackdoorBoxBasicBlock(nn.Module):
    """BackdoorBox ResNet BasicBlock used by ConvIR-ZIP classifiers."""

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut: nn.Module = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )
        self.dropout = dropout

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(images)))
        out = F.dropout(out, p=self.dropout)
        out = self.bn2(self.conv2(out))
        out = F.dropout(out, p=self.dropout)
        out += self.shortcut(images)
        out = F.relu(out)
        return F.dropout(out, p=self.dropout)


class BackdoorBoxBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        del dropout
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut: nn.Module = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(images)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(images)
        return F.relu(out)


class BackdoorBoxResNet(nn.Module):
    """ConvIR-ZIP BackdoorBox ResNet implementation."""

    def __init__(
        self,
        block: Any,
        num_blocks: list[int],
        num_classes: int = 10,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_planes = 64
        self.dropout = dropout

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self,
        block: Any,
        planes: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers: list[nn.Module] = []
        for block_stride in strides:
            layers.append(block(self.in_planes, planes, block_stride, dropout=self.dropout))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(images)))
        out = F.dropout(out, p=self.dropout)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.linear(out)


def _build_backdoorbox_resnet(depth: int, num_classes: int) -> BackdoorBoxResNet:
    if depth == 18:
        return BackdoorBoxResNet(BackdoorBoxBasicBlock, [2, 2, 2, 2], num_classes)
    if depth == 34:
        return BackdoorBoxResNet(BackdoorBoxBasicBlock, [3, 4, 6, 3], num_classes)
    if depth == 50:
        return BackdoorBoxResNet(BackdoorBoxBottleneck, [3, 4, 6, 3], num_classes)
    if depth == 101:
        return BackdoorBoxResNet(BackdoorBoxBottleneck, [3, 4, 23, 3], num_classes)
    if depth == 152:
        return BackdoorBoxResNet(BackdoorBoxBottleneck, [3, 8, 36, 3], num_classes)
    raise ValueError(f"Unsupported BackdoorBox ResNet depth '{depth}'")


def _build_architecture(model_arch: str, num_classes: int | None) -> nn.Module:
    if model_arch in {
        "backdoorbox-resnet34",
        "convir-backdoorbox-resnet34",
        "backdoorpurification-backdoorbox-resnet34",
    }:
        return _build_backdoorbox_resnet(34, num_classes or 20)
    if model_arch == "resnet18":
        model = models.resnet18(weights=None)
    elif model_arch == "resnet34":
        model = models.resnet34(weights=None)
    else:
        raise ValueError(f"Unsupported model_arch '{model_arch}'")
    if num_classes is not None:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _load_state_dict_compatible(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Mirror ConvIR-ZIP BackdoorBox checkpoint prefix handling."""

    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError:
        pass

    model_keys = tuple(model.state_dict().keys())
    checkpoint_keys = tuple(state_dict.keys())
    model_uses_module = all(key.startswith("module.") for key in model_keys)
    checkpoint_uses_module = all(key.startswith("module.") for key in checkpoint_keys)

    if model_uses_module and not checkpoint_uses_module:
        adjusted = {f"module.{key}": value for key, value in state_dict.items()}
    elif checkpoint_uses_module and not model_uses_module:
        adjusted = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }
    else:
        adjusted = state_dict
    model.load_state_dict(adjusted)


def load_classifier(
    path: Path,
    device: torch.device,
    model_arch: str,
    num_classes: int | None,
) -> nn.Module:
    try:
        model = torch.jit.load(str(path), map_location=device)
        return model.to(device).eval()
    except RuntimeError:
        loaded: Any = torch.load(path, map_location=device, weights_only=False)
    if isinstance(loaded, nn.Module):
        return loaded.to(device).eval()
    if isinstance(loaded, dict):
        state_dict = loaded.get("state_dict", loaded.get("model", loaded))
        if not isinstance(state_dict, dict):
            raise TypeError("Checkpoint state_dict/model entry must be a mapping")
        model = _build_architecture(model_arch, num_classes)
        tensor_state_dict = {
            str(key): value for key, value in state_dict.items() if isinstance(value, torch.Tensor)
        }
        _load_state_dict_compatible(model, tensor_state_dict)
        return model.to(device).eval()
    raise TypeError("Unsupported classifier checkpoint format")


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class FeatureExtractor(nn.Module):
    """ResNet-oriented feature extractor with a generic penultimate fallback."""

    def __init__(self, model: nn.Module, layers: tuple[str, ...]) -> None:
        super().__init__()
        self.model = model
        self.layers = layers
        self.outputs: dict[str, torch.Tensor] = {}
        self.handles: list[Any] = []
        self.hookless = False
        for name in layers:
            module = self._resolve_module(name)
            try:
                if name == "penultimate" and hasattr(self.model, "linear"):
                    self.handles.append(module.register_forward_pre_hook(self._make_pre_hook(name)))
                else:
                    self.handles.append(module.register_forward_hook(self._make_hook(name)))
            except RuntimeError:
                if layers != ("penultimate",):
                    raise RuntimeError(
                        "TorchScript models only support the penultimate fallback; "
                        "provide a regular nn.Module for named layer hooks."
                    ) from None
                self.hookless = True

    def _resolve_module(self, name: str) -> nn.Module:
        if name == "penultimate" and "RecursiveScriptModule" in type(self.model).__name__:
            return self.model
        if name == "penultimate" and hasattr(self.model, "avgpool"):
            return getattr(self.model, "avgpool")
        if name == "penultimate" and hasattr(self.model, "linear"):
            return getattr(self.model, "linear")
        modules = dict(self.model.named_modules())
        if name not in modules:
            raise ValueError(f"Layer '{name}' not found in model")
        return modules[name]

    def _make_hook(self, name: str) -> Any:
        def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            self.outputs[name] = torch.flatten(output.detach(), start_dim=1)

        return hook

    def _make_pre_hook(self, name: str) -> Callable[[nn.Module, tuple[torch.Tensor, ...]], None]:
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            self.outputs[name] = torch.flatten(inputs[0].detach(), start_dim=1)

        return hook

    @torch.inference_mode()
    def extract(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        self.outputs = {}
        logits = self.model(images)
        if self.hookless:
            self.outputs["penultimate"] = torch.flatten(logits.detach(), start_dim=1)
        return dict(self.outputs)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
