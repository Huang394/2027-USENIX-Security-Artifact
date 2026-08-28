"""Input degradation and restoration pipelines."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Literal, cast

from PIL import Image
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import functional as tvf

DegradationType = Literal["none", "gaussian_noise", "haze", "blur", "motion_blur", "jpeg", "resize"]


class IdentityRestorer(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images


class ConvIRCompatibleDegrader:
    """Degradation implementation aligned with ConvIR-ZIP's Degrader."""

    def __init__(self, degradation: str, strength: float, seed: int) -> None:
        self.degradation = _normalize_degradation(degradation)
        self.strength = float(strength)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))

    def degrade(self, images: torch.Tensor) -> torch.Tensor:
        if self.degradation == "none":
            return images.clone()
        if self.degradation == "gaussian_noise":
            return self._gaussian_noise(images)
        if self.degradation == "haze":
            return self._haze(images)
        if self.degradation == "blur":
            return self._blur(images)
        if self.degradation == "motion_blur":
            return self._motion_blur(images)
        if self.degradation == "jpeg":
            return self._jpeg(images)
        if self.degradation == "resize":
            return self._resize(images)
        raise AssertionError("Unexpected degradation type")

    def _gaussian_noise(self, images: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(
            images.shape,
            generator=self.generator,
            dtype=images.dtype,
            device="cpu",
        ).to(images.device)
        return torch.clamp(images + noise * self.strength, 0.0, 1.0)

    def _haze(self, images: torch.Tensor) -> torch.Tensor:
        alpha = max(0.0, min(1.0, self.strength))
        haze = torch.full_like(images, 1.0)
        return torch.clamp(images * (1.0 - alpha) + haze * alpha, 0.0, 1.0)

    def _blur(self, images: torch.Tensor) -> torch.Tensor:
        kernel_size = max(3, int(round(self.strength)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        sigma = max(0.1, self.strength)
        return torch.clamp(
            tvf.gaussian_blur(
                images,
                kernel_size=[kernel_size, kernel_size],
                sigma=[sigma, sigma],
            ),
            0.0,
            1.0,
        )

    def _motion_blur(self, images: torch.Tensor) -> torch.Tensor:
        kernel_size = max(3, int(round(self.strength)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        channels = images.shape[1]
        kernel = torch.zeros(
            (channels, 1, kernel_size, kernel_size),
            dtype=images.dtype,
            device=images.device,
        )
        center = kernel_size // 2
        kernel[:, 0, center, :] = 1.0 / float(kernel_size)
        blurred = F.conv2d(images, kernel, padding=center, groups=channels)
        return torch.clamp(blurred, 0.0, 1.0)

    def _jpeg(self, images: torch.Tensor) -> torch.Tensor:
        quality = int(round(100.0 - self.strength * 40.0))
        quality = max(5, min(95, quality))
        restored = []
        for image in images:
            pil_image = tvf.to_pil_image(torch.clamp(image, 0.0, 1.0))
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            restored.append(tvf.to_tensor(Image.open(buffer).convert("RGB")).to(images.device))
        return torch.stack(restored, dim=0)

    def _resize(self, images: torch.Tensor) -> torch.Tensor:
        scale = max(1.0, self.strength)
        height, width = images.shape[-2:]
        down_height = max(1, int(round(height / scale)))
        down_width = max(1, int(round(width / scale)))
        resized = F.interpolate(
            images,
            size=(down_height, down_width),
            mode="bilinear",
            align_corners=False,
        )
        return torch.clamp(
            F.interpolate(resized, size=(height, width), mode="bilinear", align_corners=False),
            0.0,
            1.0,
        )


def _normalize_degradation(degradation: str) -> DegradationType:
    normalized = degradation.strip().lower().replace("-", "_")
    aliases = {
        "none": "none",
        "identity": "none",
        "noop": "none",
        "clean": "none",
        "noise": "gaussian_noise",
        "gaussian": "gaussian_noise",
        "gaussian_noise": "gaussian_noise",
        "fog": "haze",
        "haze": "haze",
        "blur": "blur",
        "gaussian_blur": "blur",
        "motion": "motion_blur",
        "motion_blur": "motion_blur",
        "motionblur": "motion_blur",
        "linear_motion_blur": "motion_blur",
        "jpeg": "jpeg",
        "compression": "jpeg",
        "resize": "resize",
        "downsample": "resize",
        "down_up": "resize",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported degradation type: {degradation}")
    return cast(DegradationType, aliases[normalized])


def apply_degradation(
    images: torch.Tensor,
    degradation: str,
    strength: float,
    seed: int = 1234,
) -> torch.Tensor:
    return ConvIRCompatibleDegrader(degradation, strength, seed).degrade(images)


def load_restorer(path: Path | None, device: torch.device) -> nn.Module:
    if path is None:
        return IdentityRestorer().to(device).eval()
    try:
        model = torch.jit.load(str(path), map_location=device)
    except RuntimeError:
        loaded = torch.load(path, map_location=device, weights_only=False)
        if not isinstance(loaded, nn.Module):
            raise TypeError(
                "Restoration checkpoint must be TorchScript or a serialized torch.nn.Module"
            )
        model = loaded
    return model.to(device).eval()


@torch.inference_mode()
def apply_restoration(images: torch.Tensor, restorer: nn.Module) -> torch.Tensor:
    restored = restorer(images)
    if isinstance(restored, tuple):
        restored = restored[0]
    return torch.clamp(restored, 0.0, 1.0)


@torch.inference_mode()
def build_pipeline_images(
    clean: torch.Tensor,
    poisoned: torch.Tensor,
    degradation: str,
    strength: float,
    restorer: nn.Module,
    degradation_seed: int = 1234,
    batch_size: int = 32,
) -> dict[str, torch.Tensor]:
    clean_degrader = ConvIRCompatibleDegrader(degradation, strength, degradation_seed)
    poisoned_degrader = ConvIRCompatibleDegrader(degradation, strength, degradation_seed)
    degraded_clean_parts = []
    degraded_poisoned_parts = []
    restored_degraded_clean_parts = []
    restored_degraded_poisoned_parts = []
    for start, end in _batch_slices(clean.shape[0], batch_size):
        degraded_clean_batch = clean_degrader.degrade(clean[start:end])
        degraded_poisoned_batch = poisoned_degrader.degrade(poisoned[start:end])
        degraded_clean_parts.append(degraded_clean_batch.detach().cpu())
        degraded_poisoned_parts.append(degraded_poisoned_batch.detach().cpu())
        restored_degraded_clean_parts.append(
            apply_restoration(degraded_clean_batch, restorer).detach().cpu()
        )
        restored_degraded_poisoned_parts.append(
            apply_restoration(degraded_poisoned_batch, restorer).detach().cpu()
        )
    degraded_clean = torch.cat(degraded_clean_parts, dim=0)
    degraded_poisoned = torch.cat(degraded_poisoned_parts, dim=0)
    return {
        "clean": clean.detach().cpu(),
        "poisoned": poisoned.detach().cpu(),
        "degraded_clean": degraded_clean,
        "degraded_poisoned": degraded_poisoned,
        "restored_degraded_clean": torch.cat(restored_degraded_clean_parts, dim=0),
        "restored_degraded_poisoned": torch.cat(restored_degraded_poisoned_parts, dim=0),
    }


def _batch_slices(total: int, batch_size: int) -> list[tuple[int, int]]:
    step = max(1, batch_size)
    return [(start, min(start + step, total)) for start in range(0, total, step)]
