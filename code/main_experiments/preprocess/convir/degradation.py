from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Literal, cast

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF  # type: ignore[import-untyped]


DegradationType = Literal["none", "gaussian_noise", "haze", "blur", "motion_blur", "jpeg", "resize"]


@dataclass(frozen=True)
class DegradationConfig:
    """Parameters for the degradation stage."""

    degradation_type: DegradationType = "gaussian_noise"
    strength: float = 0.1
    seed: int = 0


class Degrader:
    """Apply a deterministic degradation to an image tensor."""

    def __init__(self, degradation_type: str, strength: float, seed: int) -> None:
        self.config = DegradationConfig(
            degradation_type=self._normalize_type(degradation_type),
            strength=float(strength),
            seed=int(seed),
        )
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(self.config.seed)

    @staticmethod
    def _normalize_type(degradation_type: str) -> DegradationType:
        normalized = degradation_type.strip().lower().replace("-", "_")
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
            raise ValueError(f"Unsupported degradation type: {degradation_type}")
        return cast(DegradationType, aliases[normalized])

    @staticmethod
    def _as_batch(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if x.ndim == 3:
            return x.unsqueeze(0), True
        if x.ndim == 4:
            return x, False
        raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(x.shape)}")

    @staticmethod
    def _restore_shape(x: torch.Tensor, squeezed: bool) -> torch.Tensor:
        if squeezed:
            return x.squeeze(0)
        return x

    def degrade(self, x: torch.Tensor) -> torch.Tensor:
        batch, squeezed = self._as_batch(x)
        if self.config.degradation_type == "none":
            degraded = batch.clone()
        elif self.config.degradation_type == "gaussian_noise":
            degraded = self._gaussian_noise(batch)
        elif self.config.degradation_type == "haze":
            degraded = self._haze(batch)
        elif self.config.degradation_type == "blur":
            degraded = self._blur(batch)
        elif self.config.degradation_type == "motion_blur":
            degraded = self._motion_blur(batch)
        elif self.config.degradation_type == "jpeg":
            degraded = self._jpeg(batch)
        elif self.config.degradation_type == "resize":
            degraded = self._resize(batch)
        else:
            raise AssertionError("Unexpected degradation type")
        return self._restore_shape(degraded, squeezed)

    def _gaussian_noise(self, batch: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(
            batch.shape,
            generator=self._generator,
            dtype=batch.dtype,
            device="cpu",
        ).to(batch.device)
        return torch.clamp(batch + noise * self.config.strength, 0.0, 1.0)

    def _haze(self, batch: torch.Tensor) -> torch.Tensor:
        alpha = max(0.0, min(1.0, self.config.strength))
        haze = torch.full_like(batch, 1.0)
        return torch.clamp(batch * (1.0 - alpha) + haze * alpha, 0.0, 1.0)

    def _blur(self, batch: torch.Tensor) -> torch.Tensor:
        kernel_size = max(3, int(round(self.config.strength)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        sigma = max(0.1, float(self.config.strength))
        return torch.clamp(
            TF.gaussian_blur(batch, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma]),
            0.0,
            1.0,
        )

    def _motion_blur(self, batch: torch.Tensor) -> torch.Tensor:
        kernel_size = max(3, int(round(self.config.strength)))
        if kernel_size % 2 == 0:
            kernel_size += 1

        channels = batch.shape[1]
        kernel = torch.zeros(
            (channels, 1, kernel_size, kernel_size),
            dtype=batch.dtype,
            device=batch.device,
        )
        center = kernel_size // 2
        kernel[:, 0, center, :] = 1.0 / float(kernel_size)
        blurred = F.conv2d(batch, kernel, padding=center, groups=channels)
        return torch.clamp(blurred, 0.0, 1.0)

    def _jpeg(self, batch: torch.Tensor) -> torch.Tensor:
        quality = int(round(100.0 - self.config.strength * 40.0))
        quality = max(5, min(95, quality))
        restored = []
        for img in batch:
            pil_img = TF.to_pil_image(torch.clamp(img, 0.0, 1.0))
            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            restored.append(TF.to_tensor(Image.open(buffer).convert("RGB")).to(batch.device))
        return torch.stack(restored, dim=0)

    def _resize(self, batch: torch.Tensor) -> torch.Tensor:
        scale = max(1.0, float(self.config.strength))
        h, w = batch.shape[-2:]
        down_h = max(1, int(round(h / scale)))
        down_w = max(1, int(round(w / scale)))
        resized = F.interpolate(batch, size=(down_h, down_w), mode="bilinear", align_corners=False)
        return torch.clamp(
            F.interpolate(resized, size=(h, w), mode="bilinear", align_corners=False),
            0.0,
            1.0,
        )
