from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from .network import build_net


@dataclass(frozen=True)
class RestorerConfig:
    version: str = "base"
    ckpt_path: str = ""
    min_size: int = 256


class ConvIRRestorer:
    """Callable wrapper around the ConvIR restoration network."""

    def __init__(
        self,
        version: str = "base",
        ckpt_path: str = "",
        device: Optional[torch.device] = None,
        min_size: int = 256,
    ) -> None:
        self.config = RestorerConfig(version=version, ckpt_path=ckpt_path, min_size=min_size)
        self.device = device or torch.device("cpu")
        self.model = build_net(version).to(self.device)
        self.model.eval()
        if ckpt_path:
            self.load_checkpoint(ckpt_path)

    def load_checkpoint(self, ckpt_path: str) -> None:
        checkpoint_path = Path(ckpt_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"ConvIR checkpoint not found: {ckpt_path}")
        state = torch.load(checkpoint_path, map_location=self.device)
        state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

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
        h, w = x.shape[-2:]
        target_h = cls._padded_size(h, factor, min_size)
        target_w = cls._padded_size(w, factor, min_size)
        padded = x
        while padded.shape[-2] < target_h or padded.shape[-1] < target_w:
            current_h, current_w = padded.shape[-2:]
            remaining_h = target_h - current_h
            remaining_w = target_w - current_w
            pad_h = min(remaining_h, max(0, current_h - 1))
            pad_w = min(remaining_w, max(0, current_w - 1))
            padded = F.pad(padded, (0, pad_w, 0, pad_h), mode="reflect")
        return padded, (h, w)

    def restore(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.unsqueeze(0) if x.ndim == 3 else x
        squeezed = x.ndim == 3
        batch = batch.to(self.device)
        padded, original_shape = self._pad_to_factor(batch, min_size=self.config.min_size)
        with torch.no_grad():
            restored = self.model(padded)[-1]
        restored = restored[..., : original_shape[0], : original_shape[1]]
        restored = torch.clamp(restored, 0.0, 1.0)
        if squeezed:
            return restored.squeeze(0)
        return restored
