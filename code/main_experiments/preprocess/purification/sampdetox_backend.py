from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Type

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF  # type: ignore[import-untyped]

try:
    from skimage.metrics import structural_similarity as _structural_similarity
except ImportError:  # pragma: no cover - validated at runtime with a clear error.
    _structural_similarity = None

from preprocess.purification.class_layout import (
    ClassLayout,
    SplitCLeanDataset,
    SplitDataset,
    image_root,
    nonSplitDataset,
    save_tensor_image,
)


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _default_source_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "SampDetox"


def _resolve_source_dir(args: Any) -> Path:
    raw_source = getattr(args, "sampdetox_source_dir", "")
    if raw_source:
        source_dir = Path(raw_source)
        if not source_dir.is_absolute():
            source_dir = Path.cwd() / source_dir
    else:
        source_dir = _default_source_dir()
    return source_dir.resolve()


def _resolve_checkpoint(args: Any, source_dir: Path) -> Path:
    raw_ckpt = getattr(args, "sampdetox_ckpt", "")
    if raw_ckpt:
        ckpt_path = Path(raw_ckpt)
        if not ckpt_path.is_absolute():
            ckpt_path = Path.cwd() / ckpt_path
        return ckpt_path.resolve()
    return (source_dir / "diffusion" / "Checkpoints" / "ckpt_999_.pt").resolve()


def _get_mask(image_size: int, num_blocks: int, batch_size: int, device: torch.device) -> torch.Tensor:
    if image_size % num_blocks != 0:
        raise ValueError("sampdetox_image_size must be divisible by sampdetox_num_blocks")
    block_len = image_size // num_blocks
    mask = torch.ones((batch_size, 3, num_blocks, num_blocks), device=device)
    mask[:, :, ::2, ::2] = 0.0
    mask[:, :, 1::2, 1::2] = 0.0
    return mask.repeat_interleave(repeats=block_len, dim=2).repeat_interleave(repeats=block_len, dim=3)


class _SampDetoxRestorer:
    def __init__(self, args: Any) -> None:
        self.args = args
        self.source_dir = _resolve_source_dir(args)
        self.ckpt_path = _resolve_checkpoint(args, self.source_dir)
        self.image_size = int(getattr(args, "sampdetox_image_size", 32))
        self.t1 = int(getattr(args, "sampdetox_t1", 120))
        self.t2 = int(getattr(args, "sampdetox_t2", 120))
        self.num_blocks = int(getattr(args, "sampdetox_num_blocks", 8))
        self.save_diagnostics = bool(getattr(args, "sampdetox_save_diagnostics", False))
        self.device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
        self.sampler: Any = self._build_sampler()

    def _build_sampler(self) -> nn.Module:
        if _structural_similarity is None:
            raise ImportError("SampDetox backend requires scikit-image. Install it in the tf environment.")
        if not self.source_dir.exists():
            raise FileNotFoundError(f"SampDetox source directory not found: {self.source_dir}")
        if not self.ckpt_path.exists():
            raise FileNotFoundError(
                "SampDetox diffusion checkpoint not found. Pass --sampdetox_ckpt or place ckpt_999_.pt under "
                f"{self.source_dir / 'diffusion' / 'Checkpoints'}."
            )

        model_module = _load_module(
            "sampdetox_diffusion_model",
            self.source_dir / "diffusion" / "Diffusion" / "Model.py",
        )
        diffusion_module = _load_module(
            "sampdetox_diffusion_sampler",
            self.source_dir / "diffusion" / "Diffusion" / "Diffusion.py",
        )
        unet_cls: Type[nn.Module] = model_module.UNet
        sampler_cls: Type[nn.Module] = diffusion_module.GaussianDiffusionSampler

        model = unet_cls(
            T=1000,
            ch=128,
            ch_mult=[1, 2, 3, 4],
            attn=[2],
            num_res_blocks=2,
            dropout=0.0,
        )
        state_dict = torch.load(self.ckpt_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        sampler = sampler_cls(model, 1e-4, 0.02, 1000).to(self.device)
        sampler.eval()
        return sampler

    def _resize_for_sampdetox(self, image: torch.Tensor) -> torch.Tensor:
        image = torch.clamp(image.detach().cpu(), 0.0, 1.0)
        if image.shape[-2:] == (self.image_size, self.image_size):
            return image
        return TF.resize(image, [self.image_size, self.image_size], antialias=True)

    def _ssim_difference(self, old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        if _structural_similarity is None:
            raise ImportError("SampDetox backend requires scikit-image. Install it in the tf environment.")
        old_np = old.squeeze(0).detach().cpu().numpy()
        new_np = new.squeeze(0).detach().cpu().numpy()
        _, difference = _structural_similarity(
            old_np,
            new_np,
            channel_axis=0,
            data_range=1.0,
            gaussian_weights=True,
            sigma=1.5,
            full=True,
        )
        diff = torch.as_tensor(difference, dtype=torch.float32)
        grey = 0.11 * diff[0] + 0.59 * diff[1] + 0.3 * diff[2]
        denom = torch.clamp(torch.max(grey) - torch.min(grey), min=1e-12)
        return (grey - torch.min(grey)) / denom

    @torch.no_grad()
    def restore(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        original_size = tuple(image.shape[-2:])
        x0_clean = self._resize_for_sampdetox(image)
        x0 = x0_clean.unsqueeze(0).to(self.device)
        x0_scaled = 2.0 * (x0 - 0.5)

        _get_mask(self.image_size, self.num_blocks, 1, self.device)
        noise = torch.randn((1, 3, self.image_size, self.image_size), device=self.device)
        t1 = x0.new_ones([1], dtype=torch.long) * self.t1
        xt1 = (
            self.sampler.extract(self.sampler.sqrt_alpha_t, t1, x0.shape) * x0_scaled
            + self.sampler.extract(self.sampler.sqrt_1_alpha_t, t1, x0.shape) * noise
        )
        sampled = self.sampler(xt1, self.t1, None)
        x0bar = torch.clamp(sampled * 0.5 + 0.5, 0.0, 1.0)

        grey = self._ssim_difference(x0, x0bar).to(self.device)
        t2_map = ((1.0 - grey) * self.t2).to(torch.long).clamp_(0, self.t2)
        a_map = self.sampler.sqrt_alpha_t[t2_map].float().to(self.device)
        b_map = self.sampler.sqrt_1_alpha_t[t2_map].float().to(self.device)
        xtbar = (2.0 * (x0bar - 0.5)) * a_map.unsqueeze(0).unsqueeze(0) + noise * b_map.unsqueeze(0).unsqueeze(0)
        out = torch.clamp(self.sampler(xtbar, self.t2, t2_map) * 0.5 + 0.5, 0.0, 1.0).squeeze(0).cpu()

        if original_size != (self.image_size, self.image_size):
            out = TF.resize(out, list(original_size), antialias=True)
        if not self.save_diagnostics:
            return out, None, None
        return out, x0_clean.cpu(), grey.cpu().unsqueeze(0).repeat(3, 1, 1)


class Purify:
    def __init__(self, args: Any, config: Any, type: str, dataset: Any) -> None:
        del config
        self.args = args
        self.type = type
        self.dataset = dataset
        self.layout = ClassLayout.from_dataset(dataset, getattr(args, "purified_label_style", "class_name"))
        self.restorer = _SampDetoxRestorer(args)

    def _output_root(self) -> str:
        return image_root(self.args, self.type)

    def _save_sample(self, image: torch.Tensor, label: Any, index: int) -> None:
        sample_path = self.layout.sample_path(self._output_root(), label, index)
        save_tensor_image(image, sample_path.path)

    def _save_diagnostics(
        self,
        original: torch.Tensor | None,
        grey: torch.Tensor | None,
        label: Any,
        index: int,
    ) -> None:
        if original is None or grey is None:
            return
        root = self._output_root()
        sample_path = self.layout.sample_path(root, label, index)
        relative_path = os.path.relpath(sample_path.path, root)
        save_tensor_image(original, os.path.join(root + "_sampdetox_original", relative_path))
        save_tensor_image(grey, os.path.join(root + "_sampdetox_grey", relative_path))

    def pur(self) -> Dataset:
        os.makedirs(self._output_root(), exist_ok=True)
        for index in range(len(self.dataset)):
            image, label = self.dataset[index]
            purified, original, grey = self.restorer.restore(image)
            self._save_sample(purified, label, index)
            self._save_diagnostics(original, grey, label, index)

        if not self.args.concat:
            return nonSplitDataset(self.args, self.type)
        if self.type == "test":
            return SplitCLeanDataset(self.dataset, self.args, self.type)
        if self.type in {"train", "test_pois"}:
            return SplitDataset(self.dataset, self.args, self.type)
        raise ValueError(f"Unsupported purification type: {self.type}")


__all__ = ["Purify"]
