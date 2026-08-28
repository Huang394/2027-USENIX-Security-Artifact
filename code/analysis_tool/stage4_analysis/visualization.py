"""Plotting helpers for mechanism-analysis outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torchvision.utils import make_grid


def save_tensor_grid(tensors: list[torch.Tensor], titles: list[str], output_path: Path) -> None:
    rows = []
    for tensor in tensors:
        rows.append(tensor[: min(8, tensor.shape[0])].cpu())
    grid = make_grid(torch.cat(rows), nrow=min(8, tensors[0].shape[0]), padding=2)
    image = grid.permute(1, 2, 0).numpy()
    plt.figure(figsize=(12, 2.2 * len(tensors)))
    plt.imshow(image)
    plt.axis("off")
    row_height = image.shape[0] / len(tensors)
    for idx, title in enumerate(titles):
        plt.text(2, idx * row_height + 16, title, color="white", fontsize=10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_bar_plot(df: pd.DataFrame, x: str, y: str, output_path: Path, title: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.bar(df[x].astype(str), df[y])
    plt.title(title)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_line_plot(
    df: pd.DataFrame,
    x: str,
    ys: list[str],
    output_path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(8, 4))
    for y in ys:
        plt.plot(df[x], df[y], marker="o", label=y)
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()
