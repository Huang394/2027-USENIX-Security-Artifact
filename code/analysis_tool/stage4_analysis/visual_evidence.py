"""BackdoorBench-style visual evidence generated from mechanism-analysis outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from stage4_analysis.config import CaseConfig
from stage4_analysis.models import FeatureExtractor
from stage4_analysis.pipeline_settings import PIPELINE_SETTINGS, ResolvedDefenseOutput


@dataclass(frozen=True)
class VisualEvidenceResult:
    generated: list[str]
    skipped: list[str]


def write_visual_evidence(
    case_dir: Path,
    config: CaseConfig,
    defense_output: ResolvedDefenseOutput,
    classifier: nn.Module,
    images: dict[str, torch.Tensor],
    labels: torch.Tensor,
    logit_df: pd.DataFrame,
) -> VisualEvidenceResult:
    output_dir = case_dir / "visual_evidence"
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    skipped: list[str] = []

    generated.extend(_write_confusion_matrices(output_dir, config, defense_output, logit_df))
    generated.extend(_write_prediction_flow(output_dir, config, defense_output, logit_df))
    generated.extend(_write_frequency_spectrum(output_dir, defense_output, images))

    embedding_result = _write_feature_embedding(
        output_dir,
        config,
        defense_output,
        classifier,
        images,
        labels,
    )
    generated.extend(embedding_result.generated)
    skipped.extend(embedding_result.skipped)

    gradcam_result = _write_gradcam_grid(
        output_dir,
        config,
        defense_output,
        classifier,
        images,
    )
    generated.extend(gradcam_result.generated)
    skipped.extend(gradcam_result.skipped)

    _write_manifest(output_dir, generated, skipped)
    return VisualEvidenceResult(generated=generated, skipped=skipped)


def _write_confusion_matrices(
    output_dir: Path,
    config: CaseConfig,
    defense_output: ResolvedDefenseOutput,
    logit_df: pd.DataFrame,
) -> list[str]:
    out_dir = output_dir / "confusion_matrix"
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    poisoned = logit_df[logit_df["domain"] == "poisoned"]
    if not poisoned.empty:
        path = out_dir / "poisoned_pipeline_confusion.png"
        _plot_multi_setting_confusion(
            poisoned,
            path,
            config.num_classes,
            title="Poisoned pipeline confusion matrices",
        )
        generated.append(str(path.relative_to(output_dir)))

    final_rows = logit_df[logit_df["setting"] == defense_output.poisoned_setting]
    if not final_rows.empty:
        path = out_dir / "final_output_confusion.png"
        _plot_confusion(
            final_rows["true_label"].to_numpy(dtype=np.int64),
            final_rows["predicted_label"].to_numpy(dtype=np.int64),
            path,
            config.num_classes,
            title=f"Final poisoned output: {defense_output.poisoned_setting}",
        )
        generated.append(str(path.relative_to(output_dir)))
    return generated


def _write_prediction_flow(
    output_dir: Path,
    config: CaseConfig,
    defense_output: ResolvedDefenseOutput,
    logit_df: pd.DataFrame,
) -> list[str]:
    out_dir = output_dir / "prediction_flow"
    out_dir.mkdir(parents=True, exist_ok=True)
    origin = logit_df[logit_df["setting"] == "origin_poisoned"][
        ["sample_index", "predicted_label"]
    ].rename(columns={"predicted_label": "origin_predicted_label"})
    final = logit_df[logit_df["setting"] == defense_output.poisoned_setting][
        ["sample_index", "predicted_label", "true_label"]
    ].rename(columns={"predicted_label": "final_predicted_label"})
    merged = final.merge(origin, on="sample_index", how="inner")
    if merged.empty:
        return []
    csv_path = out_dir / "origin_to_final_prediction_flow.csv"
    merged.to_csv(csv_path, index=False)
    heatmap_path = out_dir / "origin_to_final_prediction_flow.png"
    matrix = _count_matrix(
        merged["origin_predicted_label"].to_numpy(dtype=np.int64),
        merged["final_predicted_label"].to_numpy(dtype=np.int64),
        config.num_classes,
    )
    _plot_matrix(
        matrix,
        heatmap_path,
        xlabel="Final poisoned prediction",
        ylabel="Origin poisoned prediction",
        title="Prediction flow: origin poisoned -> final poisoned",
    )
    return [
        str(csv_path.relative_to(output_dir)),
        str(heatmap_path.relative_to(output_dir)),
    ]


def _write_feature_embedding(
    output_dir: Path,
    config: CaseConfig,
    defense_output: ResolvedDefenseOutput,
    classifier: nn.Module,
    images: dict[str, torch.Tensor],
    labels: torch.Tensor,
) -> VisualEvidenceResult:
    out_dir = output_dir / "embedding"
    out_dir.mkdir(parents=True, exist_ok=True)
    layer = config.layers[-1] if config.layers else "penultimate"
    device = _module_device(classifier)
    settings = [
        "origin_clean",
        "origin_poisoned",
        "degradation_poisoned",
        "restoration_poisoned",
        "degradation_restoration_poisoned",
    ]
    settings = [setting for setting in settings if PIPELINE_SETTINGS[setting] in images]
    try:
        extractor = FeatureExtractor(classifier, (layer,))
        try:
            feature_rows = []
            feature_blocks = []
            for setting in settings:
                image_key = PIPELINE_SETTINGS[setting]
                extracted = _extract_layer_batched(
                    extractor,
                    images[image_key],
                    layer,
                    config.batch_size,
                    device,
                )
                feature_blocks.append(extracted)
                for sample_index in range(extracted.shape[0]):
                    feature_rows.append(
                        {
                            "sample_index": sample_index,
                            "setting": setting,
                            "true_label": int(labels[sample_index].item()),
                            "is_final_output": setting
                            in {defense_output.clean_setting, defense_output.poisoned_setting},
                        }
                    )
        finally:
            extractor.close()
    except (RuntimeError, ValueError) as exc:
        note = out_dir / "README.md"
        note.write_text(f"Feature embedding skipped: {exc}\n", encoding="utf-8")
        return VisualEvidenceResult(generated=[], skipped=[str(note.relative_to(output_dir))])

    features = torch.cat(feature_blocks, dim=0).numpy()
    coords = _project_2d(features)
    point_df = pd.DataFrame(feature_rows)
    point_df["x"] = coords[:, 0]
    point_df["y"] = coords[:, 1]
    csv_path = out_dir / "feature_embedding_points.csv"
    point_df.to_csv(csv_path, index=False)
    fig_path = out_dir / "feature_embedding.png"
    _plot_embedding(point_df, fig_path, title=f"Feature embedding ({layer})")
    return VisualEvidenceResult(
        generated=[str(csv_path.relative_to(output_dir)), str(fig_path.relative_to(output_dir))],
        skipped=[],
    )


def _write_gradcam_grid(
    output_dir: Path,
    config: CaseConfig,
    defense_output: ResolvedDefenseOutput,
    classifier: nn.Module,
    images: dict[str, torch.Tensor],
) -> VisualEvidenceResult:
    out_dir = output_dir / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)
    layer = _resolve_gradcam_layer(classifier, config.visual_gradcam_layer)
    device = _module_device(classifier)
    if layer is None:
        note = out_dir / "README.md"
        note.write_text(
            "Grad-CAM skipped: no convolutional layer was found. "
            "Pass --visual-gradcam-layer to select a layer explicitly.\n",
            encoding="utf-8",
        )
        return VisualEvidenceResult(generated=[], skipped=[str(note.relative_to(output_dir))])

    settings = [
        ("origin_poisoned", "poisoned"),
        (defense_output.poisoned_setting, defense_output.poisoned_image_key),
    ]
    if "degraded_poisoned" in images and defense_output.poisoned_image_key != "degraded_poisoned":
        settings.insert(1, ("degradation_poisoned", "degraded_poisoned"))

    generated: list[str] = []
    for setting, image_key in settings:
        if image_key not in images:
            continue
        count = min(config.visual_num_samples, images[image_key].shape[0])
        if count == 0:
            continue
        samples = images[image_key][:count].to(device)
        try:
            heatmaps = _gradcam(classifier, layer, samples)
        except RuntimeError as exc:
            note = out_dir / f"{setting}_README.md"
            note.write_text(f"Grad-CAM skipped for {setting}: {exc}\n", encoding="utf-8")
            return VisualEvidenceResult(
                generated=generated,
                skipped=[str(note.relative_to(output_dir))],
            )
        fig_path = out_dir / f"{setting}_gradcam.png"
        _plot_gradcam_grid(samples, heatmaps, fig_path, title=f"Grad-CAM: {setting}")
        generated.append(str(fig_path.relative_to(output_dir)))
    return VisualEvidenceResult(generated=generated, skipped=[])


def _write_frequency_spectrum(
    output_dir: Path,
    defense_output: ResolvedDefenseOutput,
    images: dict[str, torch.Tensor],
) -> list[str]:
    out_dir = output_dir / "frequency"
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = [
        ("clean", "clean"),
        ("poisoned", "poisoned"),
        (defense_output.poisoned_setting, defense_output.poisoned_image_key),
        (defense_output.clean_setting, defense_output.clean_image_key),
    ]
    spectra = []
    labels = []
    for setting, image_key in settings:
        if image_key not in images:
            continue
        spectra.append(_mean_log_spectrum(images[image_key]))
        labels.append(setting)
    if not spectra:
        return []
    path = out_dir / "spectrum_grid.png"
    _plot_spectrum_grid(spectra, labels, path)
    return [str(path.relative_to(output_dir))]


def _write_manifest(output_dir: Path, generated: list[str], skipped: list[str]) -> None:
    lines = [
        "# Visual Evidence",
        "",
        "These figures are BackdoorBench-style evidence views generated from mechanism-analysis tensors and CSV outputs.",
        "",
        "## Generated",
        "",
        *[f"- `{item}`" for item in generated],
        "",
        "## Skipped",
        "",
        *([f"- `{item}`" for item in skipped] if skipped else ["- None"]),
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_multi_setting_confusion(
    df: pd.DataFrame,
    output_path: Path,
    num_classes: int | None,
    title: str,
) -> None:
    settings = list(df["setting"].drop_duplicates())
    cols = min(3, len(settings))
    rows = int(np.ceil(len(settings) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, setting in zip(axes.flat, settings):
        group = df[df["setting"] == setting]
        matrix = _count_matrix(
            group["true_label"].to_numpy(dtype=np.int64),
            group["predicted_label"].to_numpy(dtype=np.int64),
            num_classes,
        )
        _draw_matrix(axis, matrix, setting, "Predicted", "True")
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_confusion(
    y_true: np.ndarray[Any, Any],
    y_pred: np.ndarray[Any, Any],
    output_path: Path,
    num_classes: int | None,
    title: str,
) -> None:
    matrix = _count_matrix(y_true, y_pred, num_classes)
    _plot_matrix(matrix, output_path, xlabel="Predicted", ylabel="True", title=title)


def _plot_matrix(
    matrix: np.ndarray[Any, Any],
    output_path: Path,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    fig, axis = plt.subplots(figsize=(7, 6))
    _draw_matrix(axis, matrix, title, xlabel, ylabel)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _draw_matrix(
    axis: plt.Axes,
    matrix: np.ndarray[Any, Any],
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    axis.imshow(matrix, cmap="Blues")
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_xticks(range(matrix.shape[1]))
    axis.set_yticks(range(matrix.shape[0]))
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = int(matrix[row, col])
            if value:
                axis.text(col, row, str(value), ha="center", va="center", fontsize=7)


def _count_matrix(
    y_true: np.ndarray[Any, Any],
    y_pred: np.ndarray[Any, Any],
    num_classes: int | None,
) -> np.ndarray[Any, Any]:
    max_label = int(max(y_true.max(initial=0), y_pred.max(initial=0)))
    size = max(num_classes or 0, max_label + 1)
    matrix = np.zeros((size, size), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def _project_2d(features: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    centered = features - features.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    if vh.shape[0] == 1:
        return np.column_stack([centered @ vh[0], np.zeros(centered.shape[0])])
    return centered @ vh[:2].T


def _batch_slices(total: int, batch_size: int) -> list[tuple[int, int]]:
    step = max(1, batch_size)
    return [(start, min(start + step, total)) for start in range(0, total, step)]


def _extract_layer_batched(
    extractor: FeatureExtractor,
    tensor: torch.Tensor,
    layer: str,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    blocks = []
    for start, end in _batch_slices(tensor.shape[0], batch_size):
        blocks.append(extractor.extract(tensor[start:end].to(device))[layer].detach().cpu())
    return torch.cat(blocks, dim=0)


def _module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _plot_embedding(point_df: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, axis = plt.subplots(figsize=(8, 6))
    for setting, group in point_df.groupby("setting"):
        axis.scatter(group["x"], group["y"], label=setting, s=24, alpha=0.78)
    axis.set_title(title)
    axis.set_xlabel("Component 1")
    axis.set_ylabel("Component 2")
    axis.legend(fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _resolve_gradcam_layer(classifier: nn.Module, requested: str | None) -> nn.Module | None:
    modules = dict(classifier.named_modules())
    if requested is not None:
        module = modules.get(requested)
        if module is None:
            raise ValueError(f"Grad-CAM layer '{requested}' not found in classifier")
        return module
    preferred = (
        "layer4.2.conv2",
        "layer4.1.conv2",
        "layer4.0.conv2",
        "layer4",
        "features",
    )
    for name in preferred:
        module = modules.get(name)
        if module is not None:
            return module
    for module in reversed(list(modules.values())):
        if isinstance(module, nn.Conv2d):
            return module
    return None


def _gradcam(classifier: nn.Module, target_layer: nn.Module, samples: torch.Tensor) -> torch.Tensor:
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        activations.append(output)

    def backward_hook(
        _module: nn.Module,
        _grad_input: tuple[torch.Tensor, ...] | torch.Tensor,
        grad_output: tuple[torch.Tensor, ...] | torch.Tensor,
    ) -> tuple[torch.Tensor, ...] | torch.Tensor | None:
        output_grad = grad_output[0] if isinstance(grad_output, tuple) else grad_output
        gradients.append(output_grad)
        return None

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    was_training = classifier.training
    classifier.eval()
    try:
        classifier.zero_grad(set_to_none=True)
        logits = classifier(samples)
        targets = logits.argmax(dim=1)
        score = logits.gather(1, targets[:, None]).sum()
        score.backward()
    finally:
        forward_handle.remove()
        backward_handle.remove()
        classifier.zero_grad(set_to_none=True)
        if was_training:
            classifier.train()
    if not activations or not gradients:
        raise RuntimeError("target layer did not produce activations and gradients")
    activation = activations[-1].detach()
    gradient = gradients[-1].detach()
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cams = torch.relu((weights * activation).sum(dim=1, keepdim=True))
    cams = F.interpolate(cams, size=samples.shape[-2:], mode="bilinear", align_corners=False)
    flat = cams.flatten(start_dim=1)
    mins = flat.min(dim=1).values[:, None, None, None]
    maxs = flat.max(dim=1).values[:, None, None, None]
    return (cams - mins) / torch.clamp(maxs - mins, min=1e-8)


def _plot_gradcam_grid(
    samples: torch.Tensor,
    heatmaps: torch.Tensor,
    output_path: Path,
    title: str,
) -> None:
    count = min(samples.shape[0], heatmaps.shape[0])
    fig, axes = plt.subplots(count, 2, figsize=(7, 3 * count), squeeze=False)
    for idx in range(count):
        image = samples[idx].detach().cpu().permute(1, 2, 0).numpy()
        heatmap = heatmaps[idx, 0].detach().cpu().numpy()
        axes[idx, 0].imshow(image)
        axes[idx, 0].axis("off")
        axes[idx, 1].imshow(image)
        axes[idx, 1].imshow(heatmap, cmap="jet", alpha=0.45)
        axes[idx, 1].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _mean_log_spectrum(images: torch.Tensor) -> np.ndarray[Any, Any]:
    gray = images.mean(dim=1)
    spectrum = torch.fft.fftshift(torch.fft.fft2(gray), dim=(-2, -1)).abs()
    spectrum = torch.log1p(spectrum).mean(dim=0)
    normalized = spectrum - spectrum.min()
    normalized = normalized / torch.clamp(normalized.max(), min=1e-8)
    return normalized.detach().cpu().numpy()


def _plot_spectrum_grid(
    spectra: list[np.ndarray[Any, Any]],
    labels: list[str],
    output_path: Path,
) -> None:
    cols = min(4, len(spectra))
    rows = int(np.ceil(len(spectra) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, spectrum, label in zip(axes.flat, spectra, labels):
        axis.imshow(spectrum, cmap="magma")
        axis.set_title(label)
        axis.axis("off")
    fig.suptitle("Mean FFT magnitude spectrum")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
