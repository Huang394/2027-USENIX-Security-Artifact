"""Configuration objects for mechanism analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AnalysisModuleConfig:
    """Configuration for one trigger-representation analysis module."""

    name: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CaseConfig:
    """A single Attack x Degradation x Restoration Prior analysis case."""

    name: str
    attack: str
    degradation: str
    strength: float
    restoration_prior: str
    backdoored_model_path: Path
    clean_dataset_path: Path
    poisoned_dataset_path: Path
    output_dir: Path
    defense_output_setting: str = "auto"
    expected_behavior: str | None = None
    degraded_clean_dataset_path: Path | None = None
    degraded_poisoned_dataset_path: Path | None = None
    restored_clean_dataset_path: Path | None = None
    restored_poisoned_dataset_path: Path | None = None
    restored_degraded_clean_dataset_path: Path | None = None
    restored_degraded_poisoned_dataset_path: Path | None = None
    num_samples: int = 128
    target_class: int | None = None
    true_class_filter: int | None = None
    trigger_metadata: str | None = None
    analysis_modules: tuple[AnalysisModuleConfig, ...] = ()
    trigger_mask_path: Path | None = None
    layers: tuple[str, ...] = ("penultimate",)
    save_images: bool = True
    save_features: bool = False
    device: str = "auto"
    model_arch: str = "backdoorbox-resnet34"
    num_classes: int | None = 20
    image_size: int = 256
    batch_size: int = 32
    restoration_model_path: Path | None = None
    degradation_seed: int = 1234
    prediction_distribution: bool = True
    centroid_sample_count: int = 128
    ca_acceptable_threshold: float = 0.70
    asr_low_threshold: float = 0.20
    asr_high_threshold: float = 0.50
    pa_high_threshold: float = 0.50
    pa_recovery_delta: float = 0.20
    asr_drop_delta: float = 0.20
    asr_recovery_delta: float = 0.20
    visual_evidence: bool = True
    visual_gradcam_layer: str | None = None
    visual_num_samples: int = 8

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in (
            "backdoored_model_path",
            "clean_dataset_path",
            "poisoned_dataset_path",
            "output_dir",
            "degraded_clean_dataset_path",
            "degraded_poisoned_dataset_path",
            "restored_clean_dataset_path",
            "restored_poisoned_dataset_path",
            "restored_degraded_clean_dataset_path",
            "restored_degraded_poisoned_dataset_path",
            "trigger_mask_path",
            "restoration_model_path",
        ):
            if data[key] is not None:
                data[key] = str(data[key])
        return data


@dataclass(frozen=True)
class DecisionThresholds:
    """Thresholds used by the v3.0 behavior decision module."""

    ca_acceptable_threshold: float = 0.70
    asr_low_threshold: float = 0.20
    asr_high_threshold: float = 0.50
    pa_high_threshold: float = 0.50
    pa_recovery_delta: float = 0.20
    asr_drop_delta: float = 0.20
    asr_recovery_delta: float = 0.20


def thresholds_from_config(config: CaseConfig) -> DecisionThresholds:
    return DecisionThresholds(
        ca_acceptable_threshold=config.ca_acceptable_threshold,
        asr_low_threshold=config.asr_low_threshold,
        asr_high_threshold=config.asr_high_threshold,
        pa_high_threshold=config.pa_high_threshold,
        pa_recovery_delta=config.pa_recovery_delta,
        asr_drop_delta=config.asr_drop_delta,
        asr_recovery_delta=config.asr_recovery_delta,
    )


DEFAULT_MODULES_BY_TRIGGER_METADATA: dict[str, tuple[str, ...]] = {
    "localized_patch": ("local_patch", "model_output", "feature_space"),
    "distributed_global": ("frequency_global", "input_space", "feature_space"),
    "geometric_warp": ("geometry_structure", "model_output", "feature_space"),
    "invisible_sample_specific": (
        "feature_invisible",
        "frequency_global",
        "model_output",
    ),
    "semantic_natural": ("semantic_natural", "feature_space", "model_output"),
}


def resolve_analysis_modules(
    explicit_modules: tuple[AnalysisModuleConfig, ...],
    trigger_metadata: str | None,
) -> tuple[AnalysisModuleConfig, ...]:
    """Resolve analysis modules without relying on attack name."""

    if explicit_modules:
        return explicit_modules
    if trigger_metadata in DEFAULT_MODULES_BY_TRIGGER_METADATA:
        return tuple(
            AnalysisModuleConfig(name=name)
            for name in DEFAULT_MODULES_BY_TRIGGER_METADATA[trigger_metadata]
        )
    return (
        AnalysisModuleConfig(name="input_space"),
        AnalysisModuleConfig(name="model_output"),
        AnalysisModuleConfig(name="feature_space"),
    )
