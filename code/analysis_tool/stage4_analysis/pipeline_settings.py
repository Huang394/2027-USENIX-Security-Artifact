"""Canonical pipeline setting names and v3.0 defense-output resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from stage4_analysis.config import CaseConfig


PIPELINE_SETTINGS: dict[str, str] = {
    "origin_clean": "clean",
    "origin_poisoned": "poisoned",
    "degradation_clean": "degraded_clean",
    "degradation_poisoned": "degraded_poisoned",
    "restoration_clean": "restored_clean",
    "restoration_poisoned": "restored_poisoned",
    "degradation_restoration_clean": "restored_degraded_clean",
    "degradation_restoration_poisoned": "restored_degraded_poisoned",
}

VALID_DEFENSE_OUTPUT_SETTINGS = {
    "auto",
    "degradation_poisoned",
    "restoration_poisoned",
    "degradation_restoration_poisoned",
}

FINAL_CLEAN_BY_POISONED_SETTING = {
    "degradation_poisoned": "degradation_clean",
    "restoration_poisoned": "restoration_clean",
    "degradation_restoration_poisoned": "degradation_restoration_clean",
}


@dataclass(frozen=True)
class ResolvedDefenseOutput:
    poisoned_setting: str
    clean_setting: str
    poisoned_image_key: str
    clean_image_key: str


def resolve_defense_output_setting(config: CaseConfig) -> ResolvedDefenseOutput:
    setting = config.defense_output_setting
    if setting not in VALID_DEFENSE_OUTPUT_SETTINGS:
        valid = ", ".join(sorted(VALID_DEFENSE_OUTPUT_SETTINGS))
        raise ValueError(f"Unsupported defense_output_setting '{setting}'. Valid values: {valid}")
    if setting == "auto":
        setting = _auto_defense_output_setting(config)
    clean_setting = FINAL_CLEAN_BY_POISONED_SETTING[setting]
    return ResolvedDefenseOutput(
        poisoned_setting=setting,
        clean_setting=clean_setting,
        poisoned_image_key=PIPELINE_SETTINGS[setting],
        clean_image_key=PIPELINE_SETTINGS[clean_setting],
    )


def validate_required_inputs(config: CaseConfig, resolved: ResolvedDefenseOutput) -> None:
    required_paths = _required_paths_for_setting(config, resolved.poisoned_setting)
    missing = [flag for flag, value in required_paths if value is None]
    if missing:
        raise ValueError(
            f"Defense output setting '{resolved.poisoned_setting}' requires: "
            + ", ".join(missing)
        )


def _auto_defense_output_setting(config: CaseConfig) -> str:
    if config.restored_degraded_poisoned_dataset_path is not None:
        return "degradation_restoration_poisoned"
    restoration_prior = config.restoration_prior.strip().lower()
    if restoration_prior in {"none", "degradation_only", "blur_only", "motion_blur_only"}:
        return "degradation_poisoned"
    return "restoration_poisoned"


def _required_paths_for_setting(
    config: CaseConfig,
    setting: str,
) -> tuple[tuple[str, Path | None], ...]:
    if setting == "degradation_poisoned":
        return (
            ("--degraded-clean-dataset-path", config.degraded_clean_dataset_path),
            ("--degraded-poisoned-dataset-path", config.degraded_poisoned_dataset_path),
        )
    if setting == "restoration_poisoned":
        return (
            ("--restored-clean-dataset-path", config.restored_clean_dataset_path),
            ("--restored-poisoned-dataset-path", config.restored_poisoned_dataset_path),
        )
    return (
        (
            "--restored-degraded-clean-dataset-path",
            config.restored_degraded_clean_dataset_path,
        ),
        (
            "--restored-degraded-poisoned-dataset-path",
            config.restored_degraded_poisoned_dataset_path,
        ),
    )
