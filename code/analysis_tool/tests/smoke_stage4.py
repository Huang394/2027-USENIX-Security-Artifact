"""Smoke test for the mechanism analysis CLI."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import importlib.util
from pathlib import Path
from types import ModuleType

from PIL import Image
import torch
from torch import nn


class ToyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(3, 2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pooled = self.avgpool(images).flatten(1)
        return self.fc(pooled)


def _write_dataset(root: Path) -> None:
    root.mkdir(parents=True)
    rows = ["filename,label"]
    for index in range(4):
        value = 40 + index * 40
        image = Image.new("RGB", (16, 16), (value, 255 - value, 80))
        filename = f"sample_{index}.png"
        image.save(root / filename)
        rows.append(f"{filename},{index % 2}")
    (root / "labels.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _check_backdoorbox_checkpoint_loading(project_root: Path, temp_root: Path) -> None:
    sys.path.insert(0, str(project_root))
    from stage4_analysis.models import FeatureExtractor, _build_architecture, load_classifier

    checkpoint_path = temp_root / "backdoorbox_module_prefix.pth"
    source_model = _build_architecture("backdoorbox-resnet34", 2)
    prefixed_state_dict = {
        f"module.{key}": value
        for key, value in source_model.state_dict().items()
    }
    torch.save(prefixed_state_dict, checkpoint_path)

    device = torch.device("cpu")
    loaded_model = load_classifier(checkpoint_path, device, "backdoorbox-resnet34", 2)
    extractor = FeatureExtractor(loaded_model, ("penultimate",))
    try:
        features = extractor.extract(torch.rand(1, 3, 16, 16))
    finally:
        extractor.close()
    if features["penultimate"].shape != (1, 512):
        raise AssertionError("BackdoorBox penultimate feature shape mismatch")


def _load_resnet_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"source_resnet_{abs(hash(path))}", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to import ResNet module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_source_resnet_parity(project_root: Path) -> None:
    sys.path.insert(0, str(project_root))
    from stage4_analysis.models import _build_architecture

    models_root = project_root.parent
    source_paths = [
        models_root / "ConvIR-ZIP" / "attack" / "BackdoorBox" / "core" / "models" / "resnet.py",
        models_root
        / "BackdoorPurification"
        / "attack"
        / "BackdoorBox"
        / "core"
        / "models"
        / "resnet.py",
    ]
    local_model = _build_architecture("backdoorbox-resnet34", 20)
    local_keys = tuple(local_model.state_dict().keys())
    sample = torch.rand(2, 3, 16, 16)
    local_shape = tuple(local_model(sample).shape)
    for source_path in source_paths:
        if not source_path.exists():
            continue
        source_module = _load_resnet_module(source_path)
        source_model = source_module.ResNet(34, num_classes=20)
        source_keys = tuple(source_model.state_dict().keys())
        if source_keys != local_keys:
            raise AssertionError(f"BackdoorBox ResNet key mismatch against {source_path}")
        if tuple(source_model(sample).shape) != local_shape:
            raise AssertionError(f"BackdoorBox ResNet output shape mismatch against {source_path}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    temp_root = Path(tempfile.mkdtemp(prefix="stage4_smoke_"))
    try:
        _check_source_resnet_parity(project_root)
        _check_backdoorbox_checkpoint_loading(project_root, temp_root)
        clean = temp_root / "clean"
        poisoned = temp_root / "poisoned"
        restored_clean = temp_root / "restored_clean"
        restored_poisoned = temp_root / "restored_poisoned"
        degraded_clean = temp_root / "degraded_clean"
        degraded_poisoned = temp_root / "degraded_poisoned"
        restored_degraded_clean = temp_root / "restored_degraded_clean"
        restored_degraded_poisoned = temp_root / "restored_degraded_poisoned"
        output = temp_root / "out"
        model_path = temp_root / "toy_classifier.pt"
        _write_dataset(clean)
        _write_dataset(poisoned)
        _write_dataset(degraded_clean)
        _write_dataset(degraded_poisoned)
        _write_dataset(restored_clean)
        _write_dataset(restored_poisoned)
        _write_dataset(restored_degraded_clean)
        _write_dataset(restored_degraded_poisoned)
        torch.jit.script(ToyClassifier()).save(str(model_path))
        command = [
            sys.executable,
            str(project_root / "tools" / "stage4_mechanism_analysis.py"),
            "single",
            "--attack",
            "Smoke",
            "--degradation",
            "haze",
            "--strength",
            "0.1",
            "--restoration-prior",
            "identity",
            "--defense-output-setting",
            "degradation_restoration_poisoned",
            "--expected-behavior",
            "ambiguous",
            "--analysis-modules",
            "input_space",
            "frequency_global",
            "model_output",
            "feature_space",
            "--backdoored-model-path",
            str(model_path),
            "--clean-dataset-path",
            str(clean),
            "--poisoned-dataset-path",
            str(poisoned),
            "--restored-clean-dataset-path",
            str(restored_clean),
            "--restored-poisoned-dataset-path",
            str(restored_poisoned),
            "--restored-degraded-clean-dataset-path",
            str(restored_degraded_clean),
            "--restored-degraded-poisoned-dataset-path",
            str(restored_degraded_poisoned),
            "--output-dir",
            str(output),
            "--num-samples",
            "4",
            "--target-class",
            "1",
            "--image-size",
            "16",
        ]
        subprocess.run(command, check=True)
        case_dir = output / "Smoke_haze_0.1_identity"
        expected = [
            "case_config.json",
            "metrics_summary.csv",
            "behavior_decision.json",
            "figures/sample_grid.png",
            "input_space/input_residual_metrics.csv",
            "model_output/logit_trajectory.csv",
            "model_output/prediction_distribution.csv",
            "model_output/prediction_flip_summary.csv",
            "model_output/pairwise_pipeline_comparison.csv",
            "failure_cases/sample_outcome_matrix.csv",
            "failure_cases/failure_case_summary.csv",
            "failure_cases/attack_failed.csv",
            "failure_cases/final_poisoned_recovered.csv",
            "feature_space/feature_distance.csv",
            "feature_space/feature_pairwise_delta.csv",
            "trigger_representation/frequency_global/frequency_band_energy.csv",
            "visual_evidence/README.md",
            "visual_evidence/confusion_matrix/poisoned_pipeline_confusion.png",
            "visual_evidence/confusion_matrix/final_output_confusion.png",
            "visual_evidence/prediction_flow/origin_to_final_prediction_flow.png",
            "visual_evidence/embedding/feature_embedding.png",
            "visual_evidence/frequency/spectrum_grid.png",
            "stage4_summary.md",
        ]
        missing = [name for name in expected if not (case_dir / name).exists()]
        if missing:
            raise AssertionError(f"Missing outputs: {missing}")
        degradation_only_output = temp_root / "out_degradation_only"
        degradation_only_command = [
            sys.executable,
            str(project_root / "tools" / "stage4_mechanism_analysis.py"),
            "single",
            "--attack",
            "Smoke",
            "--degradation",
            "motion_blur",
            "--strength",
            "3.0",
            "--restoration-prior",
            "none",
            "--defense-output-setting",
            "degradation_poisoned",
            "--analysis-modules",
            "model_output",
            "feature_space",
            "--backdoored-model-path",
            str(model_path),
            "--clean-dataset-path",
            str(clean),
            "--poisoned-dataset-path",
            str(poisoned),
            "--degraded-clean-dataset-path",
            str(degraded_clean),
            "--degraded-poisoned-dataset-path",
            str(degraded_poisoned),
            "--output-dir",
            str(degradation_only_output),
            "--num-samples",
            "4",
            "--target-class",
            "1",
            "--image-size",
            "16",
        ]
        subprocess.run(degradation_only_command, check=True)
        degradation_case_dir = degradation_only_output / "Smoke_motion_blur_3.0_none"
        degradation_expected = [
            "case_config.json",
            "metrics_summary.csv",
            "behavior_decision.json",
            "model_output/prediction_distribution.csv",
            "model_output/prediction_flip_summary.csv",
            "model_output/pairwise_pipeline_comparison.csv",
            "failure_cases/final_poisoned_recovered.csv",
            "feature_space/feature_distance.csv",
            "visual_evidence/README.md",
            "visual_evidence/confusion_matrix/final_output_confusion.png",
            "visual_evidence/prediction_flow/origin_to_final_prediction_flow.png",
            "visual_evidence/embedding/feature_embedding.png",
            "visual_evidence/frequency/spectrum_grid.png",
            "stage4_summary.md",
        ]
        degradation_missing = [
            name for name in degradation_expected if not (degradation_case_dir / name).exists()
        ]
        if degradation_missing:
            raise AssertionError(f"Missing degradation-only outputs: {degradation_missing}")
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
