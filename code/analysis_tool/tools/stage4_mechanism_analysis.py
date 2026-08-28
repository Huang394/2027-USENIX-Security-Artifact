"""CLI for backdoor defense mechanism analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage4_analysis.analyzer import Stage4Analyzer
from stage4_analysis.config import AnalysisModuleConfig, CaseConfig


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "command"):
        parser.error("a command is required: single or batch")
    configs = load_configs(args)
    for config in configs:
        Stage4Analyzer(config).run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("single", help="Run one mechanism-analysis case")
    _add_shared_runtime_args(single)
    single.add_argument("--attack", required=True)
    single.add_argument("--degradation", required=True)
    single.add_argument("--strength", type=float, required=True)
    single.add_argument("--restoration-prior", required=True)
    single.add_argument(
        "--defense-output-setting",
        default="auto",
        choices=[
            "auto",
            "degradation_poisoned",
            "restoration_poisoned",
            "degradation_restoration_poisoned",
        ],
    )
    single.add_argument("--expected-behavior")
    single.add_argument("--backdoored-model-path", type=Path, required=True)
    single.add_argument("--clean-dataset-path", type=Path, required=True)
    single.add_argument("--poisoned-dataset-path", type=Path, required=True)
    _add_external_pipeline_args(single)
    single.add_argument("--case-name")
    single.add_argument("--target-class", type=int)
    single.add_argument("--true-class-filter", type=int)
    single.add_argument("--trigger-metadata")
    single.add_argument("--analysis-modules", nargs="*", default=[])
    single.add_argument("--trigger-mask-path", type=Path)

    batch = subparsers.add_parser("batch", help="Run cases from a YAML case config")
    _add_shared_runtime_args(batch)
    batch.add_argument("--case-config", type=Path, required=True)
    return parser


def _add_shared_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--layers", nargs="*", default=["penultimate"])
    parser.add_argument("--save-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-features", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--model-arch",
        default="backdoorbox-resnet34",
        choices=[
            "backdoorbox-resnet34",
            "convir-backdoorbox-resnet34",
            "backdoorpurification-backdoorbox-resnet34",
            "resnet18",
            "resnet34",
        ],
    )
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--restoration-model-path", type=Path)
    parser.add_argument("--degradation-seed", type=int, default=1234)
    parser.add_argument(
        "--prediction-distribution",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--centroid-sample-count", type=int, default=128)
    parser.add_argument("--ca-acceptable-threshold", type=float, default=0.70)
    parser.add_argument("--asr-low-threshold", type=float, default=0.20)
    parser.add_argument("--asr-high-threshold", type=float, default=0.50)
    parser.add_argument("--pa-high-threshold", type=float, default=0.50)
    parser.add_argument("--pa-recovery-delta", type=float, default=0.20)
    parser.add_argument("--asr-drop-delta", type=float, default=0.20)
    parser.add_argument("--asr-recovery-delta", type=float, default=0.20)
    parser.add_argument("--visual-evidence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visual-gradcam-layer")
    parser.add_argument("--visual-num-samples", type=int, default=8)


def _add_external_pipeline_args(
    parser: argparse.ArgumentParser,
    require_restoration_only: bool = False,
) -> None:
    parser.add_argument("--degraded-clean-dataset-path", type=Path)
    parser.add_argument("--degraded-poisoned-dataset-path", type=Path)
    parser.add_argument("--restored-clean-dataset-path", type=Path, required=require_restoration_only)
    parser.add_argument("--restored-poisoned-dataset-path", type=Path, required=require_restoration_only)
    parser.add_argument("--restored-degraded-clean-dataset-path", type=Path)
    parser.add_argument("--restored-degraded-poisoned-dataset-path", type=Path)


def load_configs(args: argparse.Namespace) -> list[CaseConfig]:
    if args.command == "batch":
        return _load_yaml_cases(args.case_config, args)
    case_name = args.case_name or f"{args.attack}_{args.degradation}_{args.strength}_{args.restoration_prior}".replace(
        " ", "_"
    )
    return [
        CaseConfig(
            name=case_name,
            attack=args.attack,
            degradation=args.degradation,
            strength=args.strength,
            restoration_prior=args.restoration_prior,
            defense_output_setting=args.defense_output_setting,
            expected_behavior=args.expected_behavior,
            backdoored_model_path=args.backdoored_model_path,
            clean_dataset_path=args.clean_dataset_path,
            poisoned_dataset_path=args.poisoned_dataset_path,
            output_dir=args.output_dir,
            degraded_clean_dataset_path=args.degraded_clean_dataset_path,
            degraded_poisoned_dataset_path=args.degraded_poisoned_dataset_path,
            restored_clean_dataset_path=args.restored_clean_dataset_path,
            restored_poisoned_dataset_path=args.restored_poisoned_dataset_path,
            restored_degraded_clean_dataset_path=args.restored_degraded_clean_dataset_path,
            restored_degraded_poisoned_dataset_path=args.restored_degraded_poisoned_dataset_path,
            num_samples=args.num_samples,
            target_class=args.target_class,
            true_class_filter=args.true_class_filter,
            trigger_metadata=args.trigger_metadata,
            analysis_modules=tuple(AnalysisModuleConfig(name=name) for name in args.analysis_modules),
            trigger_mask_path=args.trigger_mask_path,
            layers=tuple(args.layers),
            save_images=args.save_images,
            save_features=args.save_features,
            device=args.device,
            model_arch=args.model_arch,
            num_classes=args.num_classes,
            image_size=args.image_size,
            batch_size=args.batch_size,
            restoration_model_path=args.restoration_model_path,
            degradation_seed=args.degradation_seed,
            prediction_distribution=args.prediction_distribution,
            centroid_sample_count=args.centroid_sample_count,
            ca_acceptable_threshold=args.ca_acceptable_threshold,
            asr_low_threshold=args.asr_low_threshold,
            asr_high_threshold=args.asr_high_threshold,
            pa_high_threshold=args.pa_high_threshold,
            pa_recovery_delta=args.pa_recovery_delta,
            asr_drop_delta=args.asr_drop_delta,
            asr_recovery_delta=args.asr_recovery_delta,
            visual_evidence=args.visual_evidence,
            visual_gradcam_layer=args.visual_gradcam_layer,
            visual_num_samples=args.visual_num_samples,
        )
    ]


def _load_yaml_cases(path: Path, args: argparse.Namespace) -> list[CaseConfig]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or "cases" not in raw:
        raise ValueError("Case config must contain a top-level 'cases' list")
    configs = []
    for item in raw["cases"]:
        if not isinstance(item, dict):
            raise ValueError("Each case must be a mapping")
        configs.append(_case_from_mapping(item, args))
    return configs


def _case_from_mapping(item: dict[str, Any], args: argparse.Namespace) -> CaseConfig:
    modules = tuple(
        AnalysisModuleConfig(
            name=module["name"] if isinstance(module, dict) else str(module),
            options={k: v for k, v in module.items() if k != "name"} if isinstance(module, dict) else {},
        )
        for module in item.get("analysis_modules", ())
    )
    return CaseConfig(
        name=str(item["name"]),
        attack=str(item["attack"]),
        degradation=str(item["degradation"]),
        strength=float(item["strength"]),
        restoration_prior=str(item["restoration_prior"]),
        defense_output_setting=str(item.get("defense_output_setting", "auto")),
        expected_behavior=item.get("expected_behavior"),
        backdoored_model_path=Path(item["backdoored_model_path"]),
        clean_dataset_path=Path(item["clean_dataset_path"]),
        poisoned_dataset_path=Path(item["poisoned_dataset_path"]),
        output_dir=args.output_dir,
        degraded_clean_dataset_path=_optional_path(item.get("degraded_clean_dataset_path")),
        degraded_poisoned_dataset_path=_optional_path(item.get("degraded_poisoned_dataset_path")),
        restored_clean_dataset_path=_optional_path(item.get("restored_clean_dataset_path")),
        restored_poisoned_dataset_path=_optional_path(item.get("restored_poisoned_dataset_path")),
        restored_degraded_clean_dataset_path=_optional_path(
            item.get("restored_degraded_clean_dataset_path")
        ),
        restored_degraded_poisoned_dataset_path=_optional_path(
            item.get("restored_degraded_poisoned_dataset_path")
        ),
        num_samples=int(item.get("num_samples", args.num_samples)),
        target_class=_optional_int(item.get("target_class")),
        true_class_filter=_optional_int(item.get("true_class_filter")),
        trigger_metadata=item.get("trigger_metadata"),
        analysis_modules=modules,
        trigger_mask_path=_optional_path(item.get("trigger_mask_path")),
        layers=tuple(item.get("layers", args.layers)),
        save_images=bool(item.get("save_images", args.save_images)),
        save_features=bool(item.get("save_features", args.save_features)),
        device=str(item.get("device", args.device)),
        model_arch=str(item.get("model_arch", args.model_arch)),
        num_classes=_optional_int(item.get("num_classes", args.num_classes)),
        image_size=int(item.get("image_size", args.image_size)),
        batch_size=int(item.get("batch_size", args.batch_size)),
        restoration_model_path=_optional_path(
            item.get("restoration_model_path", args.restoration_model_path)
        ),
        degradation_seed=int(item.get("degradation_seed", args.degradation_seed)),
        prediction_distribution=bool(
            item.get("prediction_distribution", args.prediction_distribution)
        ),
        centroid_sample_count=int(item.get("centroid_sample_count", args.centroid_sample_count)),
        ca_acceptable_threshold=float(
            item.get("ca_acceptable_threshold", args.ca_acceptable_threshold)
        ),
        asr_low_threshold=float(item.get("asr_low_threshold", args.asr_low_threshold)),
        asr_high_threshold=float(item.get("asr_high_threshold", args.asr_high_threshold)),
        pa_high_threshold=float(item.get("pa_high_threshold", args.pa_high_threshold)),
        pa_recovery_delta=float(item.get("pa_recovery_delta", args.pa_recovery_delta)),
        asr_drop_delta=float(item.get("asr_drop_delta", args.asr_drop_delta)),
        asr_recovery_delta=float(item.get("asr_recovery_delta", args.asr_recovery_delta)),
        visual_evidence=bool(item.get("visual_evidence", args.visual_evidence)),
        visual_gradcam_layer=item.get("visual_gradcam_layer", args.visual_gradcam_layer),
        visual_num_samples=int(item.get("visual_num_samples", args.visual_num_samples)),
    )


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_path(value: Any) -> Path | None:
    return None if value is None else Path(value)


if __name__ == "__main__":
    main()
