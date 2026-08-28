from __future__ import annotations

import argparse
from pathlib import Path

from artifact_common import ROOT, fmt_float, load_json, read_csv_rows, write_csv


BADNET_FIELDS = [
    "dataset",
    "attack",
    "pipeline",
    "optim_seed",
    "steps",
    "lr",
    "search_samples",
    "eval_samples",
    "original_asr_before_pipeline",
    "original_asr_after_pipeline",
    "adaptive_asr_after_pipeline",
    "adaptive_minus_original",
    "clean_pipeline_accuracy",
    "source_json",
]
BLENDED_FIELDS = [
    *BADNET_FIELDS,
    "notes",
]
MECHANISM_FIELDS = [
    "case_id",
    "dataset",
    "attack",
    "pipeline",
    "diagnostic_samples",
    "delta_target_probability",
    "delta_true_class_probability",
    "final_margin",
    "feature_ratio",
    "source_case_dir",
]


def adaptive_rows(adaptive_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    badnet: list[dict[str, object]] = []
    blended: list[dict[str, object]] = []
    for path in sorted(adaptive_root.glob("*.json")):
        data = load_json(path)
        config = data.get("config", {})
        results = data.get("results", {})
        if not isinstance(config, dict) or not isinstance(results, dict):
            continue
        row = {
            "dataset": "CIFAR10" if config.get("dataset") == "CIFAR10ImageFolder" else str(config.get("dataset", "")),
            "attack": "BadNet" if "BadNet" in path.name else "Blended",
            "pipeline": str(results.get("metric", "PABR")),
            "optim_seed": config.get("optim_seed", ""),
            "steps": config.get("steps", ""),
            "lr": config.get("lr", ""),
            "search_samples": config.get("search_samples", ""),
            "eval_samples": results.get("main_adaptive_eval_total", config.get("eval_samples", "")),
            "original_asr_before_pipeline": fmt_float(float(results["original_asr_before_pipeline"])),
            "original_asr_after_pipeline": fmt_float(float(results["main_original_asr_after_pipeline"])),
            "adaptive_asr_after_pipeline": fmt_float(float(results["main_adaptive_asr_after_pipeline"])),
            "adaptive_minus_original": fmt_float(float(results["main_adaptive_minus_original"])),
            "clean_pipeline_accuracy": fmt_float(float(results["clean_pipeline_accuracy_on_eval_subset"])),
            "source_json": path.relative_to(ROOT).as_posix(),
        }
        if row["attack"] == "BadNet":
            badnet.append(row)
        else:
            row["notes"] = "Supplementary Blended adaptive observation; not part of the BadNet seed table."
            blended.append(row)
    return badnet, blended


def row_by_setting(rows: list[dict[str, str]], setting: str) -> dict[str, str] | None:
    for row in rows:
        if row.get("setting") == setting:
            return row
    return None


def mechanism_rows(cases_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case_dir in sorted(path for path in cases_root.iterdir() if path.is_dir()):
        metrics_path = case_dir / "metrics_summary.csv"
        config_path = case_dir / "case_config.json"
        if not metrics_path.exists() or not config_path.exists():
            continue
        metrics = read_csv_rows(metrics_path)
        config = load_json(config_path)
        origin = row_by_setting(metrics, "origin_poisoned")
        final = row_by_setting(metrics, "degradation_restoration_poisoned") or row_by_setting(metrics, "restoration_poisoned")
        feature_path = case_dir / "feature_space" / "feature_distance.csv"
        feature_ratio = ""
        if feature_path.exists():
            feature_rows = read_csv_rows(feature_path)
            ratios: list[float] = []
            for feature_row in feature_rows:
                restored = feature_row.get("dist_restored_poison_to_clean_centroid", "")
                poisoned = feature_row.get("dist_poison_to_clean_centroid", "")
                if restored and poisoned and float(poisoned) != 0:
                    ratios.append(float(restored) / float(poisoned))
            if ratios:
                feature_ratio = fmt_float(sum(ratios) / len(ratios))
        if origin is None or final is None:
            continue
        origin_target = float(origin["mean_target_probability"])
        final_target = float(final["mean_target_probability"])
        origin_true = float(origin["mean_true_class_probability"])
        final_true = float(final["mean_true_class_probability"])
        rows.append(
            {
                "case_id": case_dir.name,
                "dataset": str(config.get("dataset", "Imagenette2")),
                "attack": str(config.get("attack", "")),
                "pipeline": str(config.get("defense_output_setting", config.get("pipeline", ""))),
                "diagnostic_samples": final.get("num_samples", ""),
                "delta_target_probability": fmt_float(final_target - origin_target),
                "delta_true_class_probability": fmt_float(final_true - origin_true),
                "final_margin": final.get("mean_target_margin", ""),
                "feature_ratio": feature_ratio,
                "source_case_dir": case_dir.relative_to(ROOT).as_posix(),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild adaptive and mechanism summary tables.")
    parser.add_argument("--adaptive-root", type=Path, default=ROOT / "logs" / "adaptive_pipeline")
    parser.add_argument("--mechanism-root", type=Path, default=ROOT / "figures" / "mechanism_raw")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    badnet, blended = adaptive_rows(args.adaptive_root)
    write_csv(args.results_root / "adaptive_badnet_seeds.csv", BADNET_FIELDS, badnet)
    write_csv(args.results_root / "adaptive_blended_observation.csv", BLENDED_FIELDS, blended)
    mechanism = mechanism_rows(args.mechanism_root)
    write_csv(args.results_root / "mechanism_summary.csv", MECHANISM_FIELDS, mechanism)
    print(f"Wrote {len(badnet)} BadNet adaptive rows, {len(blended)} Blended rows, and {len(mechanism)} mechanism rows.")


if __name__ == "__main__":
    main()
