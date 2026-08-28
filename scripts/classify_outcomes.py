from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from artifact_common import ROOT, classify_outcome_profile, parse_float, read_csv_rows, write_csv


PROFILES = {
    "strict": {"asr_drop": 0.6, "pa_gain": 0.4, "ca_floor": 0.05, "collapse_drop": 0.2},
    "paper_default": {"asr_drop": 0.5, "pa_gain": 0.3, "ca_floor": 0.1, "collapse_drop": 0.2},
    "permissive": {"asr_drop": 0.4, "pa_gain": 0.2, "ca_floor": 0.15, "collapse_drop": 0.25},
}


def summarize_thresholds(matrix_path: Path) -> list[dict[str, object]]:
    rows = [row for row in read_csv_rows(matrix_path) if row.get("condition") != "origin_attack"]
    summary_rows: list[dict[str, object]] = []
    for profile_name, thresholds in PROFILES.items():
        counts: Counter[str] = Counter()
        for row in rows:
            label = classify_outcome_profile(
                parse_float(row.get("delta_ca", "")),
                parse_float(row.get("delta_asr", "")),
                parse_float(row.get("delta_pa", "")),
                **thresholds,
            )
            counts[label] += 1
        summary_rows.append(
            {
                "dataset": "Imagenette2",
                "threshold_profile": profile_name,
                "recovery_count": counts["prediction_recovery"],
                "target_disruption_count": counts["target_disruption"],
                "attack_preservation_count": counts["attack_preservation"],
                "utility_collapse_count": counts["utility_collapse"],
                "mixed_count": counts["mixed"],
                "source_script_or_config": "scripts/classify_outcomes.py",
            }
        )
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute outcome threshold sensitivity.")
    parser.add_argument("--matrix", type=Path, default=ROOT / "results" / "imagenette2_full_matrix.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "threshold_sensitivity.csv")
    args = parser.parse_args()

    fields = [
        "dataset",
        "threshold_profile",
        "recovery_count",
        "target_disruption_count",
        "attack_preservation_count",
        "utility_collapse_count",
        "mixed_count",
        "source_script_or_config",
    ]
    rows = summarize_thresholds(args.matrix)
    write_csv(args.output, fields, rows)
    print(f"Wrote {len(rows)} threshold profiles to {args.output}.")


if __name__ == "__main__":
    main()
