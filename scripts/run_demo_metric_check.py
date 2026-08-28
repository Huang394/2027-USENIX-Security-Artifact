from __future__ import annotations

import argparse
from pathlib import Path

from artifact_common import ROOT, read_csv_rows
from classify_outcomes import summarize_thresholds
from parse_logs import parse_all
from reproduce_tables import adaptive_rows, mechanism_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a lightweight artifact smoke test without full datasets.")
    parser.add_argument("--logs-root", type=Path, default=ROOT / "logs")
    parser.add_argument("--adaptive-root", type=Path, default=ROOT / "logs" / "adaptive_pipeline")
    parser.add_argument("--mechanism-root", type=Path, default=ROOT / "figures" / "mechanism_raw")
    args = parser.parse_args()

    metric_rows = parse_all(args.logs_root)
    imagenette_rows = [row for row in metric_rows if row["dataset"] == "Imagenette2"]
    cifar_rows = [row for row in metric_rows if row["dataset"] == "CIFAR10"]
    if not imagenette_rows:
        raise RuntimeError("No Imagenette2 metric rows were parsed.")
    if not cifar_rows:
        raise RuntimeError("No CIFAR10 metric rows were parsed.")

    badnet_rows, blended_rows = adaptive_rows(args.adaptive_root)
    if not badnet_rows:
        raise RuntimeError("No adaptive BadNet JSON rows were parsed.")
    if not blended_rows:
        raise RuntimeError("No adaptive Blended JSON rows were parsed.")

    mechanism = mechanism_rows(args.mechanism_root)
    if not mechanism:
        raise RuntimeError("No mechanism summary rows were parsed.")

    matrix_path = ROOT / "results" / "imagenette2_full_matrix.csv"
    if matrix_path.exists():
        threshold_rows = summarize_thresholds(matrix_path)
        if not threshold_rows:
            raise RuntimeError("Threshold sensitivity produced no rows.")
    else:
        threshold_rows = []

    generated = ROOT / "results" / "mechanism_summary.csv"
    generated_rows = read_csv_rows(generated) if generated.exists() else []
    print("Smoke test passed.")
    print(f"Parsed metric rows: Imagenette2={len(imagenette_rows)}, CIFAR10={len(cifar_rows)}")
    print(f"Parsed adaptive rows: BadNet={len(badnet_rows)}, Blended={len(blended_rows)}")
    print(f"Parsed mechanism rows: {len(mechanism)}")
    print(f"Existing threshold profiles: {len(threshold_rows)}")
    print(f"Existing generated mechanism_summary.csv rows: {len(generated_rows)}")


if __name__ == "__main__":
    main()
