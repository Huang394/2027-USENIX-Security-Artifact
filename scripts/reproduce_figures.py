from __future__ import annotations

import argparse
from pathlib import Path

from artifact_common import ROOT, write_csv


FIELDS = ["case_id", "evidence_type", "path"]


def collect_visual_evidence(mechanism_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for case_dir in sorted(path for path in mechanism_root.iterdir() if path.is_dir()):
        for path in sorted(case_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".csv", ".md"}:
                continue
            if not any(part in {"visual_evidence", "figures", "input_space", "trigger_representation", "feature_space"} for part in path.parts):
                continue
            evidence_type = "visual"
            if "feature_space" in path.parts:
                evidence_type = "feature_space"
            elif "input_space" in path.parts:
                evidence_type = "input_space"
            elif "trigger_representation" in path.parts:
                evidence_type = "trigger_representation"
            elif "figures" in path.parts:
                evidence_type = "case_figure"
            rows.append(
                {
                    "case_id": case_dir.name,
                    "evidence_type": evidence_type,
                    "path": path.relative_to(ROOT).as_posix(),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Index released mechanism visual evidence.")
    parser.add_argument("--mechanism-root", type=Path, default=ROOT / "figures" / "mechanism_raw")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "mechanism_visual_evidence_index.csv")
    args = parser.parse_args()

    rows = collect_visual_evidence(args.mechanism_root)
    write_csv(args.output, FIELDS, rows)
    print(f"Indexed {len(rows)} mechanism evidence files.")


if __name__ == "__main__":
    main()
