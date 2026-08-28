from __future__ import annotations

import argparse
from pathlib import Path

from artifact_common import ROOT
from reproduce_figures import collect_visual_evidence
from reproduce_tables import mechanism_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the released mechanism-analysis outputs.")
    parser.add_argument("--mechanism-root", type=Path, default=ROOT / "figures" / "mechanism_raw")
    args = parser.parse_args()

    summaries = mechanism_rows(args.mechanism_root)
    evidence = collect_visual_evidence(args.mechanism_root)
    if not summaries:
        raise RuntimeError("No mechanism-analysis cases were summarized.")
    if not evidence:
        raise RuntimeError("No mechanism visual evidence files were found.")
    print("Mechanism-analysis smoke test passed.")
    print(f"Mechanism cases: {len(summaries)}")
    print(f"Indexed evidence files: {len(evidence)}")


if __name__ == "__main__":
    main()
