from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any

from artifact_common import ROOT


ABSOLUTE_PATH_RE = re.compile(r"([A-Za-z]:\\Users\\[^\s,\"')]+|/home/[^\s,\"')]+)")
NOT_REDISTRIBUTED = "NOT_REDISTRIBUTED: see REPRODUCIBILITY.md external assets section"


def sanitize_value(key: str, value: Any) -> Any:
    if isinstance(value, dict):
        return {nested_key: sanitize_value(nested_key, nested_value) for nested_key, nested_value in value.items()}
    if isinstance(value, list):
        return [sanitize_value(key, item) for item in value]
    if not isinstance(value, str):
        return value
    if not ABSOLUTE_PATH_RE.search(value):
        return value
    if key == "output_dir":
        return "figures/mechanism_raw"
    if key.endswith("_dataset_path"):
        return "NOT_REDISTRIBUTED: full dataset-derived images are documented in REPRODUCIBILITY.md"
    if key.endswith("_model_path") or key.endswith("_ckpt") or key.endswith("_checkpoint"):
        return NOT_REDISTRIBUTED
    if key.endswith("_mask_path"):
        return "code/analysis_tool/masks or documented external trigger mask path"
    return NOT_REDISTRIBUTED


def sanitize_json_file(path: Path) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    sanitized = sanitize_value("", data)
    if sanitized == data:
        return False
    path.write_text(json.dumps(sanitized, indent=2) + "\n", encoding="utf-8")
    return True


def placeholder_for_match(match: re.Match[str]) -> str:
    raw = match.group(0)
    if raw.startswith("/"):
        name = PurePosixPath(raw).name
    else:
        name = PureWindowsPath(raw).name
    return f"EXTERNAL_SAMPLE/{name}" if name else "EXTERNAL_SAMPLE"


def sanitize_text_file(path: Path) -> bool:
    original = path.read_text(encoding="utf-8", errors="replace")
    sanitized = ABSOLUTE_PATH_RE.sub(placeholder_for_match, original)
    if sanitized == original:
        return False
    path.write_text(sanitized, encoding="utf-8")
    return True


def scan_paths(root: Path) -> list[Path]:
    candidates = [
        *root.rglob("*.json"),
        *root.rglob("*.md"),
        *root.rglob("*.py"),
        *root.rglob("*.yaml"),
        *root.rglob("*.yml"),
        *root.rglob("*.csv"),
    ]
    hits: list[Path] = []
    for path in candidates:
        if path.name == "sanitize_release_paths.py":
            continue
        if ABSOLUTE_PATH_RE.search(path.read_text(encoding="utf-8", errors="replace")):
            hits.append(path)
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanitize user-local absolute paths in the release copy.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check", action="store_true", help="Only report remaining absolute paths.")
    args = parser.parse_args()

    if not args.check:
        changed = 0
        for path in sorted((args.root / "figures" / "mechanism_raw").rglob("case_config.json")):
            if sanitize_json_file(path):
                changed += 1
        text_changed = 0
        for path in sorted((args.root / "figures" / "mechanism_raw").rglob("*.csv")):
            if sanitize_text_file(path):
                text_changed += 1
        print(f"Sanitized {changed} mechanism case_config.json files.")
        print(f"Sanitized {text_changed} mechanism CSV files.")

    hits = scan_paths(args.root)
    if hits:
        print("Remaining files with absolute paths:")
        for path in hits:
            print(path.relative_to(args.root).as_posix())
        raise SystemExit(1 if args.check else 0)
    print("No absolute paths found in scanned release text files.")


if __name__ == "__main__":
    main()
