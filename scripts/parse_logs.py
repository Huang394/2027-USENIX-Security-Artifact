from __future__ import annotations

import argparse
from pathlib import Path

from artifact_common import ACCURACY_RE, LOG_NAME_RE, ROOT, classify_outcome, fmt_float, write_csv


BASE_FIELDS = [
    "dataset",
    "attack",
    "family",
    "pipeline",
    "degradation",
    "strength",
    "restoration_prior",
    "condition",
    "target_label",
    "poison_rate",
    "ca",
    "asr",
    "pa",
    "delta_ca",
    "delta_asr",
    "delta_pa",
    "outcome",
    "source_log",
]
CIFAR_FIELDS = [*BASE_FIELDS, "input_handling"]


def normalize_dataset(raw: str) -> str:
    if raw == "CIFAR10ImageFolder":
        return "CIFAR10"
    return raw


def infer_degradation(pipeline: str, family: str) -> str:
    if pipeline == "none" or family == "attack_log":
        return "none"
    if "motion_blur" in pipeline or family == "motion_blur":
        return "motion_blur"
    if "haze" in pipeline or family == "haze":
        return "haze"
    if "zip" in pipeline or family in {"zip", "diffusionzip"}:
        return "zip"
    if "litebd" in pipeline or family.startswith("litebd"):
        return "litebd"
    return pipeline


def infer_restoration_prior(path: Path, pipeline: str, family: str) -> str:
    parts = {part.lower() for part in path.parts}
    if "gopro_defense_log" in parts or "motion_blur_gopro_defense_log" in parts:
        return "gopro"
    if "rsblur_defense_log" in parts or "motion_blur_rsblur_defense_log" in parts:
        return "rsblur"
    if "its_defense_log" in parts or "haze_its_defense_log" in parts:
        return "its"
    if "ots_defense_log" in parts or "haze_ots_defense_log" in parts:
        return "ots"
    if "haze4k_defense_log" in parts or "haze_haze4k_defense_log" in parts:
        return "haze4k"
    if family.startswith("litebd"):
        return family
    if pipeline.startswith("litebd"):
        return pipeline
    if "zip" in pipeline or family in {"zip", "diffusionzip"}:
        return "I2I-Diffusion"
    return "none"


def infer_input_handling(family: str, pipeline: str) -> str:
    if "litebd" in family or "litebd" in pipeline:
        return "Lite-BD direct processing"
    if family in {"zip", "diffusionzip"} or "zip" in pipeline:
        return "I2I-Diffusion tiling/splitting"
    if family in {"haze", "motion_blur"}:
        return "ConvIR reflect padding/crop"
    return "native evaluation"


def parse_metric_log(path: Path, logs_root: Path) -> dict[str, object] | None:
    match = LOG_NAME_RE.match(path.name)
    if match is None:
        return None
    metrics: dict[str, float] = {}
    current: str | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "benign test dataset(CA)" in line:
            current = "ca"
        elif "poisoned test dataset(ASR)" in line:
            current = "asr"
        elif "poisoned test dataset(PA)" in line:
            current = "pa"
        elif current is not None:
            accuracy_match = ACCURACY_RE.search(line)
            if accuracy_match:
                metrics[current] = float(accuracy_match.group("acc"))
                current = None
    if not {"ca", "asr", "pa"}.issubset(metrics):
        return None

    raw_dataset = match.group("dataset")
    dataset = normalize_dataset(raw_dataset)
    relative = path.relative_to(logs_root)
    family = relative.parts[1] if relative.parts[0].lower() in {"imagenette2", "cifar10"} else relative.parts[0]
    pipeline = match.group("pipeline")
    return {
        "dataset": dataset,
        "attack": match.group("attack"),
        "family": family,
        "pipeline": pipeline,
        "degradation": infer_degradation(pipeline, family),
        "strength": match.group("strength"),
        "restoration_prior": infer_restoration_prior(relative, pipeline, family),
        "condition": "origin_attack" if family == "attack_log" else "defense",
        "target_label": "1",
        "poison_rate": "0.05",
        "ca": metrics["ca"],
        "asr": metrics["asr"],
        "pa": metrics["pa"],
        "source_log": relative.as_posix(),
        "input_handling": infer_input_handling(family, pipeline),
    }


def add_deltas(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    baselines: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        if row["condition"] == "origin_attack":
            baselines[(str(row["dataset"]), str(row["attack"]))] = row
    for row in rows:
        baseline = baselines.get((str(row["dataset"]), str(row["attack"])))
        if baseline is None:
            row["delta_ca"] = ""
            row["delta_asr"] = ""
            row["delta_pa"] = ""
            row["outcome"] = "missing_origin"
            continue
        row_ca = float(str(row["ca"]))
        row_asr = float(str(row["asr"]))
        row_pa = float(str(row["pa"]))
        baseline_ca = float(str(baseline["ca"]))
        baseline_asr = float(str(baseline["asr"]))
        baseline_pa = float(str(baseline["pa"]))
        delta_ca = row_ca - baseline_ca
        delta_asr = row_asr - baseline_asr
        delta_pa = row_pa - baseline_pa
        row["delta_ca"] = fmt_float(delta_ca)
        row["delta_asr"] = fmt_float(delta_asr)
        row["delta_pa"] = fmt_float(delta_pa)
        row["outcome"] = "origin_attack" if row["condition"] == "origin_attack" else classify_outcome(delta_ca, delta_asr, delta_pa)
        row["ca"] = fmt_float(row_ca)
        row["asr"] = fmt_float(row_asr)
        row["pa"] = fmt_float(row_pa)
    return rows


def parse_all(logs_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(logs_root.rglob("*.log")):
        row = parse_metric_log(path, logs_root)
        if row is not None:
            rows.append(row)
    return add_deltas(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse paper-level CA/ASR/PA metrics from released logs.")
    parser.add_argument("--logs-root", type=Path, default=ROOT / "logs")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    rows = parse_all(args.logs_root)
    imagenette_rows = [row for row in rows if row["dataset"] == "Imagenette2"]
    cifar_rows = [row for row in rows if row["dataset"] == "CIFAR10"]
    write_csv(args.results_root / "imagenette2_full_matrix.csv", BASE_FIELDS, imagenette_rows)
    write_csv(args.results_root / "cifar10_external_check.csv", CIFAR_FIELDS, cifar_rows)
    print(f"Wrote {len(imagenette_rows)} Imagenette2 rows and {len(cifar_rows)} CIFAR10 rows.")


if __name__ == "__main__":
    main()
