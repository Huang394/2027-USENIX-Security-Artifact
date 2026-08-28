from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]

LOG_NAME_RE = re.compile(
    r"^(?P<timestamp>\d{8}_\d{6})_"
    r"(?P<dataset>.+?)_"
    r"(?P<attack>BadNet|Blended|ISSBA|WaNet)_"
    r"(?P<pipeline>.+)_"
    r"(?P<strength>[0-9.]+)_seed(?P<seed>\d+)\.log$",
)
ACCURACY_RE = re.compile(r"Top-1 correct / Total: (?P<correct>\d+)/(?P<total>\d+), Top-1 accuracy: (?P<acc>[0-9.eE+-]+)")


@dataclass(frozen=True)
class MetricRow:
    dataset: str
    attack: str
    family: str
    pipeline: str
    degradation: str
    strength: str
    restoration_prior: str
    condition: str
    target_label: str
    poison_rate: str
    ca: str
    asr: str
    pa: str
    delta_ca: str
    delta_asr: str
    delta_pa: str
    outcome: str
    source_log: str


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def fmt_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.10g}"


def parse_float(value: str) -> float | None:
    if value == "":
        return None
    return float(value)


def classify_outcome(delta_ca: float | None, delta_asr: float | None, delta_pa: float | None) -> str:
    if delta_ca is None or delta_asr is None or delta_pa is None:
        return "unclassified"
    if delta_ca <= -0.2:
        return "utility_collapse"
    if delta_asr <= -0.5 and delta_pa >= 0.3 and delta_ca > -0.1:
        return "prediction_recovery"
    if delta_asr <= -0.5 and delta_pa < 0.3:
        return "target_disruption"
    if delta_asr > -0.2:
        return "attack_preservation"
    return "mixed"


def classify_outcome_profile(
    delta_ca: float | None,
    delta_asr: float | None,
    delta_pa: float | None,
    *,
    asr_drop: float,
    pa_gain: float,
    ca_floor: float,
    collapse_drop: float,
) -> str:
    if delta_ca is None or delta_asr is None or delta_pa is None:
        return "unclassified"
    if delta_ca <= -collapse_drop:
        return "utility_collapse"
    if delta_asr <= -asr_drop and delta_pa >= pa_gain and delta_ca > -ca_floor:
        return "prediction_recovery"
    if delta_asr <= -asr_drop and delta_pa < pa_gain:
        return "target_disruption"
    if delta_asr > -(asr_drop / 2):
        return "attack_preservation"
    return "mixed"
