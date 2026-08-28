"""v3.0 behavior decision for restoration-pipeline outcome interpretation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from stage4_analysis.config import DecisionThresholds


@dataclass(frozen=True)
class BehaviorDecision:
    label: str
    rationale: list[str]
    caveats: list[str]
    evidence: dict[str, float | str | None]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def decide_behavior(
    metrics_summary: pd.DataFrame,
    pairwise_comparison: pd.DataFrame,
    defense_output_setting: str,
    final_clean_setting: str,
    thresholds: DecisionThresholds,
    target_class: int | None,
    expected_behavior: str | None,
) -> BehaviorDecision:
    evidence = _collect_evidence(
        metrics_summary,
        pairwise_comparison,
        defense_output_setting,
        final_clean_setting,
    )
    rationale: list[str] = []
    caveats: list[str] = []
    if expected_behavior is not None:
        evidence["expected_behavior"] = expected_behavior

    if target_class is None:
        caveats.append("target_class is missing, so ASR and target flip taxonomy are partial.")
        return BehaviorDecision("ambiguous", rationale, caveats, evidence)

    final_ca = _float_or_none(evidence.get("final_ca"))
    final_asr = _float_or_none(evidence.get("final_asr"))
    final_pa = _float_or_none(evidence.get("final_pa"))
    origin_asr = _float_or_none(evidence.get("origin_asr"))
    origin_pa = _float_or_none(evidence.get("origin_pa"))
    degradation_asr = _float_or_none(evidence.get("degradation_asr"))
    rd_asr = _float_or_none(evidence.get("rd_asr"))

    if final_ca is not None and final_ca < thresholds.ca_acceptable_threshold:
        rationale.append(
            f"Final clean accuracy {final_ca:.4f} is below "
            f"{thresholds.ca_acceptable_threshold:.4f}."
        )
        return BehaviorDecision(
            "destroy_or_low_utility_suppression",
            rationale,
            caveats,
            evidence,
        )

    if (
        defense_output_setting == "degradation_restoration_poisoned"
        and degradation_asr is not None
        and rd_asr is not None
        and rd_asr - degradation_asr >= thresholds.asr_recovery_delta
    ):
        rationale.append(
            "Restoration after degradation increases ASR relative to degradation-only "
            f"by {rd_asr - degradation_asr:.4f}."
        )
        caveats.append(
            "This is an attack-effect recovery signal, not trigger recovery proof without "
            "geometry, feature, or residual evidence."
        )
        return BehaviorDecision(
            "attack_effect_recovery_or_counterproductive",
            rationale,
            caveats,
            evidence,
        )

    if origin_asr is not None and final_asr is not None:
        asr_drop = origin_asr - final_asr
        pa_delta = None if origin_pa is None or final_pa is None else final_pa - origin_pa
        evidence["asr_delta_origin_to_final"] = asr_drop
        evidence["pa_delta_origin_to_final"] = pa_delta
        if asr_drop >= thresholds.asr_drop_delta and (
            (final_pa is not None and final_pa >= thresholds.pa_high_threshold)
            or (pa_delta is not None and pa_delta >= thresholds.pa_recovery_delta)
        ):
            rationale.append(
                f"ASR drops by {asr_drop:.4f} and poisoned accuracy recovers "
                "or reaches the high-PA threshold."
            )
            return BehaviorDecision("clean_preserving_suppress", rationale, caveats, evidence)
        if (
            asr_drop >= thresholds.asr_drop_delta
            and final_pa is not None
            and final_pa < thresholds.pa_high_threshold
        ):
            rationale.append(
                f"ASR drops by {asr_drop:.4f}, but final poisoned accuracy "
                f"{final_pa:.4f} remains below {thresholds.pa_high_threshold:.4f}."
            )
            return BehaviorDecision(
                "target_disruption_without_clean_recovery",
                rationale,
                caveats,
                evidence,
            )

    if (
        final_asr is not None
        and final_ca is not None
        and final_asr >= thresholds.asr_high_threshold
        and final_ca >= thresholds.ca_acceptable_threshold
    ):
        rationale.append(
            f"Final ASR {final_asr:.4f} remains high while final CA "
            f"{final_ca:.4f} is acceptable."
        )
        return BehaviorDecision("preserve_or_weak_effect", rationale, caveats, evidence)

    caveats.append("Metric pattern is mixed; inspect distribution, flips, residual, and feature evidence.")
    return BehaviorDecision("ambiguous", rationale, caveats, evidence)


def _collect_evidence(
    metrics_summary: pd.DataFrame,
    pairwise_comparison: pd.DataFrame,
    defense_output_setting: str,
    final_clean_setting: str,
) -> dict[str, float | str | None]:
    del pairwise_comparison
    final_poisoned = _row_for_setting(metrics_summary, defense_output_setting)
    final_clean = _row_for_setting(metrics_summary, final_clean_setting)
    origin_poisoned = _row_for_setting(metrics_summary, "origin_poisoned")
    degradation_poisoned = _row_for_setting(metrics_summary, "degradation_poisoned")
    rd_poisoned = _row_for_setting(metrics_summary, "degradation_restoration_poisoned")
    return {
        "defense_output_setting": defense_output_setting,
        "final_clean_setting": final_clean_setting,
        "final_ca": _metric_value(final_clean, "ca"),
        "final_asr": _metric_value(final_poisoned, "asr"),
        "final_pa": _metric_value(final_poisoned, "pa"),
        "origin_asr": _metric_value(origin_poisoned, "asr"),
        "origin_pa": _metric_value(origin_poisoned, "pa"),
        "degradation_asr": _metric_value(degradation_poisoned, "asr"),
        "rd_asr": _metric_value(rd_poisoned, "asr"),
    }


def _row_for_setting(df: pd.DataFrame, setting: str) -> pd.Series | None:
    rows = df[df["setting"] == setting]
    if rows.empty:
        return None
    return rows.iloc[0]


def _metric_value(row: pd.Series | None, name: str) -> float | None:
    if row is None or name not in row or pd.isna(row[name]):
        return None
    return float(row[name])


def _float_or_none(value: float | str | None) -> float | None:
    if value is None or isinstance(value, str):
        return None
    return float(value)
