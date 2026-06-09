"""Deterministic analytics for the Industrial Welding Intelligence MVP."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from statistics import mean
from typing import Any

TARGET_HEAT_INPUT_KJ_PER_MM = 1.15
MAX_ACCEPTABLE_DEFECT_RATE = 0.035
MIN_ACCEPTABLE_FIRST_PASS_YIELD = 0.92


def _as_float(value: Any, default: float = 0.0) -> float:
    """Convert loosely typed UI or CSV values into floats."""
    if value is None or value == "":
        return default
    return float(value)


def score_weld_event(event: Mapping[str, Any]) -> float:
    """Return a 0-100 quality risk score for a single weld event.

    Higher scores indicate greater risk. The heuristic combines process drift,
    environmental factors, operator experience, and inspection outcomes so the
    MVP can run locally without requiring an ML service.
    """
    amperage = _as_float(event.get("amperage"), 180.0)
    voltage = _as_float(event.get("voltage"), 24.0)
    travel_speed = max(_as_float(event.get("travel_speed_mm_s"), 5.0), 0.1)
    gas_flow = _as_float(event.get("gas_flow_l_min"), 18.0)
    plate_thickness = max(_as_float(event.get("plate_thickness_mm"), 8.0), 1.0)
    ambient_humidity = _as_float(event.get("ambient_humidity_pct"), 45.0)
    operator_experience = _as_float(event.get("operator_experience_years"), 3.0)
    previous_rework = _as_float(event.get("previous_rework_count"), 0.0)

    heat_input = amperage * voltage / (1000 * travel_speed)
    heat_input_drift = abs(heat_input - TARGET_HEAT_INPUT_KJ_PER_MM) / TARGET_HEAT_INPUT_KJ_PER_MM
    gas_penalty = max(0.0, 16.0 - gas_flow) * 2.8 + max(0.0, gas_flow - 24.0) * 1.5
    humidity_penalty = max(0.0, ambient_humidity - 60.0) * 0.45
    thickness_penalty = max(0.0, plate_thickness - 12.0) * 1.1
    experience_credit = min(operator_experience, 8.0) * 1.8
    rework_penalty = previous_rework * 7.5

    score = 25.0 + (heat_input_drift * 55.0) + gas_penalty + humidity_penalty + thickness_penalty + rework_penalty - experience_credit
    return round(max(0.0, min(score, 100.0)), 1)


def classify_quality_risk(score: float) -> str:
    """Map a numeric risk score into dashboard severity buckets."""
    if score >= 70:
        return "Critical"
    if score >= 45:
        return "Elevated"
    return "Nominal"


def recommend_action(score: float) -> str:
    """Return the next best action for the quality team."""
    risk = classify_quality_risk(score)
    if risk == "Critical":
        return "Quarantine weld, trigger NDT inspection, and review WPS parameters."
    if risk == "Elevated":
        return "Schedule supervisor review and validate gas flow, speed, and fit-up."
    return "Release to standard inspection cadence."


def compute_dashboard_metrics(events: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Summarize welding operations for executive KPI cards."""
    rows = list(events)
    if not rows:
        return {
            "total_welds": 0.0,
            "avg_risk_score": 0.0,
            "defect_rate": 0.0,
            "first_pass_yield": 0.0,
            "critical_welds": 0.0,
            "quality_gap": 0.0,
        }

    scores = [score_weld_event(row) for row in rows]
    defect_count = sum(1 for row in rows if str(row.get("inspection_result", "pass")).lower() != "pass")
    rework_count = sum(1 for row in rows if _as_float(row.get("previous_rework_count"), 0.0) > 0)
    total = len(rows)
    defect_rate = defect_count / total
    first_pass_yield = (total - rework_count) / total
    quality_gap = max(0.0, defect_rate - MAX_ACCEPTABLE_DEFECT_RATE) + max(0.0, MIN_ACCEPTABLE_FIRST_PASS_YIELD - first_pass_yield)

    return {
        "total_welds": float(total),
        "avg_risk_score": round(mean(scores), 1),
        "defect_rate": round(defect_rate, 3),
        "first_pass_yield": round(first_pass_yield, 3),
        "critical_welds": float(sum(1 for score in scores if classify_quality_risk(score) == "Critical")),
        "quality_gap": round(quality_gap, 3),
    }
