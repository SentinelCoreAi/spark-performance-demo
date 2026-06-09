"""Ghost-Instructor Lite — real-time operator guidance with non-blaming language.

No cost or euro metrics are present here; those belong in profitability_core.
"""

from dataclasses import dataclass
from typing import Literal

import pandas as pd

AlertLevel = Literal["OK", "WARNING", "CRITICAL"]
DeviationType = Literal["none", "arc_instability", "gas_disturbance", "heat_input_drift", "multiple"]

THRESHOLDS: dict = {
    "arc_noise_warning": 0.15,
    "arc_noise_critical": 0.35,
    "amperage_cv_warning": 0.06,
    "amperage_cv_critical": 0.10,
    "gas_flow_low_warning": 14.0,
    "gas_flow_low_critical": 10.0,
    "gas_flow_high_warning": 25.0,
    "gas_flow_high_critical": 28.0,
    "heat_input_nominal_kj_mm": 1.15,
    "heat_input_drift_warning": 0.15,
    "heat_input_drift_critical": 0.30,
}

_LEVEL_RANK: dict = {"OK": 0, "WARNING": 1, "CRITICAL": 2}


@dataclass
class Alert:
    level: AlertLevel
    deviation_type: DeviationType
    recommendation: str


def analyze_signal(df: pd.DataFrame) -> Alert:
    """Analyse a signal DataFrame and return the most relevant operator alert."""
    arc_level, arc_active = _check_arc_instability(df)
    gas_level, gas_active = _check_gas_disturbance(df)
    heat_level, heat_active = _check_heat_input_drift(df)

    active_count = sum([arc_active, gas_active, heat_active])

    if active_count == 0:
        return Alert(
            level="OK",
            deviation_type="none",
            recommendation="Process looks steady — good work keeping everything on track.",
        )

    if active_count >= 2:
        max_level: AlertLevel = max(
            [arc_level, gas_level, heat_level],
            key=lambda lv: _LEVEL_RANK[lv],
        )
        return Alert(
            level=max_level,
            deviation_type="multiple",
            recommendation=_multi_recommendation(max_level),
        )

    if arc_active:
        return Alert(
            level=arc_level,
            deviation_type="arc_instability",
            recommendation=_arc_recommendation(arc_level),
        )
    if gas_active:
        return Alert(
            level=gas_level,
            deviation_type="gas_disturbance",
            recommendation=_gas_recommendation(gas_level),
        )
    return Alert(
        level=heat_level,
        deviation_type="heat_input_drift",
        recommendation=_heat_recommendation(heat_level),
    )


def get_signal_summary(df: pd.DataFrame) -> dict:
    """Return mean signal values for display panels."""
    return {
        "amperage_A": round(float(df["amperage_A"].mean()), 1),
        "voltage_V": round(float(df["voltage_V"].mean()), 2),
        "gas_flow_l_min": round(float(df["gas_flow_l_min"].mean()), 1),
        "arc_noise": round(float(df["arc_noise"].mean()), 3),
        "heat_input_kj_mm": round(float(df["heat_input_kj_mm"].mean()), 3),
    }


def compute_arc_time_ratio(df: pd.DataFrame) -> float:
    """Fraction of time-points where key arc parameters are within nominal range."""
    good = (
        (df["arc_noise"] < THRESHOLDS["arc_noise_warning"])
        & (df["gas_flow_l_min"] >= THRESHOLDS["gas_flow_low_warning"])
        & (df["gas_flow_l_min"] <= THRESHOLDS["gas_flow_high_warning"])
    )
    return round(float(good.mean()), 4)


# ── private helpers ───────────────────────────────────────────────────────────────────────────


def _check_arc_instability(df: pd.DataFrame) -> tuple[AlertLevel, bool]:
    noise_mean = float(df["arc_noise"].mean())
    amp_cv = float(df["amperage_A"].std() / df["amperage_A"].mean())

    if noise_mean >= THRESHOLDS["arc_noise_critical"] or amp_cv >= THRESHOLDS["amperage_cv_critical"]:
        return "CRITICAL", True
    if noise_mean >= THRESHOLDS["arc_noise_warning"] or amp_cv >= THRESHOLDS["amperage_cv_warning"]:
        return "WARNING", True
    return "OK", False


def _check_gas_disturbance(df: pd.DataFrame) -> tuple[AlertLevel, bool]:
    gas_mean = float(df["gas_flow_l_min"].mean())

    if gas_mean <= THRESHOLDS["gas_flow_low_critical"] or gas_mean >= THRESHOLDS["gas_flow_high_critical"]:
        return "CRITICAL", True
    if gas_mean <= THRESHOLDS["gas_flow_low_warning"] or gas_mean >= THRESHOLDS["gas_flow_high_warning"]:
        return "WARNING", True
    return "OK", False


def _check_heat_input_drift(df: pd.DataFrame) -> tuple[AlertLevel, bool]:
    nominal = THRESHOLDS["heat_input_nominal_kj_mm"]
    deviation = abs(float(df["heat_input_kj_mm"].mean()) - nominal)

    if deviation >= THRESHOLDS["heat_input_drift_critical"]:
        return "CRITICAL", True
    if deviation >= THRESHOLDS["heat_input_drift_warning"]:
        return "WARNING", True
    return "OK", False


def _arc_recommendation(level: AlertLevel) -> str:
    if level == "CRITICAL":
        return (
            "The arc is running quite unevenly. A brief pause to inspect "
            "the contact tip, check wire stick-out, and verify the work connection "
            "would be a good idea before continuing."
        )
    return (
        "Some arc variation is showing. Keeping an eye on wire stick-out "
        "and making sure the contact tip is clean should help settle things down."
    )


def _gas_recommendation(level: AlertLevel) -> str:
    if level == "CRITICAL":
        return (
            "Gas flow is well outside the target range. Checking the hose "
            "connections and regulator setting before continuing is recommended."
        )
    return (
        "Gas flow is slightly off target. A quick look at the regulator "
        "setting and hose connections is worthwhile."
    )


def _heat_recommendation(level: AlertLevel) -> str:
    if level == "CRITICAL":
        return (
            "Heat input has drifted noticeably from the target. Reviewing "
            "travel speed and parameter settings can help bring it back on track."
        )
    return (
        "A gentle drift in heat input is appearing. Keeping travel speed "
        "consistent should help maintain the weld within the intended range."
    )


def _multi_recommendation(level: AlertLevel) -> str:
    if level == "CRITICAL":
        return (
            "Several parameters are out of range simultaneously. "
            "A short pause to check gas flow, contact tip condition, "
            "and travel speed is recommended before continuing."
        )
    return (
        "A few parameters are varying together. Reviewing the setup — "
        "gas hose, contact tip, and travel speed — would be beneficial."
    )
