"""Profitability Core — pilot-estimate business metrics for Spark Performance.

All figures are synthetic approximations. See DISCLAIMER constant.
No certified or guaranteed financial values are produced here.
"""

from dataclasses import dataclass, field

DISCLAIMER = (
    "PILOT ESTIMATES ONLY — All figures are synthetic approximations generated "
    "from a mathematical model for concept-demonstration purposes. Actual values "
    "depend on validated production data, approved test results, and site-specific "
    "parameters. These estimates do not constitute financial, quality, or "
    "operational guarantees of any kind."
)


@dataclass
class ProfitabilityEstimate:
    wasted_minutes: float
    gas_waste_l: float
    estimated_cost_eur: float
    rework_risk_fraction: float
    cost_breakdown: dict = field(default_factory=dict)
    disclaimer: str = DISCLAIMER


def compute_profitability(
    arc_time_ratio: float,
    warning_count: int,
    critical_count: int,
    session_minutes: float = 60.0,
    gas_flow_l_min: float = 18.0,
    gas_cost_eur_per_l: float = 0.008,
    rework_cost_eur_per_min: float = 2.50,
    idle_cost_eur_per_min: float = 0.80,
) -> ProfitabilityEstimate:
    """Compute synthetic pilot-estimate profitability metrics.

    Parameters are intentionally tunable so presentations can explore
    different machine / process configurations.
    """
    arc_time_ratio = max(0.0, min(1.0, arc_time_ratio))
    warning_count = max(0, warning_count)
    critical_count = max(0, critical_count)
    session_minutes = max(0.0, session_minutes)

    idle_fraction = 1.0 - arc_time_ratio
    wasted_from_idle = session_minutes * idle_fraction * 0.35
    wasted_from_alerts = critical_count * 2.8 + warning_count * 0.9
    wasted_minutes = round(max(0.0, wasted_from_idle + wasted_from_alerts), 1)

    gas_waste_l = round(max(0.0, wasted_minutes * gas_flow_l_min * 0.28), 1)

    rework_risk = min(1.0, critical_count * 0.11 + warning_count * 0.025)
    rework_risk = round(rework_risk, 4)

    rework_cost = round(wasted_minutes * rework_cost_eur_per_min * rework_risk, 2)
    idle_cost = round(wasted_minutes * idle_cost_eur_per_min * idle_fraction, 2)
    gas_cost = round(gas_waste_l * gas_cost_eur_per_l, 2)
    total_cost = round(max(0.0, rework_cost + idle_cost + gas_cost), 2)

    return ProfitabilityEstimate(
        wasted_minutes=wasted_minutes,
        gas_waste_l=gas_waste_l,
        estimated_cost_eur=total_cost,
        rework_risk_fraction=rework_risk,
        cost_breakdown={
            "rework_cost_eur": rework_cost,
            "idle_cost_eur": idle_cost,
            "gas_cost_eur": gas_cost,
        },
    )
