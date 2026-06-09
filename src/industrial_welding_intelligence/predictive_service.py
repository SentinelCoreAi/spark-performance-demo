"""Predictive Service — synthetic service-risk scoring for Spark Service.

No automatic ordering is performed. All recommendations are informational.
Dealer and service-partner integration is a placeholder only.
"""

from dataclasses import dataclass, field
from typing import Literal

ServiceStatus = Literal["OK", "Monitor", "Service Recommended", "Service Required"]
Urgency = Literal["None", "Low", "Medium", "High"]

ALLOWED_STATUSES: tuple = ("OK", "Monitor", "Service Recommended", "Service Required")
ALLOWED_URGENCIES: tuple = ("None", "Low", "Medium", "High")

SERVICE_INTERVAL_HOURS: int = 500


@dataclass
class ServiceRecommendation:
    service_status: ServiceStatus
    recommended_check: str
    urgency: Urgency
    risk_score: float
    dealer_note: str
    anomaly_summary: dict = field(default_factory=dict)


def compute_service_recommendation(
    accumulated_arc_hours: float,
    arc_instability_count: int,
    gas_anomaly_count: int,
    wire_feed_irregularity_count: int = 0,
) -> ServiceRecommendation:
    """Compute a synthetic service recommendation.

    No automatic ordering is triggered. The dealer_note is a placeholder
    for future service-partner integration.
    """
    accumulated_arc_hours = max(0.0, float(accumulated_arc_hours))
    arc_instability_count = max(0, int(arc_instability_count))
    gas_anomaly_count = max(0, int(gas_anomaly_count))
    wire_feed_irregularity_count = max(0, int(wire_feed_irregularity_count))

    risk_score = _compute_risk_score(
        accumulated_arc_hours,
        arc_instability_count,
        gas_anomaly_count,
        wire_feed_irregularity_count,
    )

    status, urgency, check = _classify_service(risk_score, accumulated_arc_hours)

    return ServiceRecommendation(
        service_status=status,
        recommended_check=check,
        urgency=urgency,
        risk_score=round(risk_score, 1),
        dealer_note=(
            "Contact your authorised service partner to schedule a preventive check. "
            "No automatic ordering is performed by this system. "
            "[Dealer / service-partner integration — placeholder]"
        ),
        anomaly_summary={
            "arc_instability_alerts": arc_instability_count,
            "gas_anomaly_count": gas_anomaly_count,
            "wire_feed_irregularity_count": wire_feed_irregularity_count,
            "accumulated_arc_hours": round(accumulated_arc_hours, 1),
            "service_interval_hours": SERVICE_INTERVAL_HOURS,
        },
    )


# ── private helpers ───────────────────────────────────────────────────────────────────────────


def _compute_risk_score(
    arc_hours: float,
    arc_instability: int,
    gas_anomalies: int,
    wire_irregularity: int,
) -> float:
    hours_fraction = min(1.0, arc_hours / SERVICE_INTERVAL_HOURS)
    score = hours_fraction * 40.0
    score += min(30.0, arc_instability * 2.5)
    score += min(20.0, gas_anomalies * 2.0)
    score += min(10.0, wire_irregularity * 1.5)
    return min(100.0, score)


def _classify_service(
    risk_score: float, arc_hours: float
) -> tuple[ServiceStatus, Urgency, str]:
    if risk_score >= 75 or arc_hours >= SERVICE_INTERVAL_HOURS:
        return (
            "Service Required",
            "High",
            (
                "Full preventive service: contact tip replacement, liner inspection, "
                "gas nozzle cleaning, drive roll wear check."
            ),
        )
    if risk_score >= 50:
        return (
            "Service Recommended",
            "Medium",
            (
                "Schedule contact tip inspection, verify gas hose connections, "
                "and check drive roll tension."
            ),
        )
    if risk_score >= 25:
        return (
            "Monitor",
            "Low",
            "Monitor arc stability and gas flow trends. Log any recurring anomalies.",
        )
    return (
        "OK",
        "None",
        "No immediate service action required. Continue standard monitoring.",
    )
