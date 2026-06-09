"""Industrial Welding Intelligence — Spark for Performance."""

from .analytics import (
    classify_quality_risk,
    compute_dashboard_metrics,
    recommend_action,
    score_weld_event,
)
from .ghost_instructor import Alert, analyze_signal, compute_arc_time_ratio, get_signal_summary
from .predictive_service import ServiceRecommendation, compute_service_recommendation
from .profitability_core import ProfitabilityEstimate, compute_profitability
from .safety import get_all_disclaimers, get_pilot_kpis, get_pilot_scope
from .signal_simulator import SCENARIOS, SCENARIO_LABELS, generate_signal

__all__ = [
    # analytics (legacy)
    "classify_quality_risk",
    "compute_dashboard_metrics",
    "recommend_action",
    "score_weld_event",
    # signal simulator
    "SCENARIOS",
    "SCENARIO_LABELS",
    "generate_signal",
    # ghost instructor
    "Alert",
    "analyze_signal",
    "compute_arc_time_ratio",
    "get_signal_summary",
    # profitability
    "ProfitabilityEstimate",
    "compute_profitability",
    # predictive service
    "ServiceRecommendation",
    "compute_service_recommendation",
    # safety
    "get_all_disclaimers",
    "get_pilot_kpis",
    "get_pilot_scope",
]
