"""Spark for Performance — Streamlit MVP.

CONCEPT DEMO · SYNTHETIC DATA ONLY · NOT FOR OPERATIONAL USE
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from industrial_welding_intelligence.signal_simulator import (
    SCENARIOS,
    SCENARIO_DESCRIPTIONS,
    SCENARIO_LABELS,
    generate_signal,
)
from industrial_welding_intelligence.ghost_instructor import (
    THRESHOLDS,
    analyze_signal,
    compute_arc_time_ratio,
    get_signal_summary,
)
from industrial_welding_intelligence.profitability_core import compute_profitability
from industrial_welding_intelligence.predictive_service import (
    ALLOWED_STATUSES,
    SERVICE_INTERVAL_HOURS,
    compute_service_recommendation,
)
from industrial_welding_intelligence.safety import (
    get_all_disclaimers,
    get_pilot_kpis,
    get_pilot_scope,
)

# ── page config ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Spark for Performance",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── custom styles ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
  .spark-header { text-align: center; padding: 1.2rem 0 0.4rem; }
  .spark-header h1 { font-size: 2.6rem; letter-spacing: -1px; margin: 0; }
  .spark-header p  { color: #aaa; margin: 0.3rem 0 0.8rem; font-size: 1.05rem; }
  .badge-row { display: flex; justify-content: center; gap: 1.2rem; flex-wrap: wrap;
               margin-bottom: 0.6rem; }
  .badge { padding: 0.3rem 1.1rem; border-radius: 20px; font-size: 0.88rem;
           font-weight: 600; letter-spacing: 0.3px; }
  .badge-guide       { background: #1a3a5c; color: #7eb8f7; }
  .badge-performance { background: #1a3d25; color: #6fcf97; }
  .badge-service     { background: #3d2a0a; color: #f2994a; }
  .alert-box { border-radius: 8px; padding: 1rem 1.2rem; margin-bottom: 0.8rem;
               border-left: 4px solid; }
  .alert-ok       { background: #0d2e1a; border-color: #27ae60; }
  .alert-warning  { background: #2e2000; border-color: #f39c12; }
  .alert-critical { background: #2e0a0a; border-color: #e74c3c; }
  .disclaimer-box { background: #1c1c1c; border: 1px solid #555; border-radius: 6px;
                    padding: 0.8rem 1rem; font-size: 0.83rem; color: #aaa;
                    margin-bottom: 0.6rem; line-height: 1.55; }
  .scope-row { display: flex; justify-content: space-between; padding: 0.45rem 0;
               border-bottom: 1px solid #333; }
  .scope-label { color: #999; font-size: 0.88rem; }
  .scope-value { font-weight: 600; font-size: 0.88rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ── header ────────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
<div class="spark-header">
  <h1>⚡ Spark for Performance</h1>
  <p>Proactive software layer concept for modern welding systems</p>
  <div class="badge-row">
    <span class="badge badge-guide">🔵 Spark Guide</span>
    <span class="badge badge-performance">🟢 Spark Performance</span>
    <span class="badge badge-service">🟠 Spark Service</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
st.caption("⚠️  CONCEPT DEMO · SYNTHETIC DATA ONLY · NOT FOR OPERATIONAL USE")
st.divider()

# ── shared constants ───────────────────────────────────────────────────────────────────────

ALERT_STYLE: dict = {
    "OK":       ("alert-ok",       "🟢", "#27ae60"),
    "WARNING":  ("alert-warning",  "🟡", "#f39c12"),
    "CRITICAL": ("alert-critical", "🔴", "#e74c3c"),
}

DEVIATION_LABELS: dict = {
    "none":              "None",
    "arc_instability":   "Arc Instability",
    "gas_disturbance":   "Gas Disturbance",
    "heat_input_drift":  "Heat Input Drift",
    "multiple":          "Multiple",
}

SIGNAL_COLORS: dict = {
    "amperage_A":        "#ff6b35",
    "voltage_V":         "#4ecdc4",
    "gas_flow_l_min":    "#45b7d1",
    "arc_noise":         "#ffd93d",
    "heat_input_kj_mm":  "#c77dff",
}

SIGNAL_LABELS: dict = {
    "amperage_A":        "Amperage (A)",
    "voltage_V":         "Voltage (V)",
    "gas_flow_l_min":    "Gas Flow (L/min)",
    "arc_noise":         "Arc Noise",
    "heat_input_kj_mm":  "Heat Input (kJ/mm)",
}

SIGNAL_NOMINALS: dict = {
    "amperage_A":        (160, 210),
    "voltage_V":         (20, 25),
    "gas_flow_l_min":    (14, 25),
    "arc_noise":         (0, 0.15),
    "heat_input_kj_mm":  (1.0, 1.3),
}

LEVEL_COLORS: dict = {
    "OK":       "#27ae60",
    "WARNING":  "#f39c12",
    "CRITICAL": "#e74c3c",
}


# ── cached helpers ────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _get_signal(scenario: str, seed: int) -> pd.DataFrame:
    return generate_signal(scenario, seed=seed)


def _build_signal_chart(df: pd.DataFrame) -> go.Figure:
    signals = list(SIGNAL_LABELS.keys())
    fig = make_subplots(
        rows=len(signals),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.032,
        subplot_titles=[SIGNAL_LABELS[s] for s in signals],
    )
    for i, sig in enumerate(signals, start=1):
        lo, hi = SIGNAL_NOMINALS[sig]
        fig.add_shape(
            type="rect",
            x0=float(df["time_s"].min()), x1=float(df["time_s"].max()),
            y0=lo, y1=hi,
            fillcolor="rgba(255,255,255,0.05)",
            line_width=0,
            row=i, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["time_s"],
                y=df[sig],
                mode="lines",
                name=SIGNAL_LABELS[sig],
                line=dict(color=SIGNAL_COLORS[sig], width=1.8),
                showlegend=False,
            ),
            row=i, col=1,
        )
    fig.update_layout(
        height=490,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=28, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False, title_text="Time (s)", row=len(signals), col=1)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")
    return fig


@st.cache_data(show_spinner=False)
def _simulated_shift(n_events: int = 24, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic shift of weld events for the performance dashboard."""
    rng = np.random.default_rng(seed)
    weights = [0.45, 0.18, 0.18, 0.12, 0.07]
    chosen = rng.choice(SCENARIOS, size=n_events, p=weights).tolist()
    rows = []
    for i, scenario in enumerate(chosen):
        df = _get_signal(scenario, seed=int(seed + i * 7))
        alert = analyze_signal(df)
        ratio = compute_arc_time_ratio(df)
        rows.append(
            {
                "event": i + 1,
                "time_min": round(i * (480.0 / n_events), 1),
                "scenario": SCENARIO_LABELS[scenario],
                "level": alert.level,
                "deviation_type": DEVIATION_LABELS[alert.deviation_type],
                "arc_time_ratio": ratio,
            }
        )
    return pd.DataFrame(rows)


# ── tabs ──────────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "⚡ Spark Guide",
        "📊 Spark Performance",
        "💰 Profitability Core",
        "🔧 Spark Service",
        "🛡️ Pilot & Safety",
    ]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SPARK GUIDE / GHOST-INSTRUCTOR LIVE
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("🔵 Spark Guide — Ghost-Instructor Live")
    st.markdown(
        "Real-time operator guidance layer. Select a scenario to simulate synthetic "
        "welding signals and see how the Ghost-Instructor responds."
    )

    ctrl_col, seed_col = st.columns([3, 1])
    with ctrl_col:
        scenario_key = st.selectbox(
            "Welding scenario",
            options=SCENARIOS,
            format_func=lambda s: SCENARIO_LABELS[s],
            key="guide_scenario",
        )
    with seed_col:
        seed_val = int(
            st.number_input("Random seed", min_value=1, max_value=9999, value=42, step=1)
        )

    st.caption(f"_{SCENARIO_DESCRIPTIONS[scenario_key]}_")

    with st.spinner("Generating synthetic signal…"):
        df_signal = _get_signal(scenario_key, seed_val)
        alert = analyze_signal(df_signal)
        summary = get_signal_summary(df_signal)
        arc_ratio = compute_arc_time_ratio(df_signal)

    chart_col, panel_col = st.columns([3, 1], gap="medium")

    with chart_col:
        st.markdown("**Synthetic signal — 120 s window**")
        st.plotly_chart(_build_signal_chart(df_signal), use_container_width=True)
        st.caption(
            "Shaded bands show nominal operating range. "
            "Synthetic data only — not from a real machine."
        )

    with panel_col:
        css_class, icon, color = ALERT_STYLE[alert.level]
        st.markdown(
            f"""
<div class="alert-box {css_class}">
  <div style="font-size:1.25rem;font-weight:700;color:{color}">{icon} {alert.level}</div>
  <div style="font-size:0.88rem;margin-top:0.25rem;color:#ccc">
    {DEVIATION_LABELS[alert.deviation_type]}
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown("**Operator guidance**")
        if alert.level == "OK":
            st.success(alert.recommendation)
        elif alert.level == "WARNING":
            st.warning(alert.recommendation)
        else:
            st.error(alert.recommendation)

        st.divider()
        st.markdown("**Signal averages**")
        for label, sig_key in [
            ("Amperage", "amperage_A"),
            ("Voltage", "voltage_V"),
            ("Gas Flow", "gas_flow_l_min"),
            ("Arc Noise", "arc_noise"),
            ("Heat Input", "heat_input_kj_mm"),
        ]:
            units = {"amperage_A": " A", "voltage_V": " V", "gas_flow_l_min": " L/min",
                     "arc_noise": "", "heat_input_kj_mm": " kJ/mm"}
            st.metric(label, f"{summary[sig_key]}{units[sig_key]}")

        st.divider()
        st.metric("Arc time ratio", f"{arc_ratio:.1%}")
        st.caption(
            "Fraction of time where arc, gas, and heat-input "
            "are all within the nominal operating band."
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SPARK PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Spark Performance — Supervisor Dashboard")
    st.markdown(
        "Alert trends, deviation patterns, and arc-efficiency metrics "
        "across a simulated 8-hour shift."
    )

    shift_seed = st.slider(
        "Shift simulation seed", 1, 200, 42, key="shift_seed",
        help="Change to explore different synthetic shift scenarios."
    )

    with st.spinner("Simulating shift data…"):
        shift_df = _simulated_shift(seed=shift_seed)

    warning_count = int((shift_df["level"] == "WARNING").sum())
    critical_count = int((shift_df["level"] == "CRITICAL").sum())
    avg_arc_ratio = float(shift_df["arc_time_ratio"].mean())

    prof_shift = compute_profitability(
        arc_time_ratio=avg_arc_ratio,
        warning_count=warning_count,
        critical_count=critical_count,
        session_minutes=480.0,
    )

    # KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Welds in shift", len(shift_df))
    k2.metric("Warning alerts", warning_count)
    k3.metric("Critical alerts", critical_count)
    k4.metric("Avg arc time ratio", f"{avg_arc_ratio:.1%}")
    k5.metric("Est. rework risk", f"{prof_shift.rework_risk_fraction:.1%}")

    st.divider()

    chart_l, chart_r = st.columns(2, gap="large")

    with chart_l:
        st.markdown("**Deviation type distribution**")
        dev_counts = (
            shift_df[shift_df["level"] != "OK"]["deviation_type"]
            .value_counts()
            .reset_index()
        )
        dev_counts.columns = ["Deviation Type", "Count"]
        if dev_counts.empty:
            st.success("No deviations recorded in this simulated shift.")
        else:
            fig_dev = px.bar(
                dev_counts,
                x="Deviation Type",
                y="Count",
                color="Deviation Type",
                color_discrete_sequence=["#ff6b35", "#4ecdc4", "#ffd93d", "#c77dff"],
                template="plotly_dark",
            )
            fig_dev.update_layout(
                showlegend=False,
                margin=dict(l=0, r=0, t=10, b=0),
                height=280,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_dev, use_container_width=True)

    with chart_r:
        st.markdown("**Alert timeline — simulated shift**")
        fig_timeline = go.Figure()
        for level, color in LEVEL_COLORS.items():
            subset = shift_df[shift_df["level"] == level]
            if subset.empty:
                continue
            fig_timeline.add_trace(
                go.Scatter(
                    x=subset["time_min"],
                    y=[level] * len(subset),
                    mode="markers",
                    marker=dict(color=color, size=13, symbol="circle"),
                    name=level,
                    text=subset["scenario"],
                    hovertemplate="%{text}<br>t = %{x} min<extra></extra>",
                )
            )
        fig_timeline.update_layout(
            template="plotly_dark",
            height=280,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Time in shift (min)",
            yaxis=dict(
                categoryorder="array",
                categoryarray=["OK", "WARNING", "CRITICAL"],
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            ),
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

    st.divider()
    st.markdown("**Arc time ratio — per weld event**")

    fig_ratio = go.Figure(
        go.Bar(
            x=shift_df["event"],
            y=shift_df["arc_time_ratio"],
            marker_color=[LEVEL_COLORS[lv] for lv in shift_df["level"]],
            hovertext=shift_df["scenario"],
            hovertemplate="Event %{x}<br>%{hovertext}<br>Arc ratio: %{y:.1%}<extra></extra>",
        )
    )
    fig_ratio.add_hline(
        y=avg_arc_ratio,
        line_dash="dash",
        line_color="#888",
        annotation_text=f"Mean {avg_arc_ratio:.1%}",
        annotation_position="top right",
        annotation_font_color="#aaa",
    )
    fig_ratio.update_layout(
        template="plotly_dark",
        height=220,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Weld event #",
        yaxis_title="Arc time ratio",
        yaxis_range=[0, 1.05],
    )
    st.plotly_chart(fig_ratio, use_container_width=True)

    with st.expander("Full event log"):
        st.dataframe(shift_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PROFITABILITY CORE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("💰 Profitability Core — Pilot Estimates")

    st.markdown(
        """
<div class="disclaimer-box">
⚠️ <strong>PILOT ESTIMATES ONLY</strong> — All figures on this page are synthetic
approximations generated for concept-demonstration purposes. They do not constitute
financial, quality, or operational guarantees. Actual values require validated
production data and site-specific parameters.
</div>
""",
        unsafe_allow_html=True,
    )

    params_col, results_col = st.columns([1, 2], gap="large")

    with params_col:
        st.markdown("**Session parameters**")
        p_arc_ratio = st.slider("Arc time ratio", 0.0, 1.0, 0.72, 0.01, key="p_arc")
        p_warnings = st.number_input("Warning count", 0, 200, 8, key="p_warn")
        p_criticals = st.number_input("Critical alert count", 0, 100, 3, key="p_crit")
        p_session = st.number_input("Session length (min)", 30, 960, 480, key="p_sess")
        p_gas_flow = st.number_input(
            "Gas flow target (L/min)", 5.0, 30.0, 18.0, key="p_gas"
        )

        st.divider()
        st.markdown("**Cost parameters**")
        p_rework_rate = st.number_input(
            "Rework rate (€/min)", 0.5, 10.0, 2.50, 0.10, key="p_rew"
        )
        p_idle_rate = st.number_input(
            "Idle rate (€/min)", 0.1, 5.0, 0.80, 0.05, key="p_idle"
        )
        p_gas_cost = st.number_input(
            "Gas cost (€/L)", 0.001, 0.05, 0.008, 0.001,
            format="%.3f", key="p_gcost"
        )

    with results_col:
        est = compute_profitability(
            arc_time_ratio=p_arc_ratio,
            warning_count=int(p_warnings),
            critical_count=int(p_criticals),
            session_minutes=float(p_session),
            gas_flow_l_min=float(p_gas_flow),
            gas_cost_eur_per_l=float(p_gas_cost),
            rework_cost_eur_per_min=float(p_rework_rate),
            idle_cost_eur_per_min=float(p_idle_rate),
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Wasted minutes", f"{est.wasted_minutes:.1f} min")
        m2.metric("Gas waste", f"{est.gas_waste_l:.1f} L")
        m3.metric("Est. cost impact", f"€ {est.estimated_cost_eur:.2f}")

        rr_col, _ = st.columns([1, 2])
        rr_col.metric("Est. rework risk", f"{est.rework_risk_fraction:.1%}")

        st.divider()
        st.markdown("**Cost breakdown**")
        breakdown = est.cost_breakdown
        labels = ["Rework (est.)", "Idle time (est.)", "Gas waste (est.)"]
        values = [
            breakdown["rework_cost_eur"],
            breakdown["idle_cost_eur"],
            breakdown["gas_cost_eur"],
        ]

        if sum(values) > 0:
            fig_pie = go.Figure(
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.44,
                    marker_colors=["#e74c3c", "#f39c12", "#45b7d1"],
                    textinfo="label+percent",
                    hovertemplate="%{label}: € %{value:.2f}<extra></extra>",
                )
            )
            fig_pie.update_layout(
                template="plotly_dark",
                height=300,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                annotations=[
                    dict(
                        text=f"€ {est.estimated_cost_eur:.2f}",
                        x=0.5, y=0.5,
                        font_size=18, showarrow=False, font_color="#eee",
                    )
                ],
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No estimated cost impact for these parameters.")

        st.caption("All values are synthetic pilot estimates — see disclaimer above.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SPARK SERVICE
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🔧 Spark Service — Predictive Service Layer")
    st.markdown(
        "Accumulated arc-hour tracking and anomaly-driven service recommendations. "
        "**No automatic ordering is performed by this system.**"
    )

    svc_left, svc_right = st.columns([1, 2], gap="large")

    with svc_left:
        st.markdown("**Machine & session inputs**")
        s_hours = st.slider(
            f"Accumulated arc hours  _(service interval: {SERVICE_INTERVAL_HOURS} h)_",
            0.0, float(SERVICE_INTERVAL_HOURS * 1.2),
            180.0, 5.0,
            key="svc_hours",
        )
        s_arc = st.number_input(
            "Arc instability alerts", 0, 100, 4, key="svc_arc"
        )
        s_gas = st.number_input(
            "Gas flow anomaly count", 0, 100, 2, key="svc_gas"
        )
        s_wire = st.number_input(
            "Wire feed irregularity _(placeholder)_", 0, 50, 0, key="svc_wire"
        )

    with svc_right:
        rec = compute_service_recommendation(
            accumulated_arc_hours=float(s_hours),
            arc_instability_count=int(s_arc),
            gas_anomaly_count=int(s_gas),
            wire_feed_irregularity_count=int(s_wire),
        )

        status_styles: dict = {
            "OK":                  ("#0d2e1a", "#27ae60", "🟢"),
            "Monitor":             ("#0d1e3a", "#3498db", "🔵"),
            "Service Recommended": ("#2e2000", "#f39c12", "🟡"),
            "Service Required":    ("#2e0a0a", "#e74c3c", "🔴"),
        }
        bg, col, icon = status_styles.get(rec.service_status, ("#1a1a1a", "#aaa", "⚪"))
        st.markdown(
            f"""
<div style="background:{bg};border-left:4px solid {col};border-radius:8px;
            padding:1rem 1.2rem;margin-bottom:1rem">
  <div style="font-size:1.3rem;font-weight:700;color:{col}">{icon} {rec.service_status}</div>
  <div style="color:#ccc;margin-top:0.25rem">
    Urgency: <strong style="color:{col}">{rec.urgency}</strong>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown("**Service risk score**")
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=rec.risk_score,
                domain={"x": [0, 1], "y": [0, 1]},
                number={"suffix": " / 100", "font": {"size": 22}},
                gauge={
                    "axis": {"range": [0, 100], "tickfont": {"size": 10}},
                    "bar": {"color": col},
                    "steps": [
                        {"range": [0, 25],   "color": "#1a3d25"},
                        {"range": [25, 50],  "color": "#1a2a3a"},
                        {"range": [50, 75],  "color": "#3d2a0a"},
                        {"range": [75, 100], "color": "#3d0a0a"},
                    ],
                    "threshold": {
                        "line": {"color": "#fff", "width": 2},
                        "thickness": 0.75,
                        "value": 75,
                    },
                },
            )
        )
        fig_gauge.update_layout(
            template="plotly_dark",
            height=200,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#eee",
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.divider()

        st.markdown("**Recommended check**")
        st.info(rec.recommended_check)

        st.markdown("**Repeated anomaly summary**")
        summ = rec.anomaly_summary
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Arc instability", summ["arc_instability_alerts"])
        a2.metric("Gas anomalies", summ["gas_anomaly_count"])
        a3.metric("Wire irregularity", summ["wire_feed_irregularity_count"])
        a4.metric("Arc hours", f"{summ['accumulated_arc_hours']:.0f} h")

        st.divider()
        st.markdown("**Service partner — dealer note**")
        st.markdown(
            f'<div class="disclaimer-box">🔧 {rec.dealer_note}</div>',
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — PILOT & SAFETY BOUNDARIES
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("🛡️ Pilot & Safety Boundaries")
    st.markdown(
        "All mandatory disclaimers, pilot-scope constraints, and KPIs for this "
        "concept demonstration."
    )

    disclaimers = get_all_disclaimers()
    scope = get_pilot_scope()
    kpis = get_pilot_kpis()

    # Disclaimers
    st.markdown("### Mandatory Disclaimers")
    disc_meta = [
        ("synthetic_data",  "🧪", "Synthetic Data"),
        ("concept_demo",    "🖥️", "Concept Demo"),
        ("quality",         "🔍", "Weld Quality"),
        ("wps",             "📄", "WPS Generation"),
        ("inspection",      "👷", "Inspection & Supervision"),
        ("pilot_estimates", "📊", "Pilot Estimates"),
    ]
    for key, icon, label in disc_meta:
        with st.expander(f"{icon}  {label}"):
            st.markdown(
                f'<div class="disclaimer-box">{disclaimers[key]}</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    col_scope, col_kpi = st.columns(2, gap="large")

    with col_scope:
        st.markdown("### Pilot Scope")
        scope_rows = [
            ("Machine scope",      scope["machine_scope"]),
            ("Process scope",      scope["process_scope"]),
            ("Data requirement",   scope["data_requirement"]),
            ("Deviation types",    scope["deviation_types"]),
            ("Validation required","Yes" if scope["validation_required"] else "No"),
        ]
        for label, value in scope_rows:
            st.markdown(
                f'<div class="scope-row">'
                f'<span class="scope-label">{label}</span>'
                f'<span class="scope-value">{value}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )

    with col_kpi:
        st.markdown("### Pilot KPIs")
        for kpi in kpis:
            st.markdown(f"- {kpi}")

        st.divider()
        st.info(
            "**Safe demo scope:** one machine model · one welding process · "
            "approved test data required after demo · 2–3 measurable deviation types per pilot"
        )

    st.divider()

    st.markdown("### Future Roadmap")
    roadmap = [
        ("Validation with approved test data",
         "Replace synthetic signals with real machine data from approved test runs."),
        ("Device connectivity",
         "Direct integration with welding power source data streams."),
        ("Customer dashboard",
         "Multi-shift, multi-operator web dashboard with role-based access."),
        ("Fleet-level analytics",
         "Cross-machine aggregation for production-floor visibility."),
        ("Service partner integration",
         "Direct service booking through authorised dealer networks."),
        ("Multilingual operator guidance",
         "Operator-facing alerts delivered in the operator's preferred language."),
    ]
    for title, description in roadmap:
        with st.expander(f"🔮  {title}"):
            st.markdown(description)
