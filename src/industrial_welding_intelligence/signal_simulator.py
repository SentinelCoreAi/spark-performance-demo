"""Synthetic welding signal generator for Spark for Performance demos."""

import numpy as np
import pandas as pd

SCENARIOS = [
    "stable_weld",
    "arc_instability",
    "shielding_gas_disturbance",
    "heat_input_drift",
    "mixed_disturbance",
]

SCENARIO_LABELS = {
    "stable_weld": "Stable Weld",
    "arc_instability": "Arc Instability",
    "shielding_gas_disturbance": "Shielding Gas Disturbance",
    "heat_input_drift": "Heat Input Drift",
    "mixed_disturbance": "Mixed Disturbance",
}

SCENARIO_DESCRIPTIONS = {
    "stable_weld": "All parameters within nominal range — no deviations detected.",
    "arc_instability": "Irregular arc with elevated noise and current variation.",
    "shielding_gas_disturbance": "Gas flow drops significantly mid-weld.",
    "heat_input_drift": "Progressive increase in heat input over time.",
    "mixed_disturbance": "Combined arc instability, gas disturbance, and heat drift.",
}

# Nominal signal targets
NOMINAL = {
    "amperage_A": 185.0,
    "voltage_V": 22.5,
    "gas_flow_l_min": 18.0,
    "arc_noise": 0.04,
    "heat_input_kj_mm": 1.15,
}

N_POINTS_DEFAULT = 120


def generate_signal(
    scenario: str, n_points: int = N_POINTS_DEFAULT, seed: int = 42
) -> pd.DataFrame:
    """Generate a synthetic time-series welding signal for the given scenario.

    Returns a DataFrame with columns:
        time_s, amperage_A, voltage_V, gas_flow_l_min, arc_noise, heat_input_kj_mm
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario}'. Valid: {SCENARIOS}")

    rng = np.random.default_rng(seed)
    t = np.linspace(0, n_points - 1, n_points)

    amp = _base_amp = NOMINAL["amperage_A"]
    volt = _base_volt = NOMINAL["voltage_V"]
    gas = _base_gas = NOMINAL["gas_flow_l_min"]
    noise_base = NOMINAL["arc_noise"]
    hi_base = NOMINAL["heat_input_kj_mm"]

    if scenario == "stable_weld":
        amperage = amp + rng.normal(0, 1.8, n_points)
        voltage = volt + rng.normal(0, 0.25, n_points)
        gas_flow = gas + rng.normal(0, 0.4, n_points)
        arc_noise = noise_base + np.abs(rng.normal(0, 0.008, n_points))
        heat_input = hi_base + rng.normal(0, 0.015, n_points)

    elif scenario == "arc_instability":
        amperage = amp + rng.normal(0, 10, n_points)
        spike_idx = rng.choice(n_points, size=8, replace=False)
        amperage[spike_idx] += rng.choice([-40, 40], size=8)
        voltage = volt + rng.normal(0, 2.0, n_points)
        gas_flow = gas + rng.normal(0, 0.5, n_points)
        arc_noise = 0.22 + rng.exponential(0.14, n_points)
        heat_input = hi_base + rng.normal(0, 0.09, n_points)

    elif scenario == "shielding_gas_disturbance":
        amperage = amp + rng.normal(0, 2.0, n_points)
        voltage = volt + rng.normal(0, 0.4, n_points)
        gas_flow = gas + rng.normal(0, 0.5, n_points)
        drop_start = n_points // 3
        gas_flow[drop_start:] -= 10.0
        arc_noise = noise_base + np.abs(rng.normal(0, 0.012, n_points))
        arc_noise[drop_start:] += 0.12
        heat_input = hi_base + rng.normal(0, 0.02, n_points)

    elif scenario == "heat_input_drift":
        drift_amp = np.linspace(0, 30, n_points)
        drift_volt = np.linspace(0, 2.5, n_points)
        amperage = amp + drift_amp + rng.normal(0, 2.0, n_points)
        voltage = volt + drift_volt + rng.normal(0, 0.3, n_points)
        gas_flow = gas + rng.normal(0, 0.45, n_points)
        arc_noise = noise_base + np.abs(rng.normal(0, 0.012, n_points))
        drift_hi = np.linspace(0, 0.48, n_points)
        heat_input = hi_base + drift_hi + rng.normal(0, 0.025, n_points)

    elif scenario == "mixed_disturbance":
        third = n_points // 3
        amperage = amp + rng.normal(0, 3, n_points)
        amperage[:third] += rng.normal(0, 12, third)
        spike_idx = rng.choice(third, size=4, replace=False)
        amperage[spike_idx] += rng.choice([-35, 35], size=4)
        voltage = volt + rng.normal(0, 0.8, n_points)
        gas_flow = gas + rng.normal(0, 0.5, n_points)
        gas_flow[third:] -= 9.0
        arc_noise = noise_base + np.abs(rng.normal(0, 0.02, n_points))
        arc_noise[: 2 * third] += 0.20
        drift_hi = np.zeros(n_points)
        drift_hi[2 * third :] = np.linspace(0, 0.38, n_points - 2 * third)
        heat_input = hi_base + drift_hi + rng.normal(0, 0.025, n_points)

    else:
        raise ValueError(f"Unhandled scenario: {scenario}")

    amperage = np.clip(amperage, 80, 320)
    voltage = np.clip(voltage, 14, 38)
    gas_flow = np.clip(gas_flow, 0.0, 35.0)
    arc_noise = np.clip(arc_noise, 0.0, 1.0)
    heat_input = np.clip(heat_input, 0.4, 2.8)

    return pd.DataFrame(
        {
            "time_s": t,
            "amperage_A": amperage,
            "voltage_V": voltage,
            "gas_flow_l_min": gas_flow,
            "arc_noise": arc_noise,
            "heat_input_kj_mm": heat_input,
        }
    )
