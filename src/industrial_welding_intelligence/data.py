"""Sample data helpers for the Streamlit MVP."""

from __future__ import annotations

from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "sample_weld_events.csv"


def sample_data_path() -> Path:
    """Return the bundled sample weld-event dataset path."""
    return DATA_PATH
