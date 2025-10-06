"""Reusable Streamlit components."""
from __future__ import annotations

from typing import Dict

import streamlit as st


def render_latency(latency_ms: float, stages: Dict[str, float]) -> None:
    st.metric(label="Total Latency (ms)", value=f"{latency_ms:.1f}")
    with st.expander("Latency breakdown"):
        for name, value in stages.items():
            st.write(f"{name}: {value} ms")
