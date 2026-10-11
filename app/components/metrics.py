"""Presentation components: metrics."""
from numbers import Integral

import streamlit as st


def metric_row(values: dict, decimals: int = 3) -> None:
    """One st.metric per entry; floats rounded, integers shown with thousands separators."""
    for column, (label, value) in zip(st.columns(len(values)), values.items()):
        column.metric(label, f"{value:,}" if isinstance(value, Integral) else f"{value:.{decimals}f}")