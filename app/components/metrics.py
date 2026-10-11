"""Presentation components: metrics."""
from numbers import Integral

import streamlit as st


def metric_row(values: dict, decimals: int = 3) -> None:
    """One st.metric per entry; floats rounded, integers shown with thousands separators, strings as given."""
    for column, (label, value) in zip(st.columns(len(values)), values.items()):
        if isinstance(value, str):
            column.metric(label, value)
        else:
            column.metric(label, f"{value:,}" if isinstance(value, Integral) else f"{value:.{decimals}f}")


def format_p_value(p_value: float) -> str:
    """Large samples drive p-values below floating-point precision, so report them as '< 0.001'."""
    return "< 0.001" if p_value < 0.001 else f"{p_value:.3f}"
