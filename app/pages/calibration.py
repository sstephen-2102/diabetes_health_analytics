"""Streamlit page: calibration."""
import streamlit as st

from app import artifacts
from app.components import charts
from app.components.results import load_or_stop, show_context, show_warnings
from app.components.tables import model_label


def render():
    """Render page controls and structured results."""
    st.title("Calibration")
    outputs = load_or_stop(artifacts.modeling_outputs)
    results, meta = outputs["results"], outputs["metadata"]
    show_context(outputs, f"Test set: {meta['test_rows']:,} rows")

    st.plotly_chart(charts.calibration_chart(results["calibration"]))

    st.subheader("Calibration summary")
    summary = results["calibration_summary"].rename(index=model_label)
    st.dataframe(summary[["brier_score", "expected_calibration_error", "n_bins", "n"]], column_config={
        "brier_score": st.column_config.NumberColumn("Brier score", format="%.4f"),
        "expected_calibration_error": st.column_config.NumberColumn("Expected calibration error", format="%.4f"),
        "n_bins": st.column_config.NumberColumn("Bins", format="%d"),
        "n": st.column_config.NumberColumn("Test rows", format="%d")})

    name = st.selectbox("Bin table for model", meta["models"], format_func=model_label)
    st.dataframe(results["calibration"][name], hide_index=True)

    show_warnings([
        "Points on the dashed line mean predicted probabilities match observed frequencies in that bin.",
        f"Bins are equal-width on predicted probability; bins with fewer than {charts.MIN_CALIBRATION_COUNT} "
        "test rows are hidden from the chart (they are still in the bin table).",
        "Brier score is the mean squared error of the probabilities (lower is better); expected calibration "
        "error is the count-weighted gap between predicted and observed in each bin.",
        "Models trained with class weighting or undersampling are not shown: rebalancing shifts probabilities "
        "upward, so their raw outputs are not calibrated to the real prevalence.",
    ])