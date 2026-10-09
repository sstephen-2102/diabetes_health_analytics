"""Streamlit page: data_explorer."""
import pandas as pd
import streamlit as st

from app import artifacts, services
from app.components import charts
from app.components.metrics import metric_row
from app.components.results import load_or_stop, select_variant, show_context, show_warnings, variable_label
from src.common.config import TARGET_DEFINITION, VARIABLE_METADATA

REFERENCE_CHECKS = {"rows": "Row count", "columns": "Column names", "missing_cells": "Missing cells",
                    "duplicate_rows": "Repeated rows", "target_counts": "Target counts"}
STATISTIC_LABELS = {"mean": "Mean", "median": "Median", "std": "Std. deviation", "variance": "Variance",
                    "min": "Min", "q1": "Q1", "q3": "Q3", "iqr": "IQR", "max": "Max"}


def render():
    """Render page controls and structured results."""
    st.title("Data Explorer")
    data = load_or_stop(artifacts.eda_data)
    name = select_variant(data["results"]["variants"])
    bundle = data["results"]["variants"][name]
    frame, quality = bundle["data"], bundle["quality"]
    meta = data["metadata"]
    show_context({"dataset_variant": name, "target_definition": TARGET_DEFINITION},
                 f"Source: {meta['source_file']} · SHA-256 {meta['dataset_version_sha256'][:12]}…")

    metric_row({"Rows": quality["rows"], "Columns": quality["columns"], "Missing cells": quality["missing_cells"],
                "Repeated rows": quality["duplicate_rows"], "Repeated-row rate (%)": quality["duplicate_rate_percent"]},
               decimals=2)
    if frame.attrs.get("dropped_duplicate_rows"):
        st.caption(f"{frame.attrs['dropped_duplicate_rows']:,} repeated rows were removed from the "
                   f"{frame.attrs['source_rows']:,}-row source, keeping the first of each.")
    else:
        st.caption(f"{quality['rows_in_repeated_patterns']:,} rows share their answers with at least one other row. "
                   "All are kept: identical answers to these coarse questions do not show that the same person "
                   "answered twice.")

    st.plotly_chart(charts.target_distribution_chart(bundle["tables"]["target_distribution"]))

    st.subheader("Variables")
    st.dataframe(_schema(quality["column_summary"]), hide_index=True, column_config={
        "Missing": st.column_config.NumberColumn(format="localized"),
        "Description": st.column_config.TextColumn(width="large")})

    st.subheader("Category mappings")
    categorical = [c for c, m in VARIABLE_METADATA.items() if m["category_labels"] and c in frame]
    column = st.selectbox("Variable", categorical, format_func=variable_label)
    counts = services.get_eda_summary(frame, column)["summary"]
    st.dataframe(counts[["category", "label", "count", "percentage"]], hide_index=True, column_config={
        "category": "Code", "label": "Label",
        "count": st.column_config.NumberColumn("Respondents", format="localized"),
        "percentage": st.column_config.NumberColumn("Share (%)", format="%.2f")})

    st.subheader("Descriptive statistics (numeric variables)")
    numeric = bundle["tables"]["numeric_summary"].drop(columns="dataset_variant").astype(float)
    numeric.index = [f"{VARIABLE_METADATA[c]['label']} ({c})" for c in numeric.index]
    st.dataframe(numeric, column_config={
        "n": st.column_config.NumberColumn("n", format="localized"),
        "missing_count": st.column_config.NumberColumn("Missing", format="localized"),
        **{c: st.column_config.NumberColumn(STATISTIC_LABELS.get(c, c), format="%.2f")
           for c in numeric.columns if c not in ("n", "missing_count")}})

    st.subheader("Data quality")
    checks, outliers = st.columns(2)
    with checks:
        st.markdown("**Raw file against the verified reference snapshot**")
        reference = data["results"]["raw_validation"]["reference_checks"]
        st.dataframe(pd.DataFrame({"Check": [REFERENCE_CHECKS.get(k, k) for k in reference],
                                   "Matches": list(reference.values())}), hide_index=True)
        invalid = [v["column"] for v in bundle["validation"]["variables"].values() if not v["is_valid"]]
        st.markdown(f"Variables failing type or range checks: {', '.join(invalid) if invalid else 'none'}. "
                    f"Constant columns: {', '.join(quality['constant_columns']) or 'none'}.")
    with outliers:
        for column, flags in quality["potential_outliers"].items():
            st.markdown(f"**{VARIABLE_METADATA[column]['label']} outside the 1.5 × IQR fences**")
            metric_row({"Lower fence": flags["lower_boundary"], "Upper fence": flags["upper_boundary"]}, decimals=1)
            metric_row({"Flagged rows": flags["flagged_count"], "Flagged (%)": flags["flagged_percent"]}, decimals=1)
            st.caption(flags["decision"])

    show_warnings(bundle["validation"]["warnings"] + data["warnings"])


def _schema(columns: dict) -> pd.DataFrame:
    return pd.DataFrame([{"Variable": name, "Label": meta["label"], "Role": meta["role"], "Type": meta["type"],
                          "Stored as": columns[name]["dtype"], "Distinct values": columns[name]["unique_values"],
                          "Missing": columns[name]["missing_count"], "Unit": meta["unit"] or "",
                          "Description": meta["description"]}
                         for name, meta in VARIABLE_METADATA.items() if name in columns])
