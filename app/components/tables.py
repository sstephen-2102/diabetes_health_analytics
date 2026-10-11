"""Presentation components: tables."""
import pandas as pd
import streamlit as st

METRIC_COLUMNS = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc", "brier_score"]
COUNT_COLUMNS = ["true_negatives", "false_positives", "false_negatives", "true_positives", "n"]


def model_label(name: str) -> str:
    return name.replace("_", " ").capitalize()


def show_metrics_table(table: pd.DataFrame, decimals: int = 3) -> None:
    """Metric columns rounded, count columns as integers, model names readable."""
    shown = table.rename(index=model_label)
    formats = {c: f"%.{decimals}f" for c in shown.columns if c in METRIC_COLUMNS}
    formats |= {c: "%d" for c in shown.columns if c in COUNT_COLUMNS}
    st.dataframe(shown, column_config={c: st.column_config.NumberColumn(c.replace("_", " "), format=f)
                                       for c, f in formats.items()})