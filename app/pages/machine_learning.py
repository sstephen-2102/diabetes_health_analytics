"""Streamlit page: machine_learning."""
import streamlit as st

from app import artifacts
from app.components import charts
from app.components.results import load_or_stop, show_context, show_warnings
from app.components.tables import METRIC_COLUMNS, model_label, show_metrics_table


def render():
    """Render page controls and structured results."""
    st.title("Machine Learning Lab")
    outputs = load_or_stop(artifacts.modeling_outputs)
    results, meta = outputs["results"], outputs["metadata"]
    show_context(outputs, f"Test set: {meta['test_rows']:,} rows · Leakage check: {meta['leakage_check']}")

    st.subheader("Model comparison at threshold 0.5")
    comparison = results["comparison"]
    show_metrics_table(comparison[METRIC_COLUMNS])

    st.subheader("Confusion matrix")
    name = st.selectbox("Model", meta["models"], format_func=model_label)
    st.plotly_chart(charts.confusion_matrix_chart(comparison.loc[name]))

    roc_tab, pr_tab = st.tabs(["ROC curves", "Precision-recall curves"])
    roc_tab.plotly_chart(charts.roc_chart(results["roc"], comparison["roc_auc"].to_dict()))
    pr_tab.plotly_chart(charts.precision_recall_chart(results["precision_recall"], comparison["pr_auc"].to_dict(),
                                                      meta["test_prevalence"]))

    show_warnings(outputs["warnings"] + [
        f"Train/test split is grouped by identical feature profiles ({meta['train_rows']:,} train rows), "
        "so no profile appears in both sets.",
        f"Only {meta['test_prevalence']:.1%} of test rows are positive: accuracy alone is misleading, "
        "and the no-skill precision-recall line sits at that prevalence.",
        "Recall is low at 0.5; see the Imbalance & Threshold Lab for the trade-off at other thresholds.",
    ])