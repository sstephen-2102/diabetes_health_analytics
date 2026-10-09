"""Streamlit page: machine_learning."""
import streamlit as st

from app import artifacts
from app.components import charts
from app.components.results import load_or_stop, show_context, show_warnings
from app.components.tables import METRIC_COLUMNS, model_label, show_metrics_table
from src.common.exceptions import DataLoadError

SCENARIO_LABELS = {"unique_grouped": "Unique, grouped (primary)", "unique_random": "Unique, random",
                   "full_grouped": "Full, grouped", "full_random": "Full, random"}
SENSITIVITY_METRICS = {"ROC AUC": "roc_auc", "PR AUC": "pr_auc", "Brier score": "brier_score",
                       "Recall": "recall", "Precision": "precision", "Accuracy": "accuracy"}


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

    _duplicate_sensitivity()


def _duplicate_sensitivity():
    """Optional section: shown only when the sensitivity export exists, without stopping the page."""
    st.subheader("Duplicate-handling sensitivity")
    try:
        sensitivity = artifacts.duplicate_sensitivity()
    except DataLoadError as exc:
        st.info(str(exc))
        return
    show_context(sensitivity, f"Threshold {sensitivity['metadata']['threshold']}")

    scenarios = sensitivity["results"]["scenarios"].rename(index=SCENARIO_LABELS).rename_axis("Scenario")
    st.dataframe(scenarios.drop(columns=["dataset_variant", "split"]), column_config={
        "train_rows": st.column_config.NumberColumn("Train rows", format="%d"),
        "test_rows": st.column_config.NumberColumn("Test rows", format="%d"),
        "seen_test_rows": st.column_config.NumberColumn("Seen in train", format="%d"),
        "seen_test_share": st.column_config.NumberColumn("Seen share", format="percent"),
    })

    label = st.selectbox("Metric (all test rows)", list(SENSITIVITY_METRICS))
    rows = sensitivity["results"]["all_test_rows"]
    table = rows.pivot(index="model", columns="scenario", values=SENSITIVITY_METRICS[label])
    table = table.loc[rows["model"].unique(), list(SCENARIO_LABELS)]
    table = table.rename(index=model_label, columns=SCENARIO_LABELS).rename_axis(index="Model", columns=None)
    st.dataframe(table, column_config={c: st.column_config.NumberColumn(c, format="%.3f") for c in table.columns})

    show_warnings(sensitivity["warnings"] + [
        "A test row is 'seen' when its exact feature vector is also in the training set; grouped splits have none."
    ], title="Reading this comparison")
