"""Streamlit page: imbalance_threshold."""
import streamlit as st

from app import artifacts, services
from app.components import charts
from app.components.metrics import metric_row
from app.components.results import load_or_stop, show_context, show_warnings
from app.components.tables import METRIC_COLUMNS, model_label, show_metrics_table


def render():
    """Render page controls and structured results."""
    st.title("Imbalance & Threshold Lab")
    outputs = load_or_stop(artifacts.modeling_outputs)
    results, meta = outputs["results"], outputs["metadata"]
    show_context(outputs)

    st.subheader("Class distribution in the training set")
    imbalance = results["imbalance"]
    original = imbalance[imbalance["strategy"] == "original"].iloc[0]
    negatives, positives = int(original["train_negatives_before"]), int(original["train_positives_before"])
    metric_row({"Negative (0)": negatives, "Positive (1)": positives,
                "Positive share": positives / (negatives + positives)})

    st.subheader("Training strategies")
    st.plotly_chart(charts.imbalance_chart(imbalance, [m for m in METRIC_COLUMNS if m != "accuracy"]))
    with st.expander("All strategy results"):
        st.dataframe(imbalance.set_index(["strategy", "model"]).drop(columns=["dataset_variant", "seed"]))

    st.subheader("Threshold trade-off")
    grid = meta["threshold_grid"]
    left, right = st.columns(2)
    name = left.selectbox("Model", meta["models"], format_func=model_label)
    threshold = right.select_slider("Threshold", grid, value=meta["threshold"])
    at_threshold = services.get_threshold_comparison(results["thresholds"], threshold)
    row = at_threshold.loc[name]
    metric_row({"Precision": row["precision"], "Recall": row["recall"],
                "Specificity": row["specificity"], "F1": row["f1"],
                "Flagged positive": int(row["true_positives"] + row["false_positives"])})
    st.plotly_chart(charts.threshold_chart(results["thresholds"][name], threshold))
    st.markdown(f"**All models at threshold {threshold:.2f}**")
    show_metrics_table(at_threshold.drop(columns="threshold"))

    show_warnings([
        "Class weighting and undersampling change only the training data; every strategy is scored on the same "
        "untouched test set.",
        "Rebalancing mainly moves the effective threshold: it raises recall and lowers precision, while ROC AUC "
        "barely changes. It also inflates probabilities, which shows up as a worse Brier score.",
        "At high thresholds few rows are flagged positive, so precision there rests on small counts and jumps "
        "around; check Flagged positive before reading it.",
        f"Threshold choice: {meta['threshold_selection']}. Pick a threshold by the cost of missed positives "
        "versus false alarms, not by maximising one metric on this table.",
    ])