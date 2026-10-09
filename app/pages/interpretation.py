"""Streamlit page: interpretation."""
import pandas as pd
import streamlit as st

from app import artifacts
from app.components import charts
from app.components.results import load_or_stop, show_context, show_warnings
from app.components.tables import model_label


def render():
    """Render page controls and structured results."""
    st.title("Interpretation")
    outputs = load_or_stop(artifacts.modeling_outputs)
    results, meta = outputs["results"], outputs["metadata"]
    importance_setup = meta["importance"]
    show_context(outputs)

    st.subheader("Permutation importance")
    left, right = st.columns([2, 1])
    name = left.selectbox("Model", meta["models"], format_func=model_label)
    top_n = right.slider("Features shown", 5, len(results["importance"][name]), 10)
    st.plotly_chart(charts.importance_chart(results["importance"][name], top_n,
                                            f"Permutation importance - {model_label(name)}"))

    st.markdown("**Importance rank by model** (1 = largest drop in ROC AUC)")
    ranks = pd.DataFrame({model_label(n): t.set_index("feature")["importance_mean"].rank(ascending=False)
                          for n, t in results["importance"].items()}).astype(int)
    st.dataframe(ranks.sort_values(ranks.columns[0]))

    st.subheader("Odds ratios (logistic regression, full dataset)")
    regression = load_or_stop(artifacts.regression_outputs)
    show_context(regression)
    odds = regression["results"]["baseline_odds_ratios"]
    st.plotly_chart(charts.odds_ratio_chart(odds, "Baseline model odds ratios"))

    show_warnings([
        f"Importance is the mean drop in {importance_setup['scoring'].upper().replace('_', ' ')} over "
        f"{importance_setup['n_repeats']} shuffles of each feature, on {importance_setup['sample_rows']:,} rows "
        f"of the {importance_setup['sample_source']}.",
        "Correlated features share credit: shuffling one (e.g. GenHlth) leaves related ones (PhysHlth, DiffWalk) "
        "to carry the signal, so each can look less important than it is.",
        "Odds ratios are adjusted for the other features and depend on each feature's units (BMI per point, "
        "Age per age band, binary features yes vs no).",
        "Both views describe associations in this survey sample; neither shows cause and effect.",
    ])