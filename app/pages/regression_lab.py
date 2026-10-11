"""Streamlit page: regression_lab."""
import streamlit as st

from app import artifacts
from app.components import charts
from app.components.metrics import metric_row
from app.components.results import load_or_stop, show_context, show_warnings


def render():
    """Render page controls and structured results."""
    st.title("Regression Lab")
    regression = load_or_stop(artifacts.regression_outputs)
    results, meta = regression["results"], regression["metadata"]
    show_context(regression, f"n = {meta['n']:,}")

    st.subheader("Does the BMI association change with age?")
    term, test = results["interaction_term"], results["model_comparison_test"]
    metric_row({"BMI × Age odds ratio": term["odds_ratio"], "95% CI lower": term["ci_lower"],
                "95% CI upper": term["ci_upper"]}, decimals=4)
    st.write(f"Adding the interaction lowers AIC by {test['delta_aic']:.1f} and BIC by {test['delta_bic']:.1f}. "
             f"Likelihood-ratio test: χ² = {test['lr_statistic']:.1f} on {test['lr_df']} df, "
             f"p = {test['lr_p_value']:.2e}.")
    if not all(meta["converged"].values()):
        st.error(f"Not every model converged: {meta['converged']}. Estimates are unreliable.")
    st.dataframe(results["model_comparison"].drop(columns="dataset_variant").set_index("model"))

    by_age = results["bmi_odds_by_age"]
    first, last = by_age.iloc[0], by_age.iloc[-1]
    st.plotly_chart(charts.bmi_by_age_chart(by_age))
    st.write(f"Holding the other features fixed, each extra BMI point multiplies the odds of the positive class "
             f"by {first['bmi_odds_ratio']:.3f} at age {first['age_group']} and by {last['bmi_odds_ratio']:.3f} "
             f"at {last['age_group']}.")
    st.caption(f"BMI and Age are centered at their means ({meta['centering_means']['BMI']:.1f} and age code "
               f"{meta['centering_means']['Age']:.1f}) before forming the interaction.")

    st.subheader("Odds ratios")
    model = st.radio("Model", ["Baseline", "Interaction"], horizontal=True)
    if model == "Baseline":
        odds = results["baseline_odds_ratios"]
    else:
        odds = results["interaction_coefficients"].query("feature != 'const'")
    st.plotly_chart(charts.odds_ratio_chart(odds, f"{model} model odds ratios"))

    with st.expander("Coefficient tables"):
        st.markdown("**Baseline**")
        st.dataframe(results["baseline_coefficients"].drop(columns="dataset_variant"), hide_index=True)
        st.markdown("**Interaction**")
        st.dataframe(results["interaction_coefficients"].drop(columns="dataset_variant"), hide_index=True)
    with st.expander("How features are coded"):
        st.table({"Feature": list(meta["feature_treatment"]), "Treatment": list(meta["feature_treatment"].values())})

    show_warnings(regression["warnings"], "Limitations")