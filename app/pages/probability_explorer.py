"""Streamlit page: probability_explorer."""
import pandas as pd
import streamlit as st

from app import artifacts, services
from app.components import charts
from app.components.metrics import metric_row
from app.components.results import load_or_stop, select_variant, show_context, show_warnings, variable_label
from src.common.config import FEATURE_COLUMNS, PROJECT_CONFIG, TARGET_DEFINITION, VARIABLE_METADATA

TARGET = PROJECT_CONFIG["target_column"]
LABELS = {name: meta["label"] for name, meta in VARIABLE_METADATA.items()}
CATEGORICAL = [c for c in FEATURE_COLUMNS if VARIABLE_METADATA[c].get("category_labels")]
MAX_CONDITIONS = 4
COUNT = st.column_config.NumberColumn(format="localized")


def render():
    """Render page controls and structured results."""
    st.title("Probability Explorer")
    st.markdown("Observed probabilities of category 1 (prediabetes or diabetes) in the survey sample: overall, "
                "for respondents who meet chosen conditions, and rebuilt through Bayes' theorem.")
    data = load_or_stop(artifacts.eda_data)
    name = select_variant(data["results"]["variants"])
    frame = data["results"]["variants"][name]["data"]
    show_context({"dataset_variant": name, "target_definition": TARGET_DEFINITION}, f"{len(frame):,} rows")
    st.caption("full_clean_v1 is the primary sample for inference; unique_profile_v1 is the sensitivity check.")

    conditional, bayes = st.tabs(["Conditional probability", "Bayes' theorem"])
    with conditional:
        result = _conditional(frame)
    with bayes:
        _bayes(frame)
    show_warnings(result["warnings"])


def _value_label(feature: str, code) -> str:
    return f"{code}: {VARIABLE_METADATA[feature]['category_labels'][code]}"


def _describe(conditions: dict) -> str:
    return " and ".join(f"{LABELS[f]} = {VARIABLE_METADATA[f]['category_labels'][v]}" for f, v in conditions.items())


def _choose_conditions() -> dict:
    features = st.multiselect(f"Condition on (up to {MAX_CONDITIONS} variables)", CATEGORICAL, default=["HighBP"],
                              format_func=variable_label, max_selections=MAX_CONDITIONS)
    conditions = {}
    for column, feature in zip(st.columns(MAX_CONDITIONS), features):
        codes = list(VARIABLE_METADATA[feature]["category_labels"])
        conditions[feature] = column.selectbox(LABELS[feature], codes, index=1 if len(codes) == 2 else 0,
                                               format_func=lambda code, f=feature: _value_label(f, code),
                                               key=f"condition_{feature}")
    return conditions


def _conditional(frame) -> dict:
    conditions = _choose_conditions()
    result = services.get_probability_analysis(frame, TARGET, 1, conditions)
    results = result["results"]
    marginal = results["marginal"]
    if not conditions:
        metric_row({"Overall share (%)": marginal["probability"] * 100,
                    "Respondents": marginal["denominator"]}, decimals=2)
        st.info("Choose one or more conditions to see how the category 1 share changes.")
        return result

    conditional = results["conditional"]
    if conditional["probability"] is None:
        st.warning(f"No respondents have {_describe(conditions)} in {result['dataset_variant']}, "
                   "so the conditional probability is undefined.")
        return result

    interval = conditional["confidence_interval"]
    metric_row({"Overall share (%)": marginal["probability"] * 100,
                "Share given conditions (%)": conditional["probability"] * 100,
                "Respondents meeting conditions": conditional["denominator"]}, decimals=2)
    st.markdown(f"Of the **{conditional['denominator']:,}** respondents with {_describe(conditions)}, "
                f"**{conditional['numerator']:,}** are in category 1: {conditional['probability']:.2%} "
                f"(95% CI {interval['lower']:.2%} to {interval['upper']:.2%}), against "
                f"{marginal['probability']:.2%} overall.")

    overall_interval = marginal["confidence_interval"]
    st.plotly_chart(charts.interval_chart(pd.DataFrame([
        {"group": "All respondents", "estimate": marginal["probability"] * 100, "n": marginal["denominator"],
         "lower": overall_interval["lower"] * 100, "upper": overall_interval["upper"] * 100},
        {"group": "Given the conditions", "estimate": conditional["probability"] * 100,
         "n": conditional["denominator"], "lower": interval["lower"] * 100, "upper": interval["upper"] * 100},
    ]), "Category 1 share with 95% Wilson interval", "Category 1 share (%)", from_zero=True))

    st.subheader("How the probabilities fit together")
    joint, joint_target = results["joint_conditions"], results["joint_with_target"]
    st.dataframe(pd.DataFrame([
        {"Probability": "P(category 1)", "Kind": "Marginal", "Numerator": marginal["numerator"],
         "Denominator": marginal["denominator"], "Value": marginal["probability"]},
        {"Probability": "P(conditions)", "Kind": "Joint (conditions only)", "Numerator": joint["numerator"],
         "Denominator": joint["denominator"], "Value": joint["probability"]},
        {"Probability": "P(conditions and category 1)", "Kind": "Joint", "Numerator": joint_target["numerator"],
         "Denominator": joint_target["denominator"], "Value": joint_target["probability"]},
        {"Probability": "P(category 1 | conditions)", "Kind": "Conditional", "Numerator": conditional["numerator"],
         "Denominator": conditional["denominator"], "Value": conditional["probability"]},
    ]), hide_index=True, column_config={"Numerator": COUNT, "Denominator": COUNT,
                                        "Value": st.column_config.NumberColumn(format="%.4f")})
    st.caption("P(category 1 | conditions) = P(conditions and category 1) / P(conditions). "
               f"{conditional['note']} The Prediction Explorer gives model estimates instead.")
    return result


def _bayes(frame):
    st.latex(r"P(\text{category 1} \mid X = x) = "
             r"\frac{P(X = x \mid \text{category 1}) \times P(\text{category 1})}{P(X = x)}")
    left, right = st.columns(2)
    feature = left.selectbox("Feature X", CATEGORICAL, format_func=variable_label, key="bayes_feature")
    codes = list(VARIABLE_METADATA[feature]["category_labels"])
    value = right.selectbox("Value x", codes, index=1 if len(codes) == 2 else 0,
                            format_func=lambda code: _value_label(feature, code), key=f"bayes_value_{feature}")
    check = services.get_probability_analysis(frame, TARGET, 1, {feature: value})["results"]["bayes"]
    if check["matches"] is None:
        st.warning(f"{check['warning']} No respondents have {_describe({feature: value})}.")
    else:
        metric_row({"Prior": check["prior"], "Likelihood": check["likelihood"], "Evidence": check["evidence"],
                    "Bayes result": check["bayes_rhs"], "Direct count": check["direct_probability"]}, decimals=4)
        st.caption("Prior = P(category 1); likelihood = P(x | category 1); evidence = P(x); "
                   "Bayes result = likelihood × prior / evidence; direct count = P(category 1 | x) counted directly.")
        verdict = "agree" if check["matches"] else "do not agree"
        st.markdown(f"Bayes' theorem and the direct count {verdict} (absolute difference "
                    f"{check['absolute_difference']:.1e}). Knowing {_describe({feature: value})} moves the "
                    f"category 1 probability from {check['prior']:.2%} to {check['direct_probability']:.2%}.")

    st.subheader("Bayes check for seven binary features")
    table = services.get_bayes_summary(frame, TARGET)["results"]["bayes"]
    shown = table.assign(feature=table["feature"].map(LABELS))[[
        "feature", "prior_p_category1", "likelihood_p_feature_given_category1", "evidence_p_feature", "bayes_rhs",
        "direct_p_category1_given_feature", "absolute_difference", "matches", "lift_over_prior"]]
    probability = st.column_config.NumberColumn(format="%.4f")
    st.dataframe(shown.rename(columns={
        "feature": "Feature (value 1 = Yes)", "prior_p_category1": "Prior", "likelihood_p_feature_given_category1":
        "Likelihood", "evidence_p_feature": "Evidence", "bayes_rhs": "Bayes result",
        "direct_p_category1_given_feature": "Direct count", "absolute_difference": "Difference", "matches": "Agree",
        "lift_over_prior": "Lift over prior"}), hide_index=True,
        column_config={**{c: probability for c in ["Prior", "Likelihood", "Evidence", "Bayes result", "Direct count"]},
                       "Difference": st.column_config.NumberColumn(format="%.1e"),
                       "Lift over prior": st.column_config.NumberColumn(format="%.2f")})
    st.caption("Lift over prior = direct count / prior: above 1, the feature value raises the observed category 1 "
               "share; below 1, it lowers it. Agreement is exact up to floating-point rounding because both sides "
               "come from the same counts.")
