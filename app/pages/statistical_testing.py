"""Streamlit page: statistical_testing."""
import pandas as pd
import streamlit as st

from app import artifacts, services
from app.components import charts
from app.components.metrics import format_p_value, metric_row
from app.components.results import load_or_stop, select_variant, show_context, show_warnings, variable_label
from src.common.config import FEATURE_COLUMNS, PROJECT_CONFIG, TARGET_DEFINITION, VARIABLE_METADATA

TARGET = PROJECT_CONFIG["target_column"]
LABELS = {name: meta["label"] for name, meta in VARIABLE_METADATA.items()}
TARGET_LABELS = VARIABLE_METADATA[TARGET]["category_labels"]
CATEGORICAL = [c for c in FEATURE_COLUMNS if VARIABLE_METADATA[c].get("category_labels")]
NUMERIC = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL]
STATEMENTS = {
    "H1": "The Diabetes_binary outcome is associated with HighBP.",
    "H2": "The Diabetes_binary outcome is associated with HighChol.",
    "H3": "Mean BMI differs between target groups.",
    "H4": "The Diabetes_binary outcome is associated with physical activity.",
    "H5": "The Diabetes_binary outcome is associated with smoking.",
    "H6": "The target outcome differs across age categories.",
    "H7": "The target outcome differs across income categories.",
}
TESTS = {"chi_square": "Chi-square test of independence (categorical variables)",
         "welch": "Welch t-test (numeric variables)"}
CONFIDENCE_LEVELS = [0.90, 0.95, 0.99]
COUNT = st.column_config.NumberColumn(format="localized")
TWO_DECIMALS = st.column_config.NumberColumn(format="%.2f")
SCOPE_NOTE = ("ANOVA and post-hoc comparisons are not included: H1-H7 need only chi-square and Welch tests, and the "
              "team scoped them out (docs/MEMBER2_HANDOFF.md).")


def render():
    """Render page controls and structured results."""
    st.title("Statistical Testing Lab")
    st.markdown("Hypothesis tests, effect sizes and confidence intervals for the association between each feature "
                "and category 1 (prediabetes or diabetes). With this many respondents almost every test is "
                "significant, so read the effect sizes to judge how large an association is.")
    data = load_or_stop(artifacts.eda_data)
    name = select_variant(data["results"]["variants"])
    frame = data["results"]["variants"][name]["data"]
    show_context({"dataset_variant": name, "target_definition": TARGET_DEFINITION}, f"{len(frame):,} rows")
    st.caption("full_clean_v1 is the primary sample for inference; unique_profile_v1 is the sensitivity check.")

    hypotheses, single, intervals = st.tabs(["Hypotheses H1-H7", "Run a test", "Confidence intervals"])
    with hypotheses:
        warnings = _hypotheses(frame)
    with single:
        _single_test(frame)
    with intervals:
        _intervals(frame)
    show_warnings(warnings + [SCOPE_NOTE])


def _hypotheses(frame) -> list[str]:
    summary = services.get_hypothesis_summary(frame, TARGET)
    table = summary["results"]["hypotheses"]
    shown = table.assign(statement=table["hypothesis"].map(STATEMENTS), feature=table["feature"].map(LABELS),
                         p_value=table["p_value"].map(format_p_value),
                         effect_measure=table["effect_measure"].map({"cramers_v": "Cramér's V", "cohens_d": "Cohen's d"}))
    st.dataframe(shown[["hypothesis", "feature", "test", "n", "statistic", "dof", "p_value", "effect_measure",
                        "effect_value", "odds_ratio", "or_lower", "or_upper", "summary", "statement"]].rename(
        columns={"hypothesis": "Hypothesis", "statement": "Alternative hypothesis", "feature": "Feature",
                 "test": "Test", "statistic": "Statistic", "dof": "df", "p_value": "p", "effect_measure": "Effect size",
                 "effect_value": "Value", "odds_ratio": "Odds ratio", "or_lower": "OR 95% CI lower",
                 "or_upper": "OR 95% CI upper", "summary": "What the data show"}),
        hide_index=True, column_config={"n": COUNT, "Statistic": TWO_DECIMALS, "df": TWO_DECIMALS,
                                        "Value": st.column_config.NumberColumn(format="%.3f"),
                                        **{c: TWO_DECIMALS for c in ["Odds ratio", "OR 95% CI lower",
                                                                     "OR 95% CI upper"]}})
    st.caption("Chi-square statistics have df = categories - 1; the Welch t-test (H3) uses Satterthwaite df. "
               "Odds ratios compare code 1 with code 0 and are shown only for two-category features.")

    chi_square = table[table["effect_measure"] == "cramers_v"]
    st.plotly_chart(charts.cramers_v_chart(chi_square, LABELS))
    for row in table[table["effect_measure"] == "cohens_d"].itertuples():
        st.markdown(f"**{row.hypothesis} ({LABELS[row.feature]})**: Cohen's d = {row.effect_value:.2f}; "
                    f"{row.summary}.")
    st.caption("Common rules of thumb: Cramér's V of 0.1, 0.3 and 0.5 (two-category features) and Cohen's d of "
               "0.2, 0.5 and 0.8 are small, medium and large. They are conventions, not clinical thresholds.")

    odds = table.dropna(subset=["odds_ratio"])
    st.plotly_chart(charts.odds_ratio_chart(
        odds.assign(feature=odds["feature"].map(LABELS)).rename(columns={"or_lower": "ci_lower", "or_upper": "ci_upper"}),
        "Unadjusted odds ratios (code 1 vs code 0)"))
    st.caption("Unadjusted: each feature is tested on its own. The Regression Lab adjusts for all features at once.")
    return summary["warnings"]


def _single_test(frame):
    left, right = st.columns(2)
    test = left.radio("Test", list(TESTS), format_func=TESTS.get)
    feature = right.selectbox("Feature", CATEGORICAL if test == "chi_square" else NUMERIC, format_func=variable_label,
                              key=f"test_feature_{test}")
    if test == "chi_square":
        _chi_square(services.get_statistical_test_result(frame, "chi_square", feature, TARGET, {}), feature)
    else:
        level = st.select_slider("Confidence level", CONFIDENCE_LEVELS, value=0.95, format_func="{:.0%}".format,
                                 key="welch_level")
        _welch(frame, services.get_statistical_test_result(frame, "welch", feature, TARGET,
                                                           {"confidence_level": level}), feature)


def _chi_square(result: dict, feature: str):
    chi = result["results"]
    metric_row({"Chi-square": f"{chi['chi_square']:,.1f}", "df": chi["degrees_of_freedom"],
                "p": format_p_value(chi["p_value"]), "Cramér's V": f"{chi['cramers_v']:.3f}", "n": chi["n"]})
    labels = VARIABLE_METADATA[feature]["category_labels"]
    table, shares = chi["contingency_table"], chi["row_percentages"]
    st.dataframe(pd.DataFrame({
        "Code": table.index, "Label": [labels.get(code, code) for code in table.index],
        "Category 0": table[0].to_numpy(), "Category 1": table[1].to_numpy(),
        "Category 1 share (%)": shares[1].to_numpy()}), hide_index=True,
        column_config={"Category 0": COUNT, "Category 1": COUNT, "Category 1 share (%)": TWO_DECIMALS})
    if "odds_ratio" in chi:
        odds = chi["odds_ratio"]
        st.markdown(f"Odds ratio, {labels[1].lower()} vs {labels[0].lower()}: **{odds['value']:.2f}** "
                    f"(95% CI {odds['lower']:.2f} to {odds['upper']:.2f}, {odds['method']}).")
    with st.expander("Expected counts under independence"):
        st.dataframe(chi["expected_counts"].rename(index=labels, columns=TARGET_LABELS),
                     column_config={code: st.column_config.NumberColumn(format="%.1f") for code in TARGET_LABELS.values()})
    st.caption(f"{chi['test'].capitalize()}.")
    if chi["assumption_warnings"]:
        show_warnings(chi["assumption_warnings"], "Assumption checks")
    else:
        st.caption("Every expected count is at least 5, so the chi-square approximation is appropriate.")


def _welch(frame, result: dict, feature: str):
    welch = result["results"]
    interval = welch["confidence_interval"]
    level = f"{interval['confidence_level']:.0%}"
    metric_row({"Mean, category 1": welch["mean_a"], "Mean, category 0": welch["mean_b"],
                "Difference (1 - 0)": welch["mean_difference"]}, decimals=2)
    st.markdown(f"Mean difference, category 1 minus category 0: **{welch['mean_difference']:.2f}** "
                f"{VARIABLE_METADATA[feature]['unit']} ({level} CI {interval['lower']:.2f} to {interval['upper']:.2f}).")
    metric_row({"t": f"{welch['t_statistic']:.2f}", "df": f"{welch['degrees_of_freedom']:,.0f}",
                "p": format_p_value(welch["p_value"]), "Cohen's d": f"{welch['cohens_d']:.2f}"})
    st.dataframe(pd.DataFrame({"Target group": [TARGET_LABELS[1], TARGET_LABELS[0]],
                               "n": [welch["n_a"], welch["n_b"]], "Mean": [welch["mean_a"], welch["mean_b"]],
                               "Std. deviation": [welch["std_a"], welch["std_b"]]}),
                 hide_index=True, column_config={"n": COUNT, "Mean": TWO_DECIMALS, "Std. deviation": TWO_DECIMALS})
    meta = VARIABLE_METADATA[feature]
    st.plotly_chart(charts.value_distribution_chart(services.get_value_distribution(frame, feature), TARGET_LABELS,
                                                    f"{meta['label']} by target group",
                                                    f"{meta['label']} ({meta['unit']})"))
    st.caption(f"{welch['test']}. Differences are category 1 minus category 0, in {meta['unit']}. The test "
               "compares means; with skewed values a mean difference can hide differences in shape.")
    show_warnings(welch["assumption_warnings"], "Assumption checks")


def _intervals(frame):
    left, right = st.columns(2)
    feature = left.selectbox("Variable", FEATURE_COLUMNS, format_func=variable_label, key="interval_feature")
    level = right.select_slider("Confidence level", CONFIDENCE_LEVELS, value=0.95, format_func="{:.0%}".format,
                                key="interval_level")
    result = services.get_confidence_intervals(frame, feature, TARGET, level)["results"]
    table = result["table"]
    percent = f"{level:.0%}"
    if result["kind"] == "proportion":
        overall = table.iloc[0]["estimate"]
        st.plotly_chart(charts.interval_chart(table, f"Category 1 share by {LABELS[feature].lower()}, {percent} CI",
                                              "Category 1 share (%)", reference=overall, from_zero=True))
        columns = {"estimate": "Category 1 share (%)", "lower": f"{percent} CI lower", "upper": f"{percent} CI upper"}
    else:
        unit = VARIABLE_METADATA[feature]["unit"]
        st.plotly_chart(charts.interval_chart(table, f"Mean {LABELS[feature].lower()} by target group, {percent} CI",
                                              f"Mean ({unit})"))
        difference = result["difference"]
        st.markdown(f"Difference in means (category 1 minus category 0): **{difference['difference']:.2f}** {unit} "
                    f"({percent} CI {difference['lower']:.2f} to {difference['upper']:.2f}).")
        columns = {"estimate": "Mean", "lower": f"{percent} CI lower", "upper": f"{percent} CI upper"}
    st.dataframe(table.rename(columns={"group": "Group", **columns}), hide_index=True,
                 column_config={"n": COUNT, **{c: TWO_DECIMALS for c in columns.values()}})
    st.caption(f"Method: {result['method']}. An interval describes sampling uncertainty in this dataset only; it "
               "does not account for survey design or measurement error.")
