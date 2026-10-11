"""Streamlit page: exploratory_analysis."""
import streamlit as st

from app import artifacts, services
from app.components import charts
from app.components.metrics import metric_row
from app.components.results import load_or_stop, select_variant, show_context, show_warnings, variable_label
from src.common.config import FEATURE_COLUMNS, PROJECT_CONFIG, TARGET_DEFINITION, VARIABLE_METADATA

TARGET = PROJECT_CONFIG["target_column"]
LABELS = {name: meta["label"] for name, meta in VARIABLE_METADATA.items()}
TARGET_LABELS = VARIABLE_METADATA[TARGET]["category_labels"]
HEALTH_DAYS = {"MentHlth": "mental", "PhysHlth": "physical"}
COUNT = st.column_config.NumberColumn(format="localized")
ESTIMATES = {"category_1_percent": "Category 1 share (%)", "mean": "Mean"}


def render():
    """Render page controls and structured results."""
    st.title("Exploratory Analysis")
    data = load_or_stop(artifacts.eda_data)
    name = select_variant(data["results"]["variants"])
    bundle = data["results"]["variants"][name]
    show_context({"dataset_variant": name, "target_definition": TARGET_DEFINITION},
                 f"{len(bundle['data']):,} rows")
    overall = bundle["tables"]["target_distribution"].set_index("category").loc[1, "percentage"]

    variable, binary, correlation, sensitivity = st.tabs(
        ["Variable explorer", "Binary indicators", "Correlations", "Full vs unique sample"])
    with variable:
        _variable_explorer(bundle, overall)
    with binary:
        _binary_indicators(bundle["tables"]["binary_comparisons"])
    with correlation:
        _correlations(bundle["tables"]["spearman_correlation"])
    with sensitivity:
        _sensitivity(data["results"]["sensitivity"])

    show_warnings(data["warnings"])


def _variable_explorer(bundle: dict, overall: float):
    feature = st.selectbox("Variable", FEATURE_COLUMNS, format_func=variable_label)
    eda = services.get_eda_summary(bundle["data"], feature)
    meta = eda["feature_metadata"]
    st.caption(meta["description"])

    if "prevalence" in eda:
        st.plotly_chart(charts.category_share_chart(eda["summary"], f"Distribution of {meta['label'].lower()}"))
        st.plotly_chart(charts.prevalence_chart(eda["prevalence"], overall,
                                                f"Category 1 share by {meta['label'].lower()}"))
        st.dataframe(eda["prevalence"].rename(columns={
            "category": "Code", "label": "Label", "total_observations": "Respondents",
            "target_count": "Category 1", "target_percentage": "Category 1 share (%)"}),
            hide_index=True, column_config={"Respondents": COUNT, "Category 1": COUNT,
                                            "Category 1 share (%)": st.column_config.NumberColumn(format="%.2f")})
        return

    summary = eda["summary"]
    metric_row({"Mean": summary["mean"], "Median": summary["median"], "Std. deviation": summary["std"],
                "IQR": summary["iqr"], "Min": summary["min"], "Max": summary["max"]}, decimals=2)
    distribution = services.get_value_distribution(bundle["data"], feature)
    st.plotly_chart(charts.value_distribution_chart(distribution, TARGET_LABELS, f"{meta['label']} by target group",
                                                    f"{meta['label']} ({meta['unit']})"))
    by_target = eda["by_target"].assign(**{TARGET: lambda t: t[TARGET].map(TARGET_LABELS)})
    st.dataframe(by_target.rename(columns={TARGET: "Target group", "count": "n"}), hide_index=True,
                 column_config={"n": COUNT, **{c: st.column_config.NumberColumn(format="%.2f")
                                               for c in ["mean", "std", "min", "25%", "50%", "75%", "max"]}})
    if feature in HEALTH_DAYS:
        kind = HEALTH_DAYS[feature]
        days = bundle["tables"]["health_days_by_target"].rename(index=TARGET_LABELS)
        st.dataframe(days[[f"{kind}_any_percent", f"{kind}_30_percent"]].rename(
            columns={f"{kind}_any_percent": "Any poor days (%)", f"{kind}_30_percent": "All 30 days (%)"})
            .rename_axis("Target group"), column_config={c: st.column_config.NumberColumn(format="%.1f")
                                                        for c in ["Any poor days (%)", "All 30 days (%)"]})
    show_warnings(eda["warnings"], title="Reading this variable")


def _binary_indicators(table):
    st.plotly_chart(charts.binary_comparison_chart(table, LABELS))
    shown = table.assign(feature=table["feature"].map(LABELS)).sort_values("difference_percentage_points",
                                                                           ascending=False)
    st.dataframe(shown.rename(columns={
        "feature": "Indicator", "no_respondents": "Code 0 respondents", "yes_respondents": "Code 1 respondents",
        "category_1_percent_no": "Category 1 share, code 0 (%)", "category_1_percent_yes": "Category 1 share, code 1 (%)",
        "difference_percentage_points": "Difference (points)"}), hide_index=True,
        column_config={"Code 0 respondents": COUNT, "Code 1 respondents": COUNT,
                       **{c: st.column_config.NumberColumn(format="%.1f") for c in
                          ["Category 1 share, code 0 (%)", "Category 1 share, code 1 (%)", "Difference (points)"]}})
    st.caption("Unadjusted differences: each indicator is compared on its own, without accounting for the others. "
               "The Regression Lab adjusts for all features at once.")


def _correlations(matrix):
    st.plotly_chart(charts.correlation_heatmap(matrix, LABELS))
    st.caption("Spearman correlation suits counts, BMI and ordered categories. Age, Education and Income are "
               "ordered codes, not exact quantities, and higher General health codes mean worse self-rated "
               "health (1 = Excellent, 5 = Poor). Sex is left out because its codes have no order. "
               "Correlations are descriptive, not a ranking of importance.")


def _sensitivity(table):
    overall = table[table["feature"] == TARGET].iloc[0]
    metric_row({"Category 1 share, full (%)": overall.estimate_full,
                "Category 1 share, unique (%)": overall.estimate_unique,
                "Change (points)": overall.change_unique_minus_full}, decimals=2)
    shown = table.assign(feature=table["feature"].map(LABELS), metric=table["metric"].map(ESTIMATES),
                         order=table["change_unique_minus_full"].abs()).sort_values("order", ascending=False)
    st.dataframe(shown.drop(columns="order").rename(columns={
        "feature": "Variable", "group": "Group", "metric": "Estimate", "estimate_full": "full_clean_v1",
        "n_full": "n (full)", "estimate_unique": "unique_profile_v1", "n_unique": "n (unique)",
        "change_unique_minus_full": "Change"}), hide_index=True,
        column_config={"n (full)": COUNT, "n (unique)": COUNT,
                       **{c: st.column_config.NumberColumn(format="%.2f")
                          for c in ["full_clean_v1", "unique_profile_v1", "Change"]}})
    st.caption("Compares both variants regardless of the selection above, largest changes first. Removing repeated "
               "rows changes which respondents are represented; it does not show which rows were true duplicates.")
