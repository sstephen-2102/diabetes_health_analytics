"""Streamlit page: dashboard."""
import streamlit as st

from app import artifacts, services
from app.components import charts
from app.components.metrics import metric_row
from app.components.results import load_or_stop, show_context, show_warnings
from app.components.tables import model_label
from src.common.config import FEATURE_COLUMNS, PROJECT_CONFIG, TARGET_DEFINITION, VARIABLE_METADATA
from src.common.exceptions import DataLoadError

TARGET = PROJECT_CONFIG["target_column"]
RESEARCH_QUESTIONS = [
    ("RQ1", "What demographic, health, lifestyle, and socioeconomic characteristics describe the analytical "
            "sample?", "Data Explorer, Exploratory Analysis"),
    ("RQ2", "How is Diabetes_binary distributed?", "Executive Dashboard, Data Explorer"),
    ("RQ3", "How does the probability of the positive Diabetes_binary class change under selected conditions "
            "such as high blood pressure, high cholesterol, physical inactivity, and demographic categories?",
     "Probability Explorer, Exploratory Analysis"),
    ("RQ4", "Can Bayes theorem be demonstrated and independently verified from the observed data?",
     "Probability Explorer"),
    ("RQ5", "What confidence intervals describe selected proportions, means, and group differences?",
     "Statistical Testing Lab"),
    ("RQ6", "Which variables show statistically detectable associations or differences with the target outcome?",
     "Statistical Testing Lab"),
    ("RQ7", "How large are the observed effects, and how should they be interpreted alongside p-values?",
     "Statistical Testing Lab"),
    ("RQ8", "Which variables remain associated with the target after simultaneous adjustment?", "Regression Lab"),
    ("RQ9", "Does the BMI–target relationship vary by age?", "Regression Lab"),
    ("RQ10", "How do Logistic Regression, Random Forest, and Gradient Boosting compare across multiple metrics?",
     "Machine Learning Lab"),
    ("RQ11", "How does class imbalance affect model behavior?", "Imbalance & Threshold Lab"),
    ("RQ12", "How do thresholds affect precision, recall, specificity, and F1, and are predicted probabilities "
             "reasonably calibrated?", "Imbalance & Threshold Lab, Calibration"),
    ("RQ13", "How does exact-duplicate handling affect train/test integrity and model evaluation?",
     "Machine Learning Lab, Exploratory Analysis"),
]


def render():
    """Render page controls and structured results."""
    st.title("Diabetes Health Analytics & Prediction Lab")
    st.markdown("An interactive research lab on the CDC BRFSS 2015 Diabetes Health Indicators dataset "
                "(UCI Machine Learning Repository, dataset 891): descriptive statistics, probability, statistical "
                "testing, regression and machine learning for one project-defined binary outcome.")
    data = load_or_stop(artifacts.eda_data)
    variants = data["results"]["variants"]
    summary = services.get_dashboard_summary(variants["full_clean_v1"]["data"], TARGET)
    quality = summary["quality"]
    show_context({"dataset_variant": summary["dataset_variant"], "target_definition": TARGET_DEFINITION},
                 f"Source: {data['metadata']['source_file']}")

    share = summary["target_distribution"].set_index("category").loc[1, "percentage"]
    metric_row({"Respondents": quality["rows"], "Features": len(FEATURE_COLUMNS), "Category 1 share (%)": share,
                "Distinct profiles": quality["distinct_patterns"]}, decimals=1)

    st.plotly_chart(charts.target_distribution_chart(summary["target_distribution"]))

    st.subheader("Selected findings")
    st.markdown("\n".join(f"- {line}" for line in _findings(variants, share)))
    st.caption("Unadjusted sample percentages from full_clean_v1 unless stated; see Exploratory Analysis for "
               "every variable.")

    st.subheader("Research questions")
    st.markdown("\n".join(f"- **{rq}** {question} *({pages})*" for rq, question, pages in RESEARCH_QUESTIONS))

    show_warnings(data["warnings"] + [
        "Diabetes_binary = 1 combines prediabetes and diabetes. It is a project-defined survey outcome, "
        "not a diagnosis.",
        "The Prediction Explorer shows model estimates from this dataset. It is not a medical diagnosis or advice.",
        f"{quality['duplicate_rows']:,} rows ({quality['duplicate_rate_percent']:.2f}%) repeat an earlier row. "
        "They are kept for descriptive work; machine learning uses unique_profile_v1 with a grouped split, so no "
        "profile is in both the training and test sets.",
        "Statistical significance is not practical importance: read effect sizes and confidence intervals "
        "alongside p-values.",
    ])


def _findings(variants: dict, share: float) -> list[str]:
    tables = variants["full_clean_v1"]["tables"]
    lines = [f"{share:.1f}% of respondents are in category 1 (prediabetes or diabetes), so the classes are "
             "imbalanced."]

    binary = tables["binary_comparisons"]
    largest = binary.loc[binary["difference_percentage_points"].abs().sort_values(ascending=False).index[:3]]
    for row in largest.itertuples():
        lines.append(f"{VARIABLE_METADATA[row.feature]['label']}: {row.category_1_percent_yes:.1f}% of those "
                     f"answering yes are in category 1, against {row.category_1_percent_no:.1f}% of those "
                     "answering no.")

    bmi = services.get_eda_summary(variants["full_clean_v1"]["data"], "BMI")["by_target"].set_index(TARGET)["mean"]
    lines.append(f"Mean BMI is {bmi.loc[1]:.1f} in category 1 and {bmi.loc[0]:.1f} in category 0.")

    health = tables["prevalence_GenHlth"]
    first, last = health.iloc[0], health.iloc[-1]
    lines.append(f"Category 1 share is {first.target_percentage:.1f}% for {first.label.lower()} general health "
                 f"and {last.target_percentage:.1f}% for {last.label.lower()}.")

    unique = variants["unique_profile_v1"]["tables"]["target_distribution"].set_index("category")
    lines.append(f"Removing repeated rows (unique_profile_v1) changes the category 1 share from {share:.1f}% to "
                 f"{unique.loc[1, 'percentage']:.1f}%.")

    try:
        comparison = artifacts.modeling_outputs()["results"]["comparison"]
    except DataLoadError:
        return lines
    best = comparison["roc_auc"].idxmax()
    lines.append(f"On the held-out test set, ROC AUC ranges from {comparison['roc_auc'].min():.3f} to "
                 f"{comparison['roc_auc'].max():.3f} ({model_label(best)} highest); see the Machine Learning Lab.")
    return lines
