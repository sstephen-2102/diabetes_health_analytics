"""Streamlit page: prediction."""
import streamlit as st

from app import artifacts, services
from app.components.results import load_or_stop, show_context, show_warnings
from app.components.tables import model_label
from src.analysis.profiles import CLASSIFICATION_LABEL, PROBABILITY_LABEL
from src.common.config import FEATURE_COLUMNS, VARIABLE_METADATA
from src.common.exceptions import DataValidationError, InvalidParameterError

FORM_INPUTS = [c for c in FEATURE_COLUMNS if VARIABLE_METADATA[c]["prediction_input"]]
BMI_RANGE = (12.0, 98.0)  # observed in the source data; the models have not seen values outside it


def _input_widget(column, feature: str, default):
    meta = VARIABLE_METADATA[feature]
    if meta["allowed_values"]:
        return column.selectbox(meta["label"], meta["allowed_values"],
                                index=meta["allowed_values"].index(default),
                                format_func=lambda code: meta["category_labels"][code], help=meta["description"])
    return column.number_input(f"{meta['label']} ({meta['unit']})", min_value=BMI_RANGE[0],
                               max_value=BMI_RANGE[1], value=float(default), step=0.5, help=meta["description"])


def render():
    """Render page controls and structured results."""
    st.title("Prediction Explorer")
    loaded = load_or_stop(artifacts.model_artifacts)
    metadata, test_metrics = loaded["metadata"], loaded["metrics"]["test_metrics"]
    defaults = metadata["default_profile"]

    left, right = st.columns(2)
    name = left.selectbox("Model", list(loaded["models"]), format_func=model_label)
    threshold = right.slider("Classification threshold", 0.05, 0.95, metadata["threshold"], 0.05)

    st.subheader("Feature profile")
    columns = st.columns(4)
    inputs = {feature: _input_widget(columns[i % 4], feature, defaults[feature])
              for i, feature in enumerate(FORM_INPUTS)}

    try:
        result = services.get_prediction_result(loaded["models"][name], inputs, threshold, metadata)
    except (DataValidationError, InvalidParameterError) as exc:
        st.error(str(exc))
        return
    prediction = result["results"]

    st.subheader("Model output")
    show_context(result)
    probability, classification = st.columns(2)
    probability.metric(PROBABILITY_LABEL, f"{prediction['probability']:.1%}")
    classification.metric(f"{CLASSIFICATION_LABEL} ({threshold:.2f})", prediction["classification"])
    st.write(prediction["classification_text"])
    st.warning(prediction["disclaimer"])

    with st.expander("Profile used for this estimate"):
        st.table({"Feature": list(prediction["profile"]), "Value": [str(v) for v in prediction["profile"].values()]})
    with st.expander("Model metadata"):
        model_test = test_metrics[name]
        st.write({**result["metadata"], "test ROC AUC": round(model_test["roc_auc"], 3),
                  "test recall at 0.5": round(model_test["recall"], 3),
                  "test precision at 0.5": round(model_test["precision"], 3)})

    typical = ", ".join(f"{VARIABLE_METADATA[f]['label']} = {VARIABLE_METADATA[f]['category_labels'][defaults[f]]}"
                        for f in ["Education", "Income"])
    show_warnings(result["warnings"][1:] + [
        "The probability reflects how often similar survey profiles were positive in the training data; "
        "it is not an individual risk assessment.",
        f"The most common training values include {typical}; these upper categories tend to lower the estimate.",
    ])