"""Cached access to exported modeling artifacts, so pages do not reload them on every rerun."""
from pathlib import Path

import streamlit as st

from app import services

OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"
RAW_CSV = Path(__file__).resolve().parents[1] / "data" / "raw" / "cdc_diabetes.csv"


@st.cache_resource(show_spinner="Loading saved models (the first launch after a fresh clone retrains them, "
                                "which takes a few minutes)...")
def model_artifacts() -> dict:
    """Fitted pipelines are shared, not copied: pages must not modify them."""
    services.ensure_model_files(str(OUTPUTS / "models"), str(RAW_CSV))
    return services.load_model_artifacts(str(OUTPUTS / "models"))


@st.cache_data(show_spinner="Loading modeling results...")
def modeling_outputs() -> dict:
    return services.load_modeling_outputs(str(OUTPUTS))


@st.cache_data(show_spinner="Loading regression results...")
def regression_outputs() -> dict:
    return services.load_regression_outputs(str(OUTPUTS))


@st.cache_data(show_spinner="Loading duplicate-sensitivity results...")
def duplicate_sensitivity() -> dict:
    return services.load_duplicate_sensitivity(str(OUTPUTS))