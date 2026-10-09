"""Main Streamlit application entry point.

Launch with `streamlit run streamlit_app.py` from the repository root. Running this file directly breaks
imports: Streamlit would put app/ on sys.path (so `app` means this file) and auto-register app/pages/*.py.
"""
import streamlit as st

from app.pages import (calibration, dashboard, data_explorer, exploratory_analysis, imbalance_threshold,
                       interpretation, machine_learning, prediction, probability_explorer, regression_lab,
                       statistical_testing)

PAGES = {
    "Data": [(dashboard, "Executive Dashboard"), (data_explorer, "Data Explorer"),
             (exploratory_analysis, "Exploratory Analysis")],
    "Statistics": [(probability_explorer, "Probability Explorer"), (statistical_testing, "Statistical Testing Lab"),
                   (regression_lab, "Regression Lab")],
    "Modeling": [(machine_learning, "Machine Learning Lab"), (imbalance_threshold, "Imbalance & Threshold Lab"),
                 (calibration, "Calibration"), (interpretation, "Interpretation"),
                 (prediction, "Prediction Explorer")],
}


def main():
    """Run the Streamlit application."""
    st.set_page_config(page_title="Diabetes Health Analytics & Prediction Lab", layout="wide")
    navigation = {section: [st.Page(module.render, title=title, url_path=module.__name__.rsplit(".", 1)[-1],
                                    default=module is dashboard)
                            for module, title in pages]
                  for section, pages in PAGES.items()}
    st.navigation(navigation).run()
