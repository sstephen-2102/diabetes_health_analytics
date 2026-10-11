"""Presentation components: results."""
import streamlit as st

from src.common.config import VARIABLE_METADATA
from src.common.exceptions import DataLoadError

VARIANT_LABELS = {"full_clean_v1": "full_clean_v1: all rows (descriptive analyses)",
                  "unique_profile_v1": "unique_profile_v1: repeated rows removed (machine learning)"}


def load_or_stop(loader):
    """Call a cached loader; if exports are missing, show how to create them and stop the page."""
    try:
        return loader()
    except DataLoadError as exc:
        st.error(str(exc))
        st.stop()


def show_context(result: dict, extra: str = "") -> None:
    """Caption naming the dataset variant and target definition, as every page must."""
    st.caption(f"Dataset variant: `{result['dataset_variant']}` · Target: {result['target_definition']}"
               + (f" · {extra}" if extra else ""))


def variable_label(column: str) -> str:
    """Readable label with the column name, e.g. 'High blood pressure (HighBP)'; 'Education' stays as is."""
    label = VARIABLE_METADATA[column]["label"]
    return label if label == column else f"{label} ({column})"


def select_variant(variants: dict) -> str:
    """Radio for the dataset variant; full_clean_v1, the descriptive variant, comes first."""
    return st.radio("Dataset variant", list(variants), format_func=lambda name: VARIANT_LABELS.get(name, name),
                    horizontal=True)


def show_warnings(warnings: list[str], title: str = "Methodology notes") -> None:
    if warnings:
        st.info(f"**{title}**\n\n" + "\n".join(f"- {w}" for w in warnings))