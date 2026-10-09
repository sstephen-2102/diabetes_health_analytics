"""Presentation components: results."""
import streamlit as st

from src.common.exceptions import DataLoadError


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


def show_warnings(warnings: list[str], title: str = "Methodology notes") -> None:
    if warnings:
        st.info(f"**{title}**\n\n" + "\n".join(f"- {w}" for w in warnings))