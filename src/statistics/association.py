"""Function signature skeleton module.

Implementation intentionally omitted; use TODOs as the contract.
"""

import pandas as pd

def calculate_correlation_matrix(data: pd.DataFrame, columns: list[str], method: str = "spearman") -> pd.DataFrame:
    """Return a labeled numeric correlation matrix, carrying dataset metadata.

    Spearman uses average ranks and Pearson correlation of ranks. Pearson is
    also supported. At least two valid pairs are required; constants yield NaN.
    Category coding and unadjusted relationships require cautious interpretation.
    """
    from src.data._checks import require_numeric, attach_metadata
    from src.common.exceptions import InvalidParameterError
    if not columns or len(set(columns)) != len(columns):
        raise InvalidParameterError("Supply a non-empty list of unique columns.")
    if method not in {"spearman", "pearson"}:
        raise InvalidParameterError("Use spearman or pearson.")
    require_numeric(data, columns)
    result = data[columns].corr(method=method, min_periods=2)
    attach_metadata(result, data)
    result.attrs["method"] = method
    return result


def calculate_effect_size(analysis_type: str, data: pd.DataFrame, **kwargs) -> dict:
    """Return appropriate effect-size measure; p-values are not effect sizes.

    TODO: validate inputs, preserve reproducibility, and return structured results.
    """
    pass


