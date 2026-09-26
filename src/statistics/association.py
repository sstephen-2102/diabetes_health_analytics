"""Function signature skeleton module.

Implementation intentionally omitted; use TODOs as the contract.
"""

import pandas as pd

def calculate_correlation_matrix(data: pd.DataFrame, columns: list[str], method: str = "spearman") -> pd.DataFrame:
    """Calculate correlation matrix with validation.

    TODO: validate inputs, preserve reproducibility, and return structured results.
    """
    pass


def calculate_effect_size(analysis_type: str, data: pd.DataFrame, **kwargs) -> dict:
    """Return appropriate effect-size measure; p-values are not effect sizes.

    TODO: validate inputs, preserve reproducibility, and return structured results.
    """
    pass


