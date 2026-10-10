"""Association measures and effect sizes (Member 2 effect sizes; Member 1 correlation matrix)."""
import numpy as np
import pandas as pd
from scipy import stats

from src.common.exceptions import InvalidParameterError


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


def cramers_v_from_chi2(chi2: float, n: int, n_rows: int, n_cols: int) -> float:
    """Cramer's V = sqrt(chi2 / (n * (min(rows, cols) - 1)))."""
    smaller_side = min(n_rows, n_cols) - 1
    if n <= 0 or smaller_side < 1:
        raise InvalidParameterError("Cramer's V needs n > 0 and a table of at least 2 x 2.")
    return float(np.sqrt(chi2 / (n * smaller_side)))


def _two_group_values(data: pd.DataFrame, numeric_feature: str, target_column: str, group_a, group_b):
    """Return the numeric values for each of two target groups, missing values dropped."""
    from src.data._checks import require_frame
    require_frame(data, [numeric_feature, target_column])
    values = []
    for group in (group_a, group_b):
        series = pd.to_numeric(data.loc[data[target_column] == group, numeric_feature], errors="coerce").dropna()
        if len(series) < 2:
            raise InvalidParameterError(f"Group {group!r} needs at least 2 numeric values.")
        values.append(series.to_numpy(dtype=float))
    return values


def _crosstab(data: pd.DataFrame, feature: str, target_column: str) -> pd.DataFrame:
    from src.data._checks import require_frame
    require_frame(data, [feature, target_column])
    table = pd.crosstab(data[feature], data[target_column])
    if table.shape[0] < 2 or table.shape[1] < 2:
        raise InvalidParameterError("Both variables need at least 2 distinct values.")
    return table


def calculate_effect_size(analysis_type: str, data: pd.DataFrame, **kwargs) -> dict:
    """Return appropriate effect-size measure; p-values are not effect sizes.

    analysis_type="cohens_d":    kwargs numeric_feature, target_column, group_a, group_b
                                 (group_a minus group_b, pooled standard deviation)
    analysis_type="cramers_v":   kwargs feature, target_column
    analysis_type="odds_ratio":  kwargs feature, target_column (both must be 2-valued);
                                 odds of the higher target value in the higher feature
                                 value vs the lower, with a 95% Woolf interval
    """
    if analysis_type == "cohens_d":
        a, b = _two_group_values(data, kwargs["numeric_feature"], kwargs["target_column"],
                                 kwargs["group_a"], kwargs["group_b"])
        pooled_var = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
        if pooled_var == 0:
            raise InvalidParameterError("Both groups are constant; Cohen's d is undefined.")
        return {"measure": "cohens_d", "value": float((a.mean() - b.mean()) / np.sqrt(pooled_var)),
                "n_a": len(a), "n_b": len(b), "direction": "group_a minus group_b",
                "dataset_variant": data.attrs.get("dataset_variant", "unspecified")}

    if analysis_type == "cramers_v":
        table = _crosstab(data, kwargs["feature"], kwargs["target_column"])
        chi2 = stats.chi2_contingency(table, correction=False)[0]
        return {"measure": "cramers_v",
                "value": cramers_v_from_chi2(chi2, int(table.to_numpy().sum()), *table.shape),
                "n": int(table.to_numpy().sum()),
                "dataset_variant": data.attrs.get("dataset_variant", "unspecified")}

    if analysis_type == "odds_ratio":
        table = _crosstab(data, kwargs["feature"], kwargs["target_column"])
        if table.shape != (2, 2):
            raise InvalidParameterError("Odds ratio needs a 2 x 2 table.")
        cells = table.to_numpy(dtype=float)
        warning = None
        if (cells == 0).any():
            cells = cells + 0.5
            warning = "A cell was zero; 0.5 added to every cell (Haldane-Anscombe)."
        odds_ratio = (cells[1, 1] * cells[0, 0]) / (cells[1, 0] * cells[0, 1])
        se_log = float(np.sqrt((1 / cells).sum()))
        z = stats.norm.ppf(0.975)
        return {"measure": "odds_ratio", "value": float(odds_ratio),
                "lower": float(np.exp(np.log(odds_ratio) - z * se_log)),
                "upper": float(np.exp(np.log(odds_ratio) + z * se_log)),
                "confidence_level": 0.95, "method": "Woolf (log) interval",
                "warning": warning, "n": int(table.to_numpy().sum()),
                "dataset_variant": data.attrs.get("dataset_variant", "unspecified")}

    raise InvalidParameterError("analysis_type must be 'cohens_d', 'cramers_v', or 'odds_ratio'.")
