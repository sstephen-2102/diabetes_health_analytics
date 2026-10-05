"""Explicit analysis variants; input frames stay unchanged."""
import pandas as pd
from src.common.exceptions import InvalidParameterError
from src.data._checks import require_frame, require_binary_target

def prepare_analysis_data(data: pd.DataFrame, target_column: str, feature_columns: list[str], drop_missing: bool = False, dataset_variant: str = 'full_clean_v1') -> pd.DataFrame:
    """Copy features and target, with optional explicit missing-row/duplicate removal.

    Existing four positional arguments remain compatible. unique_profile_v1
    deduplicates complete input rows BEFORE feature selection. balanced_train_v1
    is intentionally not created here. attrs record variant and removed counts.
    """
    if not isinstance(feature_columns,list) or not feature_columns:
        raise InvalidParameterError('feature_columns must be a non-empty list.')
    if len(set(feature_columns))!=len(feature_columns) or target_column in feature_columns:
        raise InvalidParameterError('Features must be unique and exclude target.')
    if not isinstance(drop_missing,bool): raise InvalidParameterError('drop_missing must be boolean.')
    if dataset_variant not in {'full_clean_v1','unique_profile_v1'}:
        raise InvalidParameterError('Use full_clean_v1 or unique_profile_v1; balancing is training-only.')
    columns = feature_columns + [target_column]
    require_frame(data,columns)
    working = data.dropna(subset=columns) if drop_missing else data
    require_binary_target(working,target_column)
    unique = working.drop_duplicates() if dataset_variant=='unique_profile_v1' else working
    result = unique[columns].copy()
    result.attrs.update(data.attrs)
    result.attrs.update(dataset_variant=dataset_variant,source_rows=len(data),dropped_missing_rows=len(data)-len(working),dropped_duplicate_rows=len(working)-len(unique))
    return result
