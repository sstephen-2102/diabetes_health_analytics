"""Shared validation helpers; no cleaning or UI dependencies."""
import numpy as np
import pandas as pd
from src.common.exceptions import DataValidationError, MissingColumnError

def require_frame(data, columns=(), allow_empty=False):
    if not isinstance(data,pd.DataFrame):
        raise DataValidationError('Expected a pandas DataFrame.')
    if not data.columns.is_unique:
        raise DataValidationError('Duplicate column names are not supported.')
    if not allow_empty and data.empty:
        raise DataValidationError('Dataset is empty.')
    missing = [c for c in columns if c not in data.columns]
    if missing:
        raise MissingColumnError(f'Missing columns: {missing}')

def require_numeric(data, columns):
    require_frame(data,columns)
    for column in columns:
        values = data[column]
        if not pd.api.types.is_numeric_dtype(values):
            raise DataValidationError(f'{column} must be numeric.')
        if not np.isfinite(values.dropna().to_numpy(dtype=float)).all():
            raise DataValidationError(f'{column} contains infinite values.')

def require_binary_target(data, target_column):
    require_frame(data,[target_column])
    if data[target_column].isna().any() or not data[target_column].isin([0,1]).all():
        raise DataValidationError('Target must contain only non-missing codes 0 and 1.')

def attach_metadata(result, data):
    result.attrs.update(data.attrs)
    result.attrs.setdefault('dataset_variant','unspecified')
    return result
