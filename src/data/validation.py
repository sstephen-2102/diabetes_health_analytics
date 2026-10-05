"""Non-destructive validation reports."""
import numpy as np
import pandas as pd
from src.common.config import VARIABLE_METADATA, EXPECTED_COLUMNS, REFERENCE_DATASET
from src.common.exceptions import InvalidParameterError
from src.data._checks import require_frame

def validate_variable(data: pd.DataFrame, column: str, variable_metadata: dict) -> dict:
    """Return column/is_valid/errors/missing_count/invalid_count/dtype.

    Accept one variable's metadata or the full dictionary. Missingness, text,
    infinity, invalid codes and range violations are errors. Positive BMI extremes
    pass structural validation but remain candidates for review. No coercion.
    """
    require_frame(data,[column],allow_empty=True)
    if not isinstance(variable_metadata,dict):
        raise InvalidParameterError('Metadata must be a dictionary.')
    meta = variable_metadata.get(column,variable_metadata)
    if 'type' not in meta:
        raise InvalidParameterError(f'No type metadata for {column}.')
    values = data[column]
    missing = int(values.isna().sum())
    invalid = pd.Series(False,index=values.index)
    errors = [f'{missing} missing values'] if missing else []
    if not pd.api.types.is_numeric_dtype(values):
        errors.append('Expected numeric dtype')
        invalid = values.notna()
    else:
        invalid |= values.notna() & ~np.isfinite(values)
        if meta.get('allowed_values') is not None:
            invalid |= values.notna() & ~values.isin(meta['allowed_values'])
        if meta.get('integer_only'):
            invalid |= values.notna() & values.mod(1).ne(0)
        if meta.get('minimum') is not None:
            invalid |= values.lt(meta['minimum'])
            if column == 'BMI':
                invalid |= values.eq(0)
        if meta.get('maximum') is not None:
            invalid |= values.gt(meta['maximum'])
    count = int(invalid.sum())
    if count:
        errors.append(f'{count} invalid values')
    return {'column':column,'is_valid':not errors,'errors':errors,'missing_count':missing,'invalid_count':count,'dtype':str(values.dtype)}

def validate_dataset(data: pd.DataFrame, required_columns: list[str], target_column: str) -> dict:
    """Return is_valid/errors/warnings, column checks and separate reference checks.

    required_columns defines the exact schema and includes target. Schema errors
    appear in the report; non-DataFrames/duplicate column names raise. A single
    target class warns. Reference drift is separate from structural validity, so
    small subgroups/fixtures aren't rejected for differing from full-data size.
    """
    require_frame(data,allow_empty=True)
    if not required_columns or len(set(required_columns)) != len(required_columns):
        raise InvalidParameterError('Required columns must be a non-empty unique list.')
    if target_column not in required_columns:
        raise InvalidParameterError('Include target in required_columns.')
    errors, warnings = [], []
    missing = [c for c in required_columns if c not in data]
    extra = [c for c in data if c not in required_columns]
    if data.empty: errors.append('Dataset is empty')
    if missing: errors.append(f'Missing columns: {missing}')
    if extra: errors.append(f'Unexpected columns: {extra}')
    checks = {}
    for column in required_columns:
        if column not in data: continue
        meta = VARIABLE_METADATA.get(column,{'type':'numeric'})
        if column == target_column: meta = {**meta,'allowed_values':[0,1]}
        checks[column] = validate_variable(data,column,meta)
        errors.extend(f'{column}: {e}' for e in checks[column]['errors'])
    duplicates = int(data.duplicated().sum())
    if duplicates: warnings.append('Repeated profiles retained; identical answers do not establish duplicate respondents.')
    if target_column in data and data[target_column].nunique() == 1: warnings.append('Single-class target; some analyses require both classes.')
    reference = {}
    if set(required_columns) == set(EXPECTED_COLUMNS):
        counts = data[target_column].value_counts().to_dict() if target_column in data else {}
        reference = {'rows':len(data)==REFERENCE_DATASET['rows'],'columns':set(data)==set(EXPECTED_COLUMNS),'missing_cells':int(data.isna().sum().sum())==0,'duplicate_rows':duplicates==REFERENCE_DATASET['duplicate_rows'],'target_counts':counts==REFERENCE_DATASET['target_counts']}
        if not all(reference.values()): warnings.append('Dataset differs from full-data reference; inspect provenance or variant.')
    return {'is_valid':not errors,'errors':errors,'warnings':warnings,'variables':checks,'duplicate_rows':duplicates,'dataset_variant':data.attrs.get('dataset_variant','unspecified'),'reference_checks':reference,'matches_reference':all(reference.values()) if reference else None}
