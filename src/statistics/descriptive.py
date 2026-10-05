"""Reusable EDA summaries; numeric storage does not imply numeric meaning."""
import pandas as pd
from src.common.config import VARIABLE_METADATA, TARGET_DEFINITION
from src.common.exceptions import InvalidParameterError
from src.data._checks import require_frame, require_numeric, require_binary_target, attach_metadata

def summarize_numeric_variable(data: pd.DataFrame, column: str) -> dict:
    """Return n, missing_count, mean, median, std, variance, min, q1, q3, iqr, max.

    Exclude missing values; sample SD/variance use ddof=1. Undefined summaries are
    None. Known binary/ordinal codes are rejected to avoid averaging age codes.
    """
    require_numeric(data,[column])
    if VARIABLE_METADATA.get(column,{}).get('type') in {'binary','ordinal'}:
        raise InvalidParameterError('Use categorical summaries for category codes.')
    values = data[column].dropna()
    def clean(v): return None if pd.isna(v) else float(v)
    q1,q3 = values.quantile([.25,.75])
    return {'n':len(values),'missing_count':int(data[column].isna().sum()),'mean':clean(values.mean()),'median':clean(values.median()),'std':clean(values.std(ddof=1)),'variance':clean(values.var(ddof=1)),'min':clean(values.min()),'q1':clean(q1),'q3':clean(q3),'iqr':clean(q3-q1),'max':clean(values.max()),'dataset_variant':data.attrs.get('dataset_variant','unspecified')}

def summarize_categorical_variable(data: pd.DataFrame, column: str, include_percentages: bool = True) -> pd.DataFrame:
    """Return category/label/count/percentage; denominator includes missing rows.

    Missing values have a visible group. Absent documented codes get zero counts;
    unexpected observed codes remain visible. Percentages can be omitted.
    """
    require_frame(data,[column])
    if not isinstance(include_percentages,bool): raise InvalidParameterError('include_percentages must be boolean.')
    values = data[column]
    counts = values.value_counts(dropna=True).sort_index()
    meta = VARIABLE_METADATA.get(column,{})
    allowed = meta.get('allowed_values')
    if allowed is not None:
        counts = counts.reindex(list(allowed)+[v for v in counts.index if v not in allowed],fill_value=0)
    labels = meta.get('category_labels') or {}
    result = counts.rename_axis('category').reset_index(name='count')
    result['label'] = result['category'].map(lambda v:labels.get(v,str(v)))
    if values.isna().any(): result.loc[len(result)] = {'category':None,'count':int(values.isna().sum()),'label':'Missing'}
    if include_percentages: result['percentage'] = result['count']/len(data)*100
    return attach_metadata(result,data)

def compare_by_target(data: pd.DataFrame, feature: str, target_column: str) -> pd.DataFrame:
    """Return numeric describe-by-target or categorical counts within target.

    Categorical percentages use each target group's size as denominator.
    Return attrs propagate variant and target definition.
    """
    require_frame(data,[feature,target_column])
    require_binary_target(data,target_column)
    if feature==target_column: raise InvalidParameterError('Feature must differ from target.')
    if VARIABLE_METADATA.get(feature,{}).get('type') in {'binary','ordinal'} or not pd.api.types.is_numeric_dtype(data[feature]):
        tables = []
        for code,group in data.groupby(target_column,sort=True):
            table = summarize_categorical_variable(group,feature)
            table.insert(0,target_column,code)
            tables.append(table)
        result = pd.concat(tables,ignore_index=True)
    else:
        require_numeric(data,[feature])
        result = data.groupby(target_column)[feature].describe().reset_index()
    attach_metadata(result,data)
    result.attrs['target_definition'] = TARGET_DEFINITION
    return result

def calculate_prevalence_table(data: pd.DataFrame, feature: str, target_column: str) -> pd.DataFrame:
    """Return category/label/total_observations/target_count/target_percentage.

    target_count counts code 1; percentages use feature-category denominators.
    Absent categories have zero counts and undefined (NaN) percentages. Missing
    feature values form a visible group. Return attrs describe variant and target.
    """
    require_frame(data,[feature,target_column])
    require_binary_target(data,target_column)
    if feature==target_column: raise InvalidParameterError('Feature must differ from target.')
    result = data.groupby(feature,dropna=False)[target_column].agg(total_observations='size',target_count='sum')
    meta = VARIABLE_METADATA.get(feature,{})
    allowed = meta.get('allowed_values')
    if allowed is not None:
        result = result.reindex(list(allowed)+[v for v in result.index if v not in allowed],fill_value=0)
    result['target_percentage'] = result['target_count']/result['total_observations'].replace(0,float('nan'))*100
    result = result.rename_axis('category').reset_index()
    labels = meta.get('category_labels') or {}
    result.insert(1,'label',result['category'].map(lambda v:'Missing' if pd.isna(v) else labels.get(v,str(v))))
    attach_metadata(result,data)
    result.attrs['target_definition'] = TARGET_DEFINITION
    return result
