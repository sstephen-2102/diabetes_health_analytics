"""JSON-compatible quality reports; flags never modify data."""
import numpy as np
import pandas as pd
from src.common.config import PROJECT_CONFIG, TARGET_DEFINITION, VARIABLE_METADATA
from src.data._checks import require_frame

def generate_data_quality_report(data: pd.DataFrame) -> dict:
    """Report dimensions, missingness, duplicates, types, constants, BMI IQR flags.

    Only BMI uses IQR screening; category codes/bounded days aren't treated as
    continuous outliers. Empty frames are allowed and have zero duplicate rate.
    """
    require_frame(data,allow_empty=True)
    n = len(data)
    columns = {c:{'dtype':str(data[c].dtype),'missing_count':int(data[c].isna().sum()),'missing_percent':float(data[c].isna().mean()*100) if n else 0.0,'unique_values':int(data[c].nunique()),'analytical_type':VARIABLE_METADATA.get(c,{}).get('type','unknown')} for c in data}
    target = PROJECT_CONFIG['target_column']
    counts = {str(k):int(v) for k,v in data[target].value_counts(dropna=False).items()} if target in data else {}
    outliers = {}
    if 'BMI' in data and pd.api.types.is_numeric_dtype(data['BMI']):
        values = data['BMI'].dropna()
        values = values[np.isfinite(values)]
        if not values.empty:
            q1,q3 = values.quantile([.25,.75])
            low,high = q1-1.5*(q3-q1),q3+1.5*(q3-q1)
            lower,upper = int(values.lt(low).sum()),int(values.gt(high).sum())
            outliers['BMI'] = {'q1':float(q1),'q3':float(q3),'iqr':float(q3-q1),'lower_boundary':float(low),'upper_boundary':float(high),'below_count':lower,'above_count':upper,'flagged_count':lower+upper,'flagged_percent':(lower+upper)/len(values)*100,'decision':'Retained; flags do not establish errors.'}
    duplicates = int(data.duplicated().sum())
    return {'rows':n,'columns':len(data.columns),'missing_cells':int(data.isna().sum().sum()),'duplicate_rows':duplicates,'duplicate_rate_percent':duplicates/n*100 if n else 0.0,'rows_in_repeated_patterns':int(data.duplicated(keep=False).sum()),'distinct_patterns':n-duplicates,'column_summary':columns,'constant_columns':[c for c in data if data[c].nunique(dropna=False)<=1],'potential_outliers':outliers,'target_column':target,'target_distribution':counts,'target_definition':TARGET_DEFINITION,'dataset_variant':data.attrs.get('dataset_variant','unspecified')}
