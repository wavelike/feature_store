
import pandas as pd
import json
from pandas.api.types import is_datetime64_any_dtype as is_datetime

def calc_data_identifier(feature: pd.Series):
    # Dependent on the type of the provided Series object, statistical features are calculated, converted to string and usable as a data identifier

    precision = 9

    n_rows = len(feature)

    if pd.api.types.is_numeric_dtype(feature.dtype):
        mean = round(float(feature.mean()), precision)
        max = round(float(feature.max()), precision)
        min = round(float(feature.min()), precision)
        std = round(float(feature.std()), precision)

        data_analysis = {
            'n_rows': n_rows,
            'mean': mean,
            'max': max,
            'min': min,
            'std': std,
        }

    elif pd.api.types.is_string_dtype(feature.dtype):

        value_counts = feature.value_counts().to_dict()

        data_analysis = {
            'n_rows': n_rows,
            'value_counts': value_counts,
        }

    elif is_datetime(feature.dtype):
        # convert datetime values to string to make them json exportable
        value_counts = feature.value_counts()
        value_counts.index = value_counts.index.astype(str)
        value_counts = value_counts.to_dict()

        data_analysis = {
            'n_rows': n_rows,
            'value_counts': value_counts,
        }

    else:
        raise (Exception(f"Unexpected feature dtype: {feature.dtype}"))

    data_identifier_string = json.dumps(data_analysis, sort_keys=True)

    return data_identifier_string

