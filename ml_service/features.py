import pandas as pd
from ml_service.schemas import PredictRequest


FEATURE_COLUMNS = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education.num',
    'marital.status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital.gain',
    'capital.loss',
    'hours.per.week',
    'native.country',
]


def to_dataframe(req: PredictRequest, needed_columns: list[str] = None) -> pd.DataFrame:
    if not any(req.dict().values()):
        raise ValueError("All input features are empty")

    columns = [
        column for column in needed_columns if column in FEATURE_COLUMNS
    ] if needed_columns is not None else FEATURE_COLUMNS
    
    row = []
    missing_columns = []
    
    for column in columns:
        val = getattr(req, column.replace('.', '_'))
        if val is None:
            missing_columns.append(column)
        row.append(val)
        
    if missing_columns:
        raise ValueError(f"Missing required features for current model: {missing_columns}")
        
    return pd.DataFrame([row], columns=columns)
