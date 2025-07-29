import pandas as pd

def extract_metadata(df: pd.DataFrame) -> dict:
    metadata = {}
    for col in df.columns:
        metadata[col] = {
            "dtype": str(df[col].dtype),
            "missing_values": int(df[col].isna().sum()),
            "unique_values": int(df[col].nunique()),
            "sample_values": df[col].dropna().unique()[:5].tolist()
        }
    return metadata
