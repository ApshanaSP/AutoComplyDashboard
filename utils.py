import pandas as pd
import numpy as np
import google.generativeai as genai

# --- Gemini Configuration ---
def configure_gemini(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-pro")

# --- Metadata Extraction ---
def get_metadata(df):
    metadata = []
    for col in df.columns:
        col_data = df[col]
        dtype = str(col_data.dtype)
        missing = col_data.isnull().sum()
        unique = col_data.nunique()
        sample = col_data.dropna().astype(str).unique()[:5].tolist()

        metadata.append({
            "column": col,
            "dtype": dtype,
            "missing_values": missing,
            "unique_values": unique,
            "sample_values": sample
        })
    return metadata

# --- Data Quality Score & Manual Alerts ---
def calculate_data_quality(df):
    issues = []
    total_score = 100

    if len(df) < 50:
        issues.append({
            "Column": "Dataset",
            "Data Type": "N/A",
            "Alerts": "Too few rows (< 50) for reliable analysis"
        })
        total_score -= 15

    for col in df.columns:
        col_data = df[col]
        alerts = []

        if col_data.isnull().sum() > 0:
            alerts.append(f"{col_data.isnull().sum()} missing values")
            total_score -= min(10, (col_data.isnull().sum() / len(col_data)) * 100)

        if col_data.nunique() == 1:
            alerts.append("Only one unique value")
            total_score -= 5

        if pd.api.types.is_numeric_dtype(col_data):
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            outliers = ((col_data < (q1 - 1.5 * iqr)) | (col_data > (q3 + 1.5 * iqr))).sum()
            if outliers > 0:
                alerts.append(f"{outliers} possible outliers detected")
                total_score -= min(5, (outliers / len(col_data)) * 100)

        issues.append({
            "Column": col,
            "Data Type": str(col_data.dtype),
            "Alerts": ", ".join(alerts) if alerts else "No issues"
        })

    total_score = max(0, round(total_score, 2))
    return total_score, pd.DataFrame(issues)

# --- AI Insights from Gemini ---
def get_gemini_insights(model, df, metadata):
    sample = df.head(200).to_dict(orient="records")
    prompt = """You are a helpful data assistant.
Given:
- Metadata of a dataset
- 100–200 sample rows

Write a clean, simple summary for general users. Include:
1. Overview
2. Column Details (numbered)
3. Patterns & Trends
4. Suggested Charts (2)
5. Conclusion
6. Analyst View
7. Data Quality issues
8. ML/Stats Suggestions

Use clear spacing. Avoid markdown/symbols."""
    response = model.generate_content([prompt, f"METADATA: {metadata}\n\nSAMPLE_ROWS: {sample}"])
    return response.text, []

# --- AI Cleaning Suggestion ---
def get_cleaning_suggestion(model, df, column):
    prompt = f"""You are a data expert.
Suggest a specific cleaning fix for column '{column}' based on data type and sample.
Be brief (1-2 sentences). Use context-aware logic (e.g., fill numeric with mean, categorical with mode)."""
    sample = df[[column]].dropna().sample(min(10, len(df)), random_state=1).to_dict(orient="records")
    response = model.generate_content([prompt, f"SAMPLE: {sample}"])
    return response.text.strip()

# --- AI Statistical Analysis ---
def get_statistical_analysis_code(model, df):
    metadata = get_metadata(df)
    prompt = f"""You are a data scientist.
Suggest the best statistical test(s) based on this dataset. Include:
- Test name and purpose
- When to use
- Python and R code examples
- How to interpret results (like p-value)
Use clear spacing. Avoid markdown/symbols.
DATA METADATA:
{metadata}"""
    response = model.generate_content(prompt)
    return response.text.strip()

# --- Fallback Utility ---
def get_most_frequent_value(series):
    return series.mode().iloc[0] if not series.mode().empty else None
