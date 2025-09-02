import pandas as pd
import numpy as np
import google.generativeai as genai
import json
import string
import re

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
            "missing_values": int(missing),
            "unique_values": int(unique),
            "sample_values": sample
        })
    return metadata

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

        missing_count = col_data.isnull().sum()
        if missing_count > 0:
            alerts.append(f"{missing_count} missing values")
            total_score -= min(10, (missing_count / len(col_data)) * 20)

        if col_data.nunique() == 1 and len(col_data) > 1:
            alerts.append("Only one unique value")
            total_score -= 5

        if pd.api.types.is_numeric_dtype(col_data) and not pd.api.types.is_bool_dtype(col_data):
            col_data_numeric = col_data.dropna()
            if not col_data_numeric.empty:
                q1 = col_data_numeric.quantile(0.25)
                q3 = col_data_numeric.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outliers = ((col_data_numeric < lower_bound) | (col_data_numeric > upper_bound)).sum()
                    if outliers > 0:
                        alerts.append(f"{outliers} possible outliers detected")
                        total_score -= min(5, (outliers / len(col_data_numeric)) * 100)
        
        issues.append({
            "Column": col,
            "Data Type": str(col_data.dtype),
            "Alerts": ", ".join(alerts) if alerts else "No issues"
        })

    issues_df = pd.DataFrame(issues)
    total_score = max(0, round(total_score, 2))
    return total_score, issues_df


def get_gemini_insights(model, df, metadata):
    sample = df.head(100).to_string()
    prompt = f"""You are a world-class data analyst. Your task is to analyze the provided dataset metadata and sample to generate a concise report and suggest interactive filters.

**TASK 1: Generate a Data Analysis Report**
The report must be plain text and include these numbered sections:
1.  **Overview**: A brief summary of what the dataset appears to be about.
2.  **Key Column Insights**: A numbered list describing the most important columns and their potential significance.
3.  **Initial Patterns & Trends**: Mention 2-3 potential patterns, correlations, or trends you observe from the sample data.
4.  **Data Quality Summary**: Briefly comment on missing values, potential outliers, or other quality issues based on the metadata.
5.  **Actionable Suggestions**: Propose two business or research questions that could be answered using this data.

**TASK 2: Suggest Interactive Filters**
Based on the metadata, suggest up to 4 columns that would be most useful for filtering the data. Choose 'multiselect' for categorical columns with a reasonable number of unique values (<50) and 'slider' for numeric columns. Format the suggestions as a clean JSON array string. Place this JSON array after the separator token '---FILTERS---'.

**Example JSON Output:**
---FILTERS---
[
    {{"column": "Country", "type": "multiselect"}},
    {{"column": "Sales", "type": "slider"}}
]

**METADATA:**
{json.dumps(metadata, indent=2)}

**DATA SAMPLE (first 100 rows):**
{sample}
"""
    
    response = model.generate_content(prompt)
    report_text = response.text
    filters = []

    try:
        if "---FILTERS---" in report_text:
            parts = report_text.split("---FILTERS---")
            report_text = parts[0].strip()
            json_part = parts[1].strip()
            json_start = json_part.find('[')
            json_end = json_part.rfind(']') + 1
            if json_start != -1 and json_end > json_start:
                clean_json_str = json_part[json_start:json_end]
                filters = json.loads(clean_json_str)
    except (json.JSONDecodeError, IndexError) as e:
        print(f"Error parsing filters from Gemini response: {e}")
        filters = []

    return report_text, filters

def get_cleaning_suggestion(model, df, column):
    """Generates a data cleaning suggestion for a specific column."""
    metadata = get_metadata(df[[column]])
    sample = df[[column]].dropna().sample(min(10, len(df.dropna())), random_state=1).to_string()
    prompt = f"""You are a data cleaning expert. For the column '{column}' with the following properties, suggest a specific and concise cleaning strategy.

**Metadata:**
{metadata}

**Data Sample:**
{sample}

**Suggestion (be brief, 1-2 sentences):**"""
    
    response = model.generate_content(prompt)
    return response.text.strip()
    
def get_statistical_analysis_code(model, df):
    """Generates statistical test suggestions and code."""
    metadata = get_metadata(df)
    prompt = f"""As a data scientist, analyze the following dataset metadata and suggest the most appropriate statistical test(s). For each test, provide:
- The name of the test and its purpose (e.g., "T-test for comparing means").
- The type of variables required.
- A simple Python code example using `scipy.stats`.
- A brief explanation of how to interpret the p-value.

**DATA METADATA:**
{json.dumps(metadata, indent=2)}"""
    
    response = model.generate_content(prompt)
    return response.text.strip()

def get_excel_location(col_name, row_index, df):
    """Converts a column name and row index to an Excel-style cell location (e.g., A1, B2)."""
    try:
        col_index = df.columns.get_loc(col_name)
        col_letter = ""
        n = col_index + 1
        while n > 0:
            n, remainder = divmod(n - 1, 26)
            col_letter = string.ascii_uppercase[remainder] + col_letter
        return f"{col_letter}{row_index + 2}"
    except KeyError:
        return "N/A"

def find_duplicate_records(df):
    """Finds entire rows that are duplicates."""
    duplicates_mask = df.duplicated(keep=False)
    duplicate_rows = df[duplicates_mask]
    issues = []
    if not duplicate_rows.empty:
        for idx in duplicate_rows.index:
            issues.append({
                "row_index": idx,
                "column": "All Columns",
                "issue": "Entire row is a duplicate",
                "value": "See row for details",
                "Check_Type": "Duplicate Records",
                "Excel_Cell": f"Row {idx + 2}"
            })
    return pd.DataFrame(issues)

def find_inconsistent_casing(df):
    """Finds inconsistent casing for categorical string columns."""
    issues = []
    for col in df.select_dtypes(include='object').columns:
        series = df[col].dropna()
        if not series.empty:
            normalized = series.str.lower()
            if normalized.nunique() < series.nunique():
                unique_lower = normalized.unique()
                for val in unique_lower:
                    original_cases = series[normalized == val].unique()
                    if len(original_cases) > 1:
                        issues.append({
                            "column": col,
                            "issue": "Inconsistent Casing",
                            "value": f"Multiple casings for: {val}",
                            "Check_Type": "Inconsistent Casing"
                        })
    return pd.DataFrame(issues)


def find_whitespace_issues(df):
    """Finds leading/trailing whitespace issues in string columns."""
    issues = []
    for col in df.select_dtypes(include='object').columns:
        series = df[col].dropna()
        if not series.empty:
            has_leading_trailing_space = series.str.strip() != series
            if has_leading_trailing_space.any():
                for idx in series[has_leading_trailing_space].index:
                    issues.append({
                        "row_index": idx,
                        "column": col,
                        "issue": "Leading/Trailing Spaces",
                        "value": df.at[idx, col],
                        "Check_Type": "Leading/Trailing Spaces",
                        "Excel_Cell": get_excel_location(col, idx, df)
                    })
    return pd.DataFrame(issues)


def check_date_consistency(df):
    """Identifies columns with inconsistent date formats."""
    issues = []
    for col in df.columns:
        # Check for potential date columns
        if 'date' in col.lower() or df[col].dtype == 'object':
            series = df[col].dropna()
            if not series.empty:
                # Attempt to convert to datetime with different formats
                try:
                    # 'mixed' format will try to parse different formats
                    pd.to_datetime(series, errors='raise', dayfirst=True)
                except Exception:
                    issues.append({
                        "column": col,
                        "issue": "Inconsistent Date Formats",
                        "value": "Mixed date formats detected",
                        "Check_Type": "Date Format Inconsistencies"
                    })
    return pd.DataFrame(issues)


def find_data_issues(df):
    issues = []
    id_or_name_cols = [col for col in df.columns if any(k in col.lower() for k in ["id", "name"])]
    
    # 1. Missing values and outliers
    for col in df.columns:
        col_data = df[col]
        # Missing values
        if col_data.isnull().any():
            for idx in col_data[col_data.isnull()].index:
                issue = {
                    "row_index": idx,
                    "column": col,
                    "issue": "Missing value",
                    "value": None,
                    "Check_Type": "Missing Values",
                    "Excel_Cell": get_excel_location(col, idx, df)
                }
                for ref_col in id_or_name_cols:
                    if ref_col in df.columns:
                        issue[ref_col] = df.at[idx, ref_col]
                issues.append(issue)

        # Outliers
        if pd.api.types.is_numeric_dtype(col_data) and not pd.api.types.is_bool_dtype(col_data):
            # Drop NaN values for accurate quantile calculation
            col_data_numeric = col_data.dropna()

            # Proceed only if there's data to analyze
            if not col_data_numeric.empty:
                q1 = col_data_numeric.quantile(0.25)
                q3 = col_data_numeric.quantile(0.75)
                iqr = q3 - q1

                # Avoid division by zero and perform the outlier check
                if iqr > 0:
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    # Identify rows where the value is an outlier
                    outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
                    outlier_rows = df[outlier_mask]

                    for idx, row in outlier_rows.iterrows():
                        issue = {
                            "row_index": idx,
                            "column": col,
                            "issue": "Outlier",
                            "value": row[col],
                            "Check_Type": "Outliers",
                            # Assuming get_excel_location is defined elsewhere
                            "Excel_Cell": get_excel_location(col, idx, df)
                        }

                        # Add ID/Name columns for better context
                        for ref_col in id_or_name_cols:
                            if ref_col in df.columns:
                                issue[ref_col] = row[ref_col]
                        
                        issues.append(issue)
    
    issues_df = pd.DataFrame(issues)
    
    # 2. Duplicate Records
    duplicate_issues_df = find_duplicate_records(df)
    issues_df = pd.concat([issues_df, duplicate_issues_df], ignore_index=True)

    # 3. Inconsistent Casing
    casing_issues_df = find_inconsistent_casing(df)
    issues_df = pd.concat([issues_df, casing_issues_df], ignore_index=True)
    
    # 4. Leading/Trailing Spaces
    whitespace_issues_df = find_whitespace_issues(df)
    issues_df = pd.concat([issues_df, whitespace_issues_df], ignore_index=True)

    # 5. Date Format Inconsistencies
    date_issues_df = check_date_consistency(df)
    issues_df = pd.concat([issues_df, date_issues_df], ignore_index=True)
    
    # Add other columns if they are not already present
    for ref_col in id_or_name_cols:
        if ref_col not in issues_df.columns:
            issues_df[ref_col] = None

    return issues_df
