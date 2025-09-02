import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import tempfile
import os
from utils import (
    get_metadata,
    calculate_data_quality,
    get_gemini_insights,
    get_cleaning_suggestion,
    get_statistical_analysis_code,
    find_data_issues,
    get_excel_location,
)
from google.generativeai import configure, GenerativeModel

# --- PDF Generation Imports ---
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import Image as RLImage

# --- Page Configuration ---
st.set_page_config(
    page_title="AI-Powered Dynamic Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown(
    """
<style>
    .stMetric { border-radius: 10px; background-color: #262730; border: 1px solid #4A4A4A; padding: 20px; text-align: center; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); transition: 0.3s; }
    .stMetric:hover { box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2); }
    .stMetric .st-emotion-cache-1g8i8fg { font-size: 2.5rem; font-weight: bold; color: #00F2FE; }
    .stMetric .st-emotion-cache-1xarl3l { color: #A1A1A1; }
    .stProgress > div > div > div > div { background-image: linear-gradient(to right, #4facfe, #00f2fe); }
    .stButton>button { width: 100%; border-radius: 10px; border: 1px solid #4facfe; color: #FFFFFF; background-color: #4facfe; }
    .st-emotion-cache-1r6slb0 { background-color: #262730; }
</style>
""",
    unsafe_allow_html=True,
)

# --- Helpers for Data Quality Table ---
def _default_fix(alerts: str, dtype: str) -> str:
    a = (alerts or "").lower()
    if "missing" in a or "null" in a:
        return "Impute missing: numeric→mean/median; categorical→mode; dates→forward/backfill."
    if "duplicate" in a:
        return "Drop duplicates keeping first; if IDs, enforce uniqueness constraint."
    if "outlier" in a:
        return "Review outliers; cap via IQR/winsorize or remove if erroneous."
    if "type" in a or "dtype" in a or "inconsistent" in a:
        return f"Cast to {dtype}; coerce errors and log invalids."
    if "format" in a:
        return "Normalize format with regex/strptime; standardize casing/spacing."
    if "range" in a or "negative" in a:
        return "Apply domain rules; clip to valid range and flag violations."
    return "Review column; standardize values and re-run checks."

def prepare_quality_table(df: pd.DataFrame, quality_details: pd.DataFrame) -> pd.DataFrame:
    qd = quality_details.copy()
    if "Column" not in qd.columns:
        qd.insert(0, "Column", list(df.columns)[: len(qd)])
    if "DataType" not in qd.columns:
        dtype_map = df.dtypes.astype(str).to_dict()
        qd["DataType"] = qd["Column"].map(dtype_map).fillna("unknown")
    if "Alerts" not in qd.columns:
        qd["Alerts"] = "No issues"
    qd["Alerts"] = qd["Alerts"].fillna("No issues")
    suggest_col = None
    for c in qd.columns:
        if c.strip().lower() in {"suggested fix", "suggestion", "fix", "resolution"}:
            suggest_col = c
            break
    if suggest_col is None:
        qd["Suggested Fix"] = [
            _default_fix(alerts, dtype) for alerts, dtype in zip(qd["Alerts"].astype(str), qd["DataType"].astype(str))
        ]
    else:
        qd.rename(columns={suggest_col: "Suggested Fix"}, inplace=True)
        qd["Suggested Fix"] = qd["Suggested Fix"].fillna(
            [
                _default_fix(alerts, dtype)
                for alerts, dtype in zip(qd["Alerts"].astype(str), qd["DataType"].astype(str))
            ]
        )
    keep = ["Column", "DataType", "Alerts", "Suggested Fix"]
    for col in keep:
        if col not in qd.columns:
            qd[col] = ""
    return qd[keep]

# --- App State ---
for key in [
    "df", "df_raw", "report_text", "suggested_filters", "api_key_valid",
    "df_filtered", "uploaded_file_name", "issues_df"
]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "suggested_filters" else None

# --- Title ---
st.title(" AI-Powered Dynamic Dashboard")
st.markdown("Upload any structured CSV, select your columns, and let AI build a custom dashboard for you.")

# --- Sidebar ---
with st.sidebar:
    st.header(" Configuration & Filters")
    API_KEYS = ["AIzaSyCSBs37wgBO9Pi4i-91zIBP6K8ZBQb0QZg"]
    st.session_state.api_key_valid = False
    for i, key in enumerate(API_KEYS):
        if key == "YOUR_API_KEY_HERE":
            st.warning("Please replace 'YOUR_API_KEY_HERE' in the code.", icon="⚠️")
            continue
        try:
            configure(api_key=key)
            st.session_state.model = GenerativeModel("gemini-2.0-flash")
            st.session_state.api_key_valid = True
            st.success("Using valid API Key.", icon="✅")
            break
        except Exception:
            st.warning(f"API Key #{i+1} failed. Trying next...", icon="⚠️")
    if not st.session_state.api_key_valid:
        st.error("No valid API key found. Please update it in the code.", icon="❌")

    uploaded_file = st.file_uploader(
        "Choose a CSV file", type="csv", disabled=not st.session_state.api_key_valid
    )

    if uploaded_file and st.session_state.api_key_valid:
        if st.session_state.get("uploaded_file_name") != uploaded_file.name:
            for k in ["df", "df_raw", "report_text", "suggested_filters", "df_filtered", "issues_df"]:
                st.session_state[k] = None
            st.session_state.uploaded_file_name = uploaded_file.name

        if st.session_state.df is None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df_raw = df.copy()
                st.success("File uploaded successfully! Please select columns to proceed.")
                st.subheader(" Select Columns for Analysis")
                col1, col2 = st.columns([2, 1])
                with col1:
                    selected_columns = st.multiselect("Choose columns to keep for analysis:", options=df.columns.tolist())
                with col2:
                    keep_all = st.checkbox("Keep All Columns", value=False)

                if keep_all:
                    df_filtered = df.copy()
                elif selected_columns:
                    df_filtered = df[selected_columns].copy()
                else:
                    st.warning("Please select at least one column or check 'Keep All Columns'.")
                    st.stop()

                st.markdown("### Preview of Filtered Dataset")
                st.dataframe(df_filtered.head(), use_container_width=True)

                csv_download = df_filtered.to_csv(index=False).encode("utf-8")
                st.download_button(" Download Filtered CSV", data=csv_download, file_name="filtered_data.csv", mime="text/csv")

                if st.button(" Confirm & Proceed with Analysis"):
                    st.session_state.df = df_filtered
                    with st.spinner("Analyzing filtered data with AI..."):
                        metadata = get_metadata(df_filtered.copy())
                        report, filters = get_gemini_insights(st.session_state.model, df_filtered, metadata)
                        st.session_state.report_text = report
                        st.session_state.suggested_filters = filters
                        st.session_state.df_filtered = df_filtered
                else:
                    st.stop()

            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.session_state.df = None

if st.session_state.df is None:
    st.info("Please upload a CSV and confirm columns to analyze.")
    st.stop()

df = st.session_state.df
df_filtered = st.session_state.df_filtered if st.session_state.df_filtered is not None else df

# --- Data Quality ---
st.header(" Data Quality Assessment")
quality_score, quality_details = calculate_data_quality(df.copy())
quality_table_df = prepare_quality_table(df_filtered, quality_details) if isinstance(quality_details, pd.DataFrame) else pd.DataFrame()

with st.expander("Show Detailed Quality Report & Suggestions"):
    if not quality_table_df.empty:
        st.dataframe(quality_table_df, use_container_width=True)
    else:
        st.dataframe(quality_details, use_container_width=True)

# --- Data Cleaning Suggestions ---
st.markdown("---")
st.header(" AI-Powered Data Cleaning")
columns_with_issues = quality_table_df[quality_table_df["Alerts"] != "No issues"]["Column"].tolist() if not quality_table_df.empty else []

if columns_with_issues:
    selected_col = st.selectbox("Select a column with issues", columns_with_issues)
    if st.button(" Get Cleaning Suggestion"):
        with st.spinner("Getting AI suggestion..."):
            suggestion = get_cleaning_suggestion(st.session_state.model, df_filtered, selected_col)
            st.success(suggestion)

    if st.button(" Apply Suggestion (Auto Fill)"):
        if pd.api.types.is_numeric_dtype(df_filtered[selected_col]):
            mean_val = df_filtered[selected_col].mean()
            df_filtered[selected_col].fillna(mean_val, inplace=True)
            st.success(f"Missing values in '{selected_col}' filled with mean: {mean_val:.2f}")
        else:
            mode_val = df_filtered[selected_col].mode().iloc[0]
            df_filtered[selected_col].fillna(mode_val, inplace=True)
            st.success(f"Missing values in '{selected_col}' filled with mode: {mode_val}")
else:
    st.info("No issues found that require cleaning.")

# --- KPIs ---
st.markdown("---")
st.header(" Key Performance Indicators")
kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
kpi_col1.metric("Filtered Rows", f"{df_filtered.shape[0]:,}")
kpi_col2.metric("Total Columns", f"{df.shape[1]:,}")
total_missing = df_filtered.isnull().sum().sum()
total_cells = np.prod(df_filtered.shape)
missing_percent = (total_missing / total_cells) * 100 if total_cells > 0 else 0
kpi_col3.metric("% Missing Data (Filtered)", f"{missing_percent:.2f}%")

# --- Visualizations ---
st.markdown("---")
st.header(" Interactive Visualizations")
fig_bar, fig_hist = None, None
if not df_filtered.empty:
    viz1, viz2 = st.columns(2)
    with viz1:
        cat_cols = [f["column"] for f in (st.session_state.suggested_filters or []) if isinstance(f, dict) and f.get("type")=="multiselect" and f.get("column") in df_filtered.columns]
        if cat_cols:
            chart_cat_col = st.selectbox("Choose a categorical column to plot", options=cat_cols)
            if chart_cat_col:
                counts = df_filtered[chart_cat_col].value_counts().nlargest(10)
                fig_bar = px.bar(counts, x=counts.index, y=counts.values, title=f"Top 10 in {chart_cat_col}", labels={"x": chart_cat_col,"y":"Count"}, color=counts.index, template="plotly_dark")
                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No suitable categorical columns suggested for a bar chart.")

    with viz2:
        num_cols = [f["column"] for f in (st.session_state.suggested_filters or []) if isinstance(f, dict) and f.get("type")=="slider" and f.get("column") in df_filtered.columns]
        if num_cols:
            chart_num_col = st.selectbox("Choose a numerical column to plot", options=num_cols)
            if chart_num_col:
                fig_hist = px.histogram(df_filtered, x=chart_num_col, title=f"Distribution of {chart_num_col}", nbins=30, template="plotly_dark")
                st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("No suitable numerical columns suggested for a histogram.")

# --- Statistical Analysis ---
st.markdown("---")
st.header(" Statistical Analysis & Code Suggestions")
if st.button(" Generate Statistical Tests & Code"):
    with st.spinner("Analyzing dataset..."):
        analysis_code = get_statistical_analysis_code(st.session_state.model, df_filtered)
        st.text_area(" AI Recommended Statistical Tests & Code", analysis_code, height=400)

# --- Data Table ---
st.markdown("---")
st.header(" Data Explorer")
st.dataframe(df_filtered, use_container_width=True)

# --- Data Quality Deep Dive ---
st.markdown("---")
st.header(" Data Quality Deep Dive")
if st.button("Scan for Advanced Data Quality Issues"):
    with st.spinner("Scanning for duplicates, inconsistencies, and more..."):
        issues_df = find_data_issues(df_filtered)
        st.session_state.issues_df = issues_df
        if issues_df.empty:
            st.success("✅ No data quality issues found!")
        else:
            st.error("⚠️ Issues detected in the dataset:")
            id_name_cols = [col for col in df_filtered.columns if any(k in col.lower() for k in ["id","name"])]
            for issue_type, group in issues_df.groupby("Check_Type"):
                st.subheader(f"{issue_type} Issues")
                display_cols = ["Excel_Cell","column","issue","value"] + id_name_cols
                st.dataframe(group[display_cols].drop_duplicates(), use_container_width=True)

# ======================== PDF REPORT GENERATION ========================
def generate_dashboard_pdf(df: pd.DataFrame, quality_score: float, quality_table_df: pd.DataFrame, charts: list, issues_df: pd.DataFrame | None = None):
    file_path = "dashboard_report.pdf"
    doc = SimpleDocTemplate(file_path, pagesize=A4, leftMargin=24, rightMargin=24, topMargin=24, bottomMargin=24)
    styles = getSampleStyleSheet()
    wrap_style = ParagraphStyle(name='wrap', wordWrap='CJK', fontSize=10)

    elements = []
    elements.append(Paragraph("AI-Powered Dynamic Dashboard Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    # KPIs
    elements.append(Paragraph("Key Performance Indicators", styles["Heading2"]))
    total_missing = int(df.isnull().sum().sum())
    total_cells = int(np.prod(df.shape)) if df.size else 0
    missing_percent = (total_missing / total_cells * 100) if total_cells > 0 else 0
    kpi_table = Table([["Rows", f"{df.shape[0]:,}"], ["Columns", f"{df.shape[1]:,}"], ["% Missing Data", f"{missing_percent:.2f}%"], ["Data Quality Score", f"{quality_score:.2f}"]], colWidths=[160,300])
    kpi_table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#4facfe")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("GRID",(0,0),(-1,-1),0.5,colors.black),("ALIGN",(0,0),(-1,-1),"CENTER")]))
    elements.append(kpi_table)
    elements.append(Spacer(1,12))

    # Data Quality Table
    elements.append(Paragraph("Data Quality Report", styles["Heading2"]))
    if quality_table_df is not None and not quality_table_df.empty:
        header = ["Column","DataType","Alerts","Suggested Fix"]
        data_rows = [[Paragraph(str(c),wrap_style) for c in header]]
        for row in quality_table_df.astype(str).values.tolist():
            data_rows.append([Paragraph(str(r), wrap_style) for r in row])
        dq_table = Table(data_rows, repeatRows=1, colWidths=[100,80,120,200])
        dq_table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#2b2b2b")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("GRID",(0,0),(-1,-1),0.25,colors.black),("VALIGN",(0,0),(-1,-1),"TOP")]))
        elements.append(dq_table)
    else:
        elements.append(Paragraph("No data quality details available.", styles["Normal"]))
    elements.append(Spacer(1,12))

    # Charts
    if charts:
     elements.append(Paragraph("Visualizations", styles["Heading2"]))
    for chart_file in charts:
        try:
            img = RLImage(chart_file, width=400, height=300)  # adjust width/height as needed
            elements.append(img)
            elements.append(Spacer(1,12))
        except Exception as e:
            continue

    # Issues Deep Dive
    if issues_df is not None and isinstance(issues_df,pd.DataFrame) and not issues_df.empty:
        elements.append(Paragraph("Data Quality Deep Dive (Issues)", styles["Heading2"]))
        cols = [c for c in ["Excel_Cell","column","issue","value"] if c in issues_df.columns]
        show_df = issues_df[cols].astype(str).drop_duplicates().head(200)
        data_rows = [[Paragraph(str(c), wrap_style) for c in show_df.columns]] + [[Paragraph(str(r), wrap_style) for r in row] for row in show_df.values.tolist()]
        iss_table = Table(data_rows, repeatRows=1)
        iss_table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#d9534f")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("GRID",(0,0),(-1,-1),0.25,colors.black)]))
        elements.append(iss_table)

    doc.build(elements)
    return file_path

# --- PDF Export ---
st.markdown("---")
st.header(" 📄 Export Full Dashboard Report")
if st.button("Generate & Download Report as PDF"):
    with st.spinner("Compiling dashboard into PDF..."):
        charts = []
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                if fig_bar is not None:
                    bar_path = os.path.join(tmpdir,"bar_chart.png")
                    fig_bar.write_image(bar_path)
                    charts.append(bar_path)
            except Exception:
                pass
            try:
                if fig_hist is not None:
                    hist_path = os.path.join(tmpdir,"hist_chart.png")
                    fig_hist.write_image(hist_path)
                    charts.append(hist_path)
            except Exception:
                pass
            quality_table_df_pdf = prepare_quality_table(df_filtered, quality_details) if isinstance(quality_details,pd.DataFrame) else pd.DataFrame()
            pdf_file = generate_dashboard_pdf(df_filtered, quality_score, quality_table_df_pdf, charts, st.session_state.get("issues_df"))
            with open(pdf_file,"rb") as f:
                st.download_button("⬇️ Download Dashboard PDF", f, file_name="dashboard_report.pdf", mime="application/pdf")
