import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from utils import (
    get_metadata,
    calculate_data_quality,
    get_gemini_insights,
    get_cleaning_suggestion,
    get_statistical_analysis_code,
    find_data_issues,
    get_excel_location
)
from google.generativeai import configure, GenerativeModel

# --- Page Configuration ---
st.set_page_config(
    page_title="AI-Powered Dynamic Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .stMetric {
        border-radius: 10px;
        background-color: #262730;
        border: 1px solid #4A4A4A;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
        transition: 0.3s;
    }
    .stMetric:hover {
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }
    .stMetric .st-emotion-cache-1g8i8fg {
        font-size: 2.5rem;
        font-weight: bold;
        color: #00F2FE;
    }
    .stMetric .st-emotion-cache-1xarl3l {
        color: #A1A1A1;
    }
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #4facfe, #00f2fe);
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        border: 1px solid #4facfe;
        color: #FFFFFF;
        background-color: #4facfe;
    }
    .st-emotion-cache-1r6slb0 {
        background-color: #262730;
    }
</style>
""", unsafe_allow_html=True)

# --- Gauge Chart Helper ---
def create_gauge_chart(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Data Quality Score", 'font': {'size': 24, 'color': 'white'}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#00F2FE"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#d9534f'},
                {'range': [50, 80], 'color': '#f0ad4e'},
                {'range': [80, 100], 'color': '#5cb85c'}
            ],
        }
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white", 'family': "Arial"})
    return fig

# --- App State ---
for key in ['df', 'df_raw', 'report_text', 'suggested_filters', 'api_key_valid', 'df_filtered', 'uploaded_file_name']:
    if key not in st.session_state:
        st.session_state[key] = [] if key == 'suggested_filters' else None

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

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", disabled=not st.session_state.api_key_valid)

    if uploaded_file and st.session_state.api_key_valid:
        if st.session_state.get('uploaded_file_name') != uploaded_file.name:
            # Reset state for new file
            for key in ['df', 'df_raw', 'report_text', 'suggested_filters', 'df_filtered']:
                st.session_state[key] = None
            st.session_state.uploaded_file_name = uploaded_file.name

        if st.session_state.df is None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df_raw = df.copy()
                st.success("File uploaded successfully! Please select columns to proceed.")

                st.subheader(" Select Columns for Analysis")
                col1, col2 = st.columns([2, 1])
                with col1:
                    selected_columns = st.multiselect("Choose columns to keep for analysis:",
                                                    options=df.columns.tolist())
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

                csv_download = df_filtered.to_csv(index=False).encode('utf-8')
                st.download_button(" Download Filtered CSV", data=csv_download,
                                    file_name="filtered_data.csv", mime="text/csv")

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

# --- Stop if no data selected ---
if st.session_state.df is None:
    st.info("Please upload a CSV and confirm columns to analyze.")
    st.stop()

# --- Main Dashboard ---
df = st.session_state.df
df_filtered = st.session_state.df_filtered if st.session_state.df_filtered is not None else df

# --- Data Quality ---
st.header(" Data Quality Assessment")
quality_score, quality_details = calculate_data_quality(df.copy())
score_col, details_col = st.columns([1, 2])

with score_col:
    st.plotly_chart(create_gauge_chart(quality_score), use_container_width=True)

with details_col:
    st.info("This score reflects data completeness, consistency, and anomaly detection. Higher is better.")
    with st.expander("Show Detailed Quality Report & Suggestions"):
        st.dataframe(quality_details, use_container_width=True)

# --- Data Cleaning Suggestions ---
st.markdown("---")
st.header(" AI-Powered Data Cleaning")

# Check if 'Alerts' column exists before filtering
if 'Alerts' in quality_details.columns:
    columns_with_issues = quality_details[quality_details['Alerts'] != "No issues"]['Column'].tolist()
else:
    columns_with_issues = []

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

# --- KPIs and AI Insights ---
st.markdown("---")
st.header(" Key Performance Indicators")
kpi_col1, kpi_col2, kpi_col3, insights_col = st.columns(4)

kpi_col1.metric("Filtered Rows", f"{df_filtered.shape[0]:,}")
kpi_col2.metric("Total Columns", f"{df.shape[1]:,}")

total_missing = df_filtered.isnull().sum().sum()
total_cells = np.prod(df_filtered.shape)
missing_percent = (total_missing / total_cells) * 100 if total_cells > 0 else 0
kpi_col3.metric("% Missing Data (Filtered)", f"{missing_percent:.2f}%")

with insights_col:
    st.subheader(" AI Insights")
    if st.session_state.report_text:
        st.download_button(
            label="Download Full Report",
            data=st.session_state.report_text,
            file_name="gemini_data_insights.txt",
            mime="text/plain"
        )

# --- Visualizations ---
st.markdown("---")
st.header(" Interactive Visualizations")

if df_filtered.empty:
    st.warning("No data matches the current filter settings.")
else:
    viz1, viz2 = st.columns(2)

    with viz1:
        cat_cols = [f['column'] for f in st.session_state.suggested_filters if f['type'] == 'multiselect' and f['column'] in df_filtered.columns]
        if cat_cols:
            chart_cat_col = st.selectbox("Choose a categorical column to plot", options=cat_cols)
            if chart_cat_col:
                counts = df_filtered[chart_cat_col].value_counts().nlargest(10)
                fig_bar = px.bar(counts, x=counts.index, y=counts.values, title=f"Top 10 in {chart_cat_col}",
                                    labels={'x': chart_cat_col, 'y': 'Count'}, color=counts.index,
                                    template="plotly_dark")
                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No suitable categorical columns suggested for a bar chart.")

    with viz2:
        num_cols = [f['column'] for f in st.session_state.suggested_filters if f['type'] == 'slider' and f['column'] in df_filtered.columns]
        if num_cols:
            chart_num_col = st.selectbox("Choose a numerical column to plot", options=num_cols)
            if chart_num_col:
                fig_hist = px.histogram(df_filtered, x=chart_num_col, title=f"Distribution of {chart_num_col}", nbins=30,
                                        template="plotly_dark")
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

        if issues_df.empty:
            st.success("✅ No data quality issues found!")
        else:
            st.error("⚠️ Issues detected in the dataset:")
            
            # Get ID or Name columns for context
            id_name_cols = [col for col in df_filtered.columns if any(k in col.lower() for k in ["id", "name"])]
            
            # Group and display issues by type
            for issue_type, group in issues_df.groupby('Check_Type'):
                st.subheader(f"{issue_type} Issues")
                display_cols = ["Excel_Cell", "column", "issue", "value"] + id_name_cols
                st.dataframe(group[display_cols].drop_duplicates(), use_container_width=True)
