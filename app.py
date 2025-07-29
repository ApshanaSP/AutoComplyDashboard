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
    get_most_frequent_value
)

from google.generativeai import configure, GenerativeModel

# --- Page Configuration ---
st.set_page_config(
    page_title="AI-Powered Dynamic Dashboard",
    page_icon="✨",
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
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Data Quality Score", 'font': {'size': 24, 'color': 'white'}},
        gauge = {
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
for key in ['df', 'report_text', 'suggested_filters', 'api_key_valid', 'df_filtered']:
    if key not in st.session_state:
        st.session_state[key] = [] if key == 'suggested_filters' else None

# --- Title ---
st.title("✨ AI-Powered Dynamic Dashboard")
st.markdown("Upload any structured CSV, and let AI build a custom dashboard for you.")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration & Filters")

    API_KEYS = ["AIzaSyAQ64Ic2-IDIuFmGgUrmXqQ1XGdx1LedIg"]
    st.session_state.api_key_valid = False

    for i, key in enumerate(API_KEYS):
        try:
            configure(api_key=key)
            st.session_state.model = GenerativeModel("gemini-2.5-pro")
            st.session_state.api_key_valid = True
            st.success(f"Using API Key #{i+1}", icon="✅")
            break
        except Exception:
            st.warning(f"API Key #{i+1} failed. Trying next...", icon="⚠️")

    if not st.session_state.api_key_valid:
        st.error("All API keys failed. Please update them in the code.", icon="🚨")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", disabled=not st.session_state.api_key_valid)

    if uploaded_file and st.session_state.api_key_valid:
        if st.session_state.get('uploaded_file_name') != uploaded_file.name:
            st.session_state.df = None
            st.session_state.suggested_filters = []
            st.session_state.report_text = None
            st.session_state.df_filtered = None
            st.session_state.uploaded_file_name = uploaded_file.name

        if st.session_state.df is None:
            with st.spinner("🤖 AI is analyzing your data..."):
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.df = df
                    metadata = get_metadata(df.copy())
                    report, filters = get_gemini_insights(st.session_state.model, df, metadata)
                    st.session_state.report_text = report
                    st.session_state.suggested_filters = filters
                except Exception as e:
                    st.error(f"Error processing file: {e}")
                    st.session_state.df = None

    # --- Smart Filters ---
    if st.session_state.df is not None and st.session_state.suggested_filters:
        st.markdown("---")
        st.subheader("🔬 Smart Filters")
        df_filtered_local = st.session_state.df.copy()

        for f in st.session_state.suggested_filters:
            col = f.get('column')
            filter_type = f.get('type')

            if col not in df_filtered_local.columns:
                continue

            if filter_type == 'multiselect':
                unique_vals = df_filtered_local[col].dropna().unique().tolist()
                selected = st.multiselect(f"Filter by {col}", options=unique_vals, default=unique_vals)
                if selected != unique_vals:
                    df_filtered_local = df_filtered_local[df_filtered_local[col].isin(selected)]

            elif filter_type == 'slider' and pd.api.types.is_numeric_dtype(df_filtered_local[col]):
                min_val, max_val = float(df_filtered_local[col].min()), float(df_filtered_local[col].max())
                val_range = st.slider(f"Filter by {col}", min_val, max_val, (min_val, max_val))
                df_filtered_local = df_filtered_local[df_filtered_local[col].between(val_range[0], val_range[1])]

        st.session_state.df_filtered = df_filtered_local

# --- No Data Uploaded ---
if st.session_state.df is None:
    st.info("Please provide a valid Gemini API key and upload a CSV file to begin.")
    st.stop()

# --- Main Dashboard ---
df = st.session_state.df
df_filtered = st.session_state.df_filtered if st.session_state.df_filtered is not None else df

# --- Data Quality ---
st.header("📊 Data Quality Assessment")
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
st.header("🧹 AI-Powered Data Cleaning")

columns_with_issues = quality_details[quality_details['Alerts'] != "No issues"]['Column'].tolist()
if columns_with_issues:
    selected_col = st.selectbox("Select a column with issues", columns_with_issues)

    if st.button("💡 Get Cleaning Suggestion"):
        with st.spinner("Getting AI suggestion..."):
            suggestion = get_cleaning_suggestion(st.session_state.model, df_filtered, selected_col)
            st.success(suggestion)

        if st.button("✅ Apply Suggestion (Auto Fill)"):
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
st.header("📈 Key Performance Indicators")
kpi_col1, kpi_col2, kpi_col3, insights_col = st.columns(4)

kpi_col1.metric("Filtered Rows", f"{df_filtered.shape[0]:,}")
kpi_col2.metric("Total Columns", f"{df.shape[1]:,}")

total_missing = df_filtered.isnull().sum().sum()
total_cells = np.prod(df_filtered.shape)
missing_percent = (total_missing / total_cells) * 100 if total_cells > 0 else 0
kpi_col3.metric("% Missing Data (Filtered)", f"{missing_percent:.2f}%")

with insights_col:
    st.subheader("🧠 AI Insights")
    st.download_button(
        label="⬇️ Download Full Report",
        data=st.session_state.report_text,
        file_name="gemini_data_insights.txt",
        mime="text/plain"
    )

# --- Visualizations ---
st.markdown("---")
st.header("🎨 Interactive Visualizations")

if df_filtered.empty:
    st.warning("No data matches the current filter settings.")
else:
    viz1, viz2 = st.columns(2)

    with viz1:
        cat_cols = [f['column'] for f in st.session_state.suggested_filters if f['type'] == 'multiselect' and f['column'] in df_filtered.columns]
        if cat_cols:
            chart_cat_col = st.selectbox("Choose a categorical column to plot", options=cat_cols)
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
            fig_hist = px.histogram(df_filtered, x=chart_num_col, title=f"Distribution of {chart_num_col}", nbins=30,
                                    template="plotly_dark")
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("No suitable numerical columns suggested for a histogram.")


# --- Statistical Analysis ---
st.markdown("---")
st.header("📐 Statistical Analysis & Code Suggestions")

if st.button("🧪 Generate Statistical Tests & Code"):
    with st.spinner("Analyzing dataset..."):
        analysis_code = get_statistical_analysis_code(st.session_state.model, df_filtered)
        st.text_area("🔍 AI Recommended Statistical Tests & Code", analysis_code, height=400)


# --- Data Table ---
st.markdown("---")
st.header("👢 Data Explorer")
st.dataframe(df_filtered, use_container_width=True)
