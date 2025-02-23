import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
import random

# --- Configuration ---
DATA_URL = 'Cybersecurity_Dataset.csv'  # Specify the location of the CSV file

# Set page config
st.set_page_config(layout="wide", page_title="Cybersecurity Threat Intelligence Dashboard")

# Custom CSS - Minimal Theme
st.markdown("""
    <style>
    body { background-color: #F5F5F5; color: #333333; }
    h1, h2, h3, h4, h5, h6 { color: #2E86C1; }
    </style>
    """, unsafe_allow_html=True)

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_data(url):
    """Loads data from CSV, handles basic cleaning."""
    try:
        df = pd.read_csv(url)
        if 'Date' in df.columns:
            try:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            except ValueError:
                st.warning("Could not convert 'Date' column to datetime.")
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# --- Visualization Functions ---

def create_treemap(df):
    """Creates a treemap."""
    try:
        fig = px.treemap(df, path=['Threat Category', 'Threat Actor'],
                        color='Risk Level Prediction',
                        hover_data=['Severity Score'],
                        color_continuous_scale='RdYlGn')
        fig.update_layout(title="Threat Category and Actor Treemap")
        return fig
    except Exception as e:
        st.error(f"Error creating treemap: {e}")
        return None

def create_sunburst(df):
    """Creates a sunburst chart."""
    try:
        fig = px.sunburst(df, path=['Attack Vector', 'Suggested Defense Mechanism'],
                        color='Severity Score',
                        color_continuous_scale='viridis')
        fig.update_layout(title="Attack Vector and Defense Mechanism Sunburst")
        return fig
    except Exception as e:
        st.error(f"Error creating sunburst: {e}")
        return None

def create_ioc_barchart(df, n=10):
    """Creates a bar chart of Indicators of Compromise (IOCs).  Requires the IOCs column to be pre-processed as a list"""
    # Flatten IOCs
    try:
        all_iocs = [ioc for ioc_list in df['IOCs (Indicators of Compromise)'].dropna() for ioc in ioc_list]
        ioc_counts = Counter(all_iocs)
        ioc_df = pd.DataFrame(ioc_counts.items(), columns=['IOC', 'Count']).sort_values('Count', ascending=False).head(n)
        fig = px.bar(ioc_df, x='IOC', y='Count', title=f"Top {n} IOCs")
        fig.update_layout(xaxis_title="Indicator of Compromise", yaxis_title="Frequency")
        return fig
    except Exception as e:
        st.error(f"Error creating IOC barchart: {e}")
        return None

def create_parallel_categories(df):
    """Creates a parallel categories diagram. Handles missing data."""
    try:
        columns = ['Threat Category', 'Attack Vector', 'Suggested Defense Mechanism']
        df_clean = df[columns].dropna().astype(str)  # Drop NaNs and convert to string
        fig = px.parallel_categories(df_clean, color=df['Severity Score'], color_continuous_scale='viridis')
        fig.update_layout(title="Threat Characteristics")
        return fig
    except Exception as e:
        st.error(f"Error creating parallel categories: {e}")
        return None

# --- Main App ---
def main():
    st.title("🛡️ Cybersecurity Threat Intelligence Dashboard")
    st.markdown("Simple Threat Visualizations")

    df = load_data(DATA_URL)
    if df is None:
        st.stop()

    # --- Sidebar Filters ---
    st.sidebar.title("Filters")

    if 'Date' in df.columns:
        min_date = df['Date'].min()
        max_date = df['Date'].max()
        selected_date_range = st.sidebar.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
        start_date, end_date = selected_date_range
        df = df[(df['Date'] >= pd.Timestamp(start_date)) & (df['Date'] <= pd.Timestamp(end_date))]

    selected_categories = st.sidebar.multiselect("Threat Categories", options=sorted(df['Threat Category'].unique()), default=list(df['Threat Category'].unique()))
    selected_actors = st.sidebar.multiselect("Threat Actors", options=sorted(df['Threat Actor'].unique()), default=list(df['Threat Actor'].unique()))
    selected_vectors = st.sidebar.multiselect("Attack Vectors", options=sorted(df['Attack Vector'].unique()), default=list(df['Attack Vector'].unique()))
    selected_defenses = st.sidebar.multiselect("Defense Mechanisms", options=sorted(df['Suggested Defense Mechanism'].unique()), default=list(df['Suggested Defense Mechanism'].unique()))
    selected_risks = st.sidebar.multiselect("Risk Levels", options=sorted(df['Risk Level Prediction'].unique()), default=list(df['Risk Level Prediction'].unique()))
    severity_range = st.sidebar.slider("Severity Score Range", int(df['Severity Score'].min()), int(df['Severity Score'].max()), (int(df['Severity Score'].min()), int(df['Severity Score'].max())))

    # --- Apply Filters ---
    filtered_df = df[
        df['Threat Category'].isin(selected_categories) &
        df['Threat Actor'].isin(selected_actors) &
        df['Attack Vector'].isin(selected_vectors) &
        df['Suggested Defense Mechanism'].isin(selected_defenses) &
        df['Risk Level Prediction'].isin(selected_risks) &
        (df['Severity Score'] >= severity_range[0]) &
        (df['Severity Score'] <= severity_range[1])
    ]

    # --- Visualizations ---
    col1, col2 = st.columns(2)

    with col1:
        treemap_fig = create_treemap(filtered_df)
        if treemap_fig:
            st.plotly_chart(treemap_fig, use_container_width=True)

        ioc_fig = create_ioc_barchart(filtered_df, n=10)
        if ioc_fig:
            st.plotly_chart(ioc_fig, use_container_width=True)

    with col2:
        sunburst_fig = create_sunburst(filtered_df)
        if sunburst_fig:
            st.plotly_chart(sunburst_fig, use_container_width=True)

        parallel_fig = create_parallel_categories(filtered_df)
        if parallel_fig:
            st.plotly_chart(parallel_fig, use_container_width=True)

    # --- Raw Data ---
    if st.checkbox("Show Raw Data"):
        st.dataframe(filtered_df)

if __name__ == "__main__":
    main()
