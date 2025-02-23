import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
import random
import networkx as nx
import matplotlib.pyplot as plt  # Required for NetworkX graph drawing
import io  # For capturing matplotlib output

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

def create_threat_knowledge_graph(df):
    """Creates a knowledge graph using NetworkX."""
    try:
        G = nx.Graph()

        # Add nodes for threat categories and actors
        for threat in df['Threat Category'].unique():
            G.add_node(threat, type='Threat Category', color='red')
        for actor in df['Threat Actor'].unique():
            G.add_node(actor, type='Threat Actor', color='blue')

        # Add edges representing the relationship between threats and actors
        for index, row in df.iterrows():
            G.add_edge(row['Threat Category'], row['Threat Actor'])

        # Node coloring based on type
        node_colors = [node[1]['color'] if 'color' in node[1] else 'gray' for node in G.nodes(data=True)]

        # Draw the graph with adjusted layout for better aesthetics
        plt.figure(figsize=(12, 8))  # Adjust figure size
        pos = nx.spring_layout(G, k=0.5, iterations=50)  # Adjust layout parameters

        nx.draw(G, pos, with_labels=True, node_color=node_colors, font_size=10, node_size=2000, alpha=0.7, width=0.8, edge_color="gray")  # Adjust node size, alpha, and edge color
        plt.title("Threat Category - Threat Actor Knowledge Graph")

        # Convert plot to image for Streamlit
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')  # Use 'tight' to avoid cutting labels
        buf.seek(0)
        plt.close()
        return buf

    except Exception as e:
        st.error(f"Error creating knowledge graph: {e}")
        return None

def create_attack_defense_graph(df):
    """Creates a knowledge graph linking attack vectors and defense mechanisms."""
    try:
        G = nx.Graph()

        # Add nodes for attack vectors and defense mechanisms
        for attack in df['Attack Vector'].unique():
            G.add_node(attack, type='Attack Vector', color='green')
        for defense in df['Suggested Defense Mechanism'].unique():
            G.add_node(defense, type='Defense Mechanism', color='orange')

        # Add edges representing the relationship between attacks and defenses
        for index, row in df.iterrows():
            G.add_edge(row['Attack Vector'], row['Suggested Defense Mechanism'])

        # Node coloring
        node_colors = [node[1]['color'] if 'color' in node[1] else 'gray' for node in G.nodes(data=True)]

        # Draw graph
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=0.5, iterations=50)  # Adjust layout parameters

        nx.draw(G, pos, with_labels=True, node_color=node_colors, font_size=10, node_size=2000, alpha=0.7, width=0.8, edge_color="gray")
        plt.title("Attack Vector - Defense Mechanism Knowledge Graph")

        # Convert plot to image for Streamlit
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')  # Use 'tight' to avoid cutting labels
        buf.seek(0)
        plt.close()
        return buf
    except Exception as e:
        st.error(f"Error creating attack-defense graph: {e}")
        return None


def create_scatter_matrix(df):
    """Creates a scatter matrix (pairs plot) of numerical features."""
    try:
        numerical_cols = df.select_dtypes(include=['number']).columns
        if len(numerical_cols) < 2:
            st.warning("Not enough numerical columns to create a scatter matrix.")
            return None
        fig = px.scatter_matrix(df, dimensions=numerical_cols, color="Risk Level Prediction")
        fig.update_layout(title="Scatter Matrix of Numerical Features")
        return fig
    except Exception as e:
        st.error(f"Error creating scatter matrix: {e}")
        return None

def create_correlation_heatmap(df):
    """Creates a correlation heatmap."""
    try:
        numerical_cols = df.select_dtypes(include=['number']).columns
        if len(numerical_cols) < 2:
            st.warning("Not enough numerical columns to create a correlation heatmap.")
            return None
        corr = df[numerical_cols].corr()
        fig = px.imshow(corr, labels=dict(x="Features", y="Features", color="Correlation"),
                        x=numerical_cols.tolist(),
                        y=numerical_cols.tolist(),
                        color_continuous_scale="RdBu")
        fig.update_layout(title="Correlation Heatmap of Numerical Features")
        return fig
    except Exception as e:
        st.error(f"Error creating correlation heatmap: {e}")
        return None

# --- Main App ---
def main():
    st.title("🛡️ Cybersecurity Threat Intelligence Dashboard")
    st.markdown("Enhanced Threat Visualizations")

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

        knowledge_graph_img = create_threat_knowledge_graph(filtered_df)
        if knowledge_graph_img:
            st.image(knowledge_graph_img, caption="Threat Category - Threat Actor Knowledge Graph", use_column_width=True)

    with col2:
        sunburst_fig = create_sunburst(filtered_df)
        if sunburst_fig:
            st.plotly_chart(sunburst_fig, use_container_width=True)

        parallel_fig = create_parallel_categories(filtered_df)
        if parallel_fig:
            st.plotly_chart(parallel_fig, use_container_width=True)

        attack_defense_graph_img = create_attack_defense_graph(filtered_df)
        if attack_defense_graph_img:
            st.image(attack_defense_graph_img, caption="Attack Vector - Defense Mechanism Knowledge Graph", use_column_width=True)

    col3, col4 = st.columns(2)

    with col3:
        scatter_matrix_fig = create_scatter_matrix(filtered_df)
        if scatter_matrix_fig:
            st.plotly_chart(scatter_matrix_fig, use_container_width=True)

    with col4:
        correlation_heatmap_fig = create_correlation_heatmap(filtered_df)
        if correlation_heatmap_fig:
            st.plotly_chart(correlation_heatmap_fig, use_container_width=True)

    # --- Raw Data ---
    if st.checkbox("Show Raw Data"):
        st.dataframe(filtered_df)

if __name__ == "__main__":
    main()
