import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import ast
import random
from collections import Counter

# Set page config
st.set_page_config(layout="wide", page_title="Cybersecurity Threat Intelligence Dashboard")

# Custom CSS - Modern, Clean Theme
st.markdown("""
    <style>
    .main {
        background-color: #F5F5F5; /* Light Gray */
        color: #333333; /* Dark Gray */
    }
    .stPlotlyChart {
        background-color: #FFFFFF; /* White */
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1); /* Subtle Shadow */
    }
    .stSidebar {
        background-color: #E0E0E0; /* Light Gray */
        color: #333333;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #2E86C1; /* Ocean Blue */
    }
    </style>
    """, unsafe_allow_html=True)

# Function to load and preprocess data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Cybersecurity_Dataset.csv')
        # Convert string representations of lists to actual lists
        list_columns = ['IOCs (Indicators of Compromise)', 'Keyword Extraction', 'Named Entities (NER)']
        for col in list_columns:
            df[col] = df[col].apply(ast.literal_eval)

        # Convert 'Date' column to datetime if it exists and contains strings
        if 'Date' in df.columns:
            try:
                df['Date'] = pd.to_datetime(df['Date'])
            except ValueError:
                st.warning("Could not automatically convert 'Date' column to datetime.  Please check the date format in your CSV.")
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def create_knowledge_graph(df, layout_type="spring", spring_k=0.1):
    """Creates a knowledge graph visualization of cybersecurity threats."""
    G = nx.Graph()

    # Add nodes and edges from the data
    for _, row in df.iterrows():
        G.add_node(row['Threat Actor'], node_type='actor', severity=row['Severity Score'])
        G.add_node(row['Threat Category'], node_type='category')
        G.add_node(row['Attack Vector'], node_type='vector')
        G.add_node(row['Suggested Defense Mechanism'], node_type='defense')

        G.add_edge(row['Threat Actor'], row['Threat Category'], weight=row['Risk Level Prediction'])
        G.add_edge(row['Threat Actor'], row['Attack Vector'], weight=row['Severity Score'])
        G.add_edge(row['Attack Vector'], row['Suggested Defense Mechanism'], weight=row['Risk Level Prediction'])

    # Calculate layout
    if layout_type == "spring":
        pos = nx.spring_layout(G, k=spring_k, iterations=50, seed=42)
    elif layout_type == "circular":
        pos = nx.circular_layout(G)
    elif layout_type == "random":
        pos = nx.random_layout(G, seed=42)
    elif layout_type == "spectral":
        pos = nx.spectral_layout(G, weight='weight', dim=2, scale=1)
    else:
        pos = nx.spring_layout(G, k=spring_k, iterations=50, seed=42)

    # Edge trace
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # Node trace
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    node_sizes = []
    node_hovertemplate = []

    colors = {
        'actor': '#E74C3C',    # Red
        'category': '#3498DB', # Blue
        'vector': '#F1C40F',   # Yellow
        'defense': '#2ECC71'   # Green
    }

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_type = G.nodes[node].get('node_type', 'unknown')
        node_colors.append(colors.get(node_type, '#CCCCCC'))
        node_sizes.append(15)
        hover_text = f"<b>{node}</b><br>Type: {node_type.capitalize()}"
        if node_type == 'actor':
            hover_text += f"<br>Severity: {G.nodes[node].get('severity', 'N/A')}"
        node_hovertemplate.append(hover_text)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        marker=dict(
            color=node_colors,
            size=node_sizes,
            line_width=2
        ),
        hovertemplate=node_hovertemplate
    )

    # Create the figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="Cybersecurity Threat Intelligence Network",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    return fig

def create_treemap(df, color_metric='Risk Level Prediction'):
    """Creates a treemap visualization of threat categories and actors."""
    fig = px.treemap(df, path=['Threat Category', 'Threat Actor'],
                     color=color_metric,
                     hover_data=['Severity Score'],
                     color_continuous_scale='RdYlGn')
    fig.update_layout(title="Threat Category and Actor Treemap")
    return fig

def create_sunburst(df, color_metric='Severity Score'):
    """Creates a sunburst chart to visualize hierarchical relationships between attack vectors and defense mechanisms."""
    fig = px.sunburst(df, path=['Attack Vector', 'Suggested Defense Mechanism'],
                      color=color_metric,
                      color_continuous_scale='viridis')
    fig.update_layout(title="Attack Vector and Defense Mechanism Sunburst")
    return fig

def create_ioc_barchart(df, n=10):
    """Creates a bar chart showing the frequency of Indicators of Compromise (IOCs)."""
    all_iocs = []
    for ioc_list in df['IOCs (Indicators of Compromise)']:
        all_iocs.extend(ioc_list)
    ioc_counts = Counter(all_iocs)
    ioc_df = pd.DataFrame(ioc_counts.items(), columns=['IOC', 'Count']).sort_values('Count', ascending=False).head(n)
    fig = px.bar(ioc_df, x='IOC', y='Count',
                 title=f"Top {n} Most Frequent Indicators of Compromise")
    fig.update_layout(xaxis_title="Indicator of Compromise", yaxis_title="Frequency")
    return fig

def create_scatter_matrix(df, columns=['Severity Score', 'Risk Level Prediction'], color_column=None):
    """Creates a scatter matrix for selected columns."""

    # Check if the color column exists
    if color_column and color_column not in df.columns:
        st.warning(f"The specified color column '{color_column}' does not exist in the data.  Coloring will be disabled.")
        color_column = None

    try:
        fig = px.scatter_matrix(df[columns], color=color_column)
        fig.update_layout(title="Scatter Matrix of Threat Attributes")
        return fig
    except ValueError as e:
        st.error(f"Error creating Scatter Matrix: {e}")
        return None


def create_parallel_categories(df, columns=['Threat Category', 'Attack Vector', 'Suggested Defense Mechanism'],
                               threat_categories=None, attack_vectors=None, defense_mechanisms=None,
                               color_metric='Severity Score'):
    """Creates a parallel categories diagram to visualize relationships with filtering and customization."""

    # Check if specified columns exist in the dataframe
    valid_columns = [col for col in columns if col in df.columns]
    if not valid_columns:
        st.warning("None of the specified columns for the Parallel Categories diagram exist in the data.")
        return None

    # Apply filters
    filtered_df = df.copy()
    if threat_categories:
        filtered_df = filtered_df[filtered_df['Threat Category'].isin(threat_categories)]
    if attack_vectors:
        filtered_df = filtered_df[filtered_df['Attack Vector'].isin(attack_vectors)]
    if defense_mechanisms:
        filtered_df = filtered_df[filtered_df['Suggested Defense Mechanism'].isin(defense_mechanisms)]

    if filtered_df.empty:
        st.warning("No data to display after applying filters to the Parallel Categories diagram.")
        return None

    # Customizing hover data to show more information
    category_counts = filtered_df.groupby(valid_columns).size().reset_index(name='Count')
    filtered_df = pd.merge(filtered_df, category_counts, on=valid_columns, how='left')

    #List of columns to include. hover_data argument not required
    columns_to_include = valid_columns + ['Severity Score', 'Risk Level Prediction', 'Count']


    try:
        fig = px.parallel_categories(filtered_df[columns_to_include],
                                     color=filtered_df[color_metric],
                                     color_continuous_scale='viridis',
                                     labels={col: col.replace("_", " ") for col in columns_to_include}) # Clean labels
        fig.update_layout(title="Parallel Categories Diagram of Threat Characteristics",
                          title_x=0.5) #Center title
        return fig
    except ValueError as e:
        st.error(f"Error creating Parallel Categories diagram: {e}")
        return None

# Load the data
df = load_data()

if df is not None:
    # Title and description
    st.title("🛡️ Cybersecurity Threat Intelligence Dashboard")
    st.markdown("### Unveiling Threat Landscapes with Innovative Visualizations")

    # Sidebar filters
    st.sidebar.title("Filters")

    # Date Range Filter (assuming a 'Date' column exists)
    if 'Date' in df.columns:
        min_date = df['Date'].min()
        max_date = df['Date'].max()
        selected_date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
        start_date, end_date = selected_date_range
        df = df[(df['Date'] >= pd.Timestamp(start_date)) & (df['Date'] <= pd.Timestamp(end_date))]

    # Threat Category filter
    selected_categories = st.sidebar.multiselect(
        "Select Threat Categories",
        options=df['Threat Category'].unique(),
        default=list(df['Threat Category'].unique())  # Default to all
    )

    # Threat Actor filter
    selected_actors = st.sidebar.multiselect(
        "Select Threat Actors",
        options=df['Threat Actor'].unique(),
        default=list(df['Threat Actor'].unique()) # Default to all
    )

    # Attack Vector filter
    selected_attack_vectors = st.sidebar.multiselect(
        "Select Attack Vectors",
        options=df['Attack Vector'].unique(),
        default=list(df['Attack Vector'].unique())
    )

     #Defense Mechanism filter
    selected_defense_mechanisms = st.sidebar.multiselect(
        "Select Defense Mechanisms",
        options=df['Suggested Defense Mechanism'].unique(),
        default=list(df['Suggested Defense Mechanism'].unique())
    )

    # Risk Level filter
    selected_risk_levels = st.sidebar.multiselect(
        "Select Risk Levels",
        options=df['Risk Level Prediction'].unique(),
        default=list(df['Risk Level Prediction'].unique())
    )

    # Severity Score Filter
    selected_severity_range = st.sidebar.slider(
        "Select Severity Score Range",
        min_value=int(df['Severity Score'].min()),
        max_value=int(df['Severity Score'].max()),
        value=(int(df['Severity Score'].min()), int(df['Severity Score'].max()))
    )

    # Filter the dataframe
    filtered_df = df[
        (df['Threat Category'].isin(selected_categories)) &
        (df['Threat Actor'].isin(selected_actors)) &
        (df['Attack Vector'].isin(selected_attack_vectors)) &
        (df['Suggested Defense Mechanism'].isin(selected_defense_mechanisms)) &
        (df['Risk Level Prediction'].isin(selected_risk_levels)) &
        (df['Severity Score'] >= selected_severity_range[0]) &
        (df['Severity Score'] <= selected_severity_range[1])
    ]

    # --- Visualizations ---
    col1, col2 = st.columns(2)

    with col1:
        # Knowledge Graph
        st.subheader("Threat Relationship Network")
        layout_type = st.selectbox("Knowledge Graph Layout", options=["spring", "circular", "random", "spectral"], key="kg_layout")
        fig_kg = create_knowledge_graph(filtered_df, layout_type=layout_type)
        st.plotly_chart(fig_kg, use_container_width=True)

        # IOC Barchart
        st.subheader("Common Indicators of Compromise")
        num_iocs = st.slider("Number of IOCs to Display", min_value=5, max_value=30, value=10)
        fig_ioc_bar = create_ioc_barchart(filtered_df, n=num_iocs)
        st.plotly_chart(fig_ioc_bar, use_container_width=True)



    with col2:
        # Treemap
        st.subheader("Threat Landscape Overview")
        treemap_color = st.selectbox("Treemap Color Metric", options=['Risk Level Prediction', 'Severity Score'], key="treemap_color")
        fig_treemap = create_treemap(filtered_df, color_metric=treemap_color)
        st.plotly_chart(fig_treemap, use_container_width=True)

        # Sunburst
        st.subheader("Attack Vectors and Defense Strategies")
        sunburst_color = st.selectbox("Sunburst Color Metric", options=['Severity Score', 'Risk Level Prediction'], key="sunburst_color")
        fig_sunburst = create_sunburst(filtered_df, color_metric=sunburst_color)
        st.plotly_chart(fig_sunburst, use_container_width=True)


    #Scatter Matrix
    st.subheader("Scatter Matrix of Threat Attributes")
    scatter_color = st.selectbox("Scatter Matrix Color", options=[None, 'Severity Score', 'Risk Level Prediction', 'Threat Category'], key="scatter_color")
    fig_scatter = create_scatter_matrix(filtered_df, color_column=scatter_color)
    if fig_scatter:
        st.plotly_chart(fig_scatter, use_container_width=True)

    # --- Parallel Categories Filters ---
    st.sidebar.subheader("Parallel Categories Filters")
    pc_threat_categories = st.sidebar.multiselect(
        "PC Threat Categories",
        options=filtered_df['Threat Category'].unique(),
        default=list(filtered_df['Threat Category'].unique())
    )
    pc_attack_vectors = st.sidebar.multiselect(
        "PC Attack Vectors",
        options=filtered_df['Attack Vector'].unique(),
        default=list(filtered_df['Attack Vector'].unique())
    )
    pc_defense_mechanisms = st.sidebar.multiselect(
        "PC Defense Mechanisms",
        options=filtered_df['Suggested Defense Mechanism'].unique(),
        default=list(filtered_df['Suggested Defense Mechanism'].unique())
    )

    #Parallel Categories
    st.subheader("Parallel Categories of Threat Characteristics")

    #Explanation for the chart
    st.write("""
            This Parallel Categories Diagram visualizes the relationships between different threat characteristics. 
            Each vertical line represents a category: Threat Category, Attack Vector, and Suggested Defense Mechanism. 
            The lines connecting the categories show the distribution of threats across these characteristics. 
            The color of the lines indicates the Severity Score, with darker colors representing higher severity.
            Hover over the lines to see detailed information about each category and their relationships, including the count of occurrences for each combination.
            """)

    fig_parallel = create_parallel_categories(
        filtered_df,
        threat_categories=pc_threat_categories,
        attack_vectors=pc_attack_vectors,
        defense_mechanisms=pc_defense_mechanisms
    )
    if fig_parallel:  # Only display if the chart was successfully created
        st.plotly_chart(fig_parallel, use_container_width=True)

    # --- Key Metrics ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Threats", len(filtered_df))
    with col2:
        st.metric("Unique Threat Actors", filtered_df['Threat Actor'].nunique())
    with col3:
        st.metric("Average Severity", round(filtered_df['Severity Score'].mean(), 2))
    with col4:
        st.metric("High Risk Threats (>=4)", len(filtered_df[filtered_df['Risk Level Prediction'] >= 4]))

    # --- Raw Data Exploration ---
    st.subheader("Raw Data Exploration")
    if st.checkbox("Show Raw Data"):
        st.dataframe(filtered_df)

    # Download filtered data
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name='filtered_cybersecurity_data.csv',
        mime='text/csv',
    )

else:
    st.error("Please ensure 'Cybersecurity_Dataset.csv' is in the current working directory.")