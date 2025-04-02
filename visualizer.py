# visualizer.py
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pyLDAvis
# Ensure pyLDAvis.lda_model is imported correctly if needed, although prepare handles it
# import pyLDAvis.lda_model # Usually not needed directly if using prepare
import networkx as nx
from wordcloud import WordCloud
import pandas as pd
import numpy as np
from itertools import combinations # Make sure combinations is imported
import os
import re # Make sure re is imported
import logging
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Function to Save Plots ---
def save_plot(fig, filename, plot_dir=config.PLOT_DIR):
    """Saves a Plotly figure to HTML or Matplotlib figure to PNG."""
    filepath = os.path.join(plot_dir, filename)
    try:
        # Ensure the directory exists
        os.makedirs(plot_dir, exist_ok=True)

        if isinstance(fig, plt.Figure): # Check if it's a Matplotlib figure instance
            fig.savefig(filepath, bbox_inches='tight')
            logging.info(f"Matplotlib plot saved to {filepath}")
            plt.close(fig) # Close the figure to free memory
        elif hasattr(fig, 'write_html'): # Check if it's a Plotly figure object
            fig.write_html(filepath)
            logging.info(f"Plotly plot saved to {filepath}")
        # Handle case where fig might be Matplotlib's pyplot module itself (if plt.gcf() was passed)
        elif fig.__name__ == 'matplotlib.pyplot':
             fig.savefig(filepath, bbox_inches='tight')
             logging.info(f"Matplotlib plot saved to {filepath}")
             fig.close() # Close the current figure associated with pyplot
        else:
             logging.warning(f"Unsupported figure type for saving: {type(fig)}")
    except Exception as e:
        logging.error(f"Failed to save plot {filename}: {e}", exc_info=True) # Add traceback

# --- Core Visualization Functions ---

def plot_coherence_perplexity(eval_df, filename="coherence_perplexity.png"):
    """Plots coherence and perplexity scores vs. number of topics."""
    if eval_df is None or eval_df.empty:
        logging.warning("Evaluation data is empty. Skipping coherence/perplexity plot.")
        return

    logging.info("Generating coherence and perplexity plot...")
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Coherence Plot
        axes[0].plot(eval_df['n_topics'], eval_df['coherence'], marker='o')
        axes[0].set_title('Coherence Score vs. Number of Topics')
        axes[0].set_xlabel('Number of Topics')
        axes[0].set_ylabel('Coherence Score (c_v)')
        axes[0].grid(True)

        # Perplexity Plot
        axes[1].plot(eval_df['n_topics'], eval_df['perplexity'], marker='o', color='orange')
        axes[1].set_title('Perplexity vs. Number of Topics')
        axes[1].set_xlabel('Number of Topics')
        axes[1].set_ylabel('Log Perplexity')
        axes[1].grid(True)

        plt.tight_layout()
        # Pass the figure object explicitly
        save_plot(fig, filename)
    except Exception as e:
         logging.error(f"Failed to generate coherence/perplexity plot: {e}", exc_info=True)
         if 'fig' in locals() and isinstance(fig, plt.Figure):
              plt.close(fig) # Ensure plot is closed on error

def visualize_pyldavis(lda_model, bow_matrix, vectorizer, filename="pyldavis_visualization.html"):
    """Generates and saves the pyLDAvis interactive visualization."""
    if lda_model is None or bow_matrix is None or vectorizer is None:
        logging.warning("Missing model, BoW, or vectorizer. Skipping pyLDAvis generation.")
        return
    logging.info("Preparing pyLDAvis visualization...")
    try:
        # Note: pyLDAvis.enable_notebook() is only for Jupyter notebooks.
        # Use n_jobs=1 if t-SNE fails with parallelism or causes issues
        panel = pyLDAvis.prepare(lda_model, bow_matrix, vectorizer, mds='tsne', n_jobs=1)
        filepath = os.path.join(config.PLOT_DIR, filename)
        pyLDAvis.save_html(panel, filepath)
        logging.info(f"pyLDAvis visualization saved to {filepath}")
    except ImportError:
        logging.error("pyLDAvis library not found. Cannot generate visualization. Please install it (`pip install pyldavis`).")
    except Exception as e:
        logging.error(f"Failed to generate or save pyLDAvis visualization: {e}", exc_info=True)


def plot_topic_prevalence_timeline(df, filename="topic_prevalence_timeline.html"):
    """Plots topic document count over time."""
    logging.info("Generating topic prevalence timeline...")
    if not isinstance(df, pd.DataFrame) or df.empty:
        logging.warning("Input DataFrame is invalid or empty. Skipping prevalence timeline.")
        return
    required_cols = ['topic_name', 'time']
    if not all(col in df.columns for col in required_cols):
         logging.warning(f"Missing required columns ({required_cols}). Skipping prevalence timeline.")
         return
    try:
        df_copy = df.copy()
        # Handle potential errors during datetime conversion
        df_copy['time'] = pd.to_datetime(df_copy['time'], errors='coerce')
        df_copy.dropna(subset=['time'], inplace=True) # Remove rows where time conversion failed
        if df_copy.empty:
             logging.warning("No valid time data after conversion. Skipping prevalence timeline.")
             return

        # Suppress the specific UserWarning about timezone loss if desired
        with pd.option_context('mode.chained_assignment', None): # Suppress SettingWithCopyWarning if any
            import warnings
            # Temporarily ignore the specific UserWarning from dt.to_period
            warnings.filterwarnings("ignore", message="Converting to PeriodArray/Index representation will drop timezone information.")
            df_copy['Month'] = df_copy['time'].dt.to_period('M').dt.to_timestamp()
            warnings.resetwarnings() # Reset warnings filter

        timeline_data = df_copy.groupby(['Month', 'topic_name'], observed=True).size().reset_index(name='document_count')

        fig = px.area(
            timeline_data, x='Month', y='document_count', color='topic_name',
            title='Topic Prevalence Over Months',
            labels={'document_count': 'Number of Posts', 'topic_name': 'Topics', 'Month': 'Month'}, # Label x-axis too
            height=config.VIZ_HEIGHT_STANDARD, width=config.VIZ_WIDTH
        )
        fig.update_layout(
            xaxis_title='Month', yaxis_title='Number of Posts', legend_title='Topics',
            hovermode='x unified',
            xaxis=dict(tickformat='%b %Y', dtick='M1', ticklabelmode='period')
        )
        save_plot(fig, filename)
    except Exception as e:
        logging.error(f"Failed to generate prevalence timeline: {e}", exc_info=True)


def plot_hourly_topic_activity(df, filename="hourly_topic_activity.html"):
    """Plots hourly topic document count (stacked bar)."""
    logging.info("Generating hourly topic activity plot...")
    if not isinstance(df, pd.DataFrame) or df.empty:
        logging.warning("Input DataFrame is invalid or empty. Skipping hourly activity plot.")
        return
    required_cols = ['topic_name', 'time']
    if not all(col in df.columns for col in required_cols):
         logging.warning(f"Missing required columns ({required_cols}). Skipping hourly activity plot.")
         return
    try:
        df_copy = df.copy()
        # Handle potential errors during datetime conversion
        df_copy['time'] = pd.to_datetime(df_copy['time'], errors='coerce')
        df_copy.dropna(subset=['time'], inplace=True) # Remove rows where time conversion failed
        if df_copy.empty:
             logging.warning("No valid time data after conversion. Skipping hourly activity plot.")
             return

        df_copy['hour'] = df_copy['time'].dt.hour
        hourly_data = df_copy.groupby(['hour', 'topic_name'], observed=True).size().reset_index(name='count')

        fig = px.bar(
            hourly_data, x='hour', y='count', color='topic_name',
            title='Hourly Topic Activity Patterns',
            labels={'hour': 'Hour of Day', 'count': 'Number of Posts', 'topic_name': 'Topics'},
            height=config.VIZ_HEIGHT_STANDARD, width=config.VIZ_WIDTH, barmode='stack'
        )
        fig.update_layout(
            xaxis=dict(tickmode='array', tickvals=list(range(24)), ticktext=[f"{h:02d}:00" for h in range(24)], title='Hour of Day (24h)'),
            yaxis_title='Number of Posts', legend_title='Topics', hovermode='x unified'
        )
        save_plot(fig, filename)
    except Exception as e:
         logging.error(f"Failed to generate hourly activity plot: {e}", exc_info=True)


def plot_topic_network_graph(topic_keywords, topic_names, filename="topic_network_graph.html"):
    """Creates a network graph of topics based on shared top words."""
    logging.info("Generating topic network graph...")
    # Validate inputs
    if not topic_keywords or not isinstance(topic_keywords, list):
         logging.warning("Invalid or empty topic_keywords provided. Skipping network graph.")
         return
    if not topic_names or not isinstance(topic_names, list):
         logging.warning("Invalid or empty topic_names provided. Skipping network graph.")
         return
    if len(topic_keywords) != len(topic_names):
        logging.warning(f"Mismatch between number of keywords ({len(topic_keywords)}) and names ({len(topic_names)}). Skipping network graph.")
        return

    try:
        # Ensure names are strings for consistency
        topic_names_str = [str(name) for name in topic_names]
        topic_words = {name: set(keywords) for name, keywords in zip(topic_names_str, topic_keywords)}

        G = nx.Graph()
        for topic in topic_names_str:
            G.add_node(topic) # Add nodes using the string names

        # Add edges based on shared words
        for (topic1, words1), (topic2, words2) in combinations(topic_words.items(), 2):
            shared_words = words1.intersection(words2)
            if shared_words:
                shared_words_str = ', '.join(sorted(list(shared_words))[:5]) + ('...' if len(shared_words) > 5 else '')
                G.add_edge(topic1, topic2, weight=len(shared_words), shared_words_info=f"{len(shared_words)} shared: {shared_words_str}")

        if not G.nodes():
             logging.warning("Graph has no nodes after processing. Skipping network plot.")
             return

        # Calculate layout (can fail if graph is disconnected, add error handling)
        try:
             pos = nx.spring_layout(G, k=0.6, iterations=50, seed=42) # Adjust k if needed
        except nx.NetworkXError as ne:
             logging.warning(f"NetworkX layout failed (possibly disconnected graph): {ne}. Using random layout as fallback.")
             pos = nx.random_layout(G, seed=42)


        # Prepare edge data for Plotly
        edge_x, edge_y, edge_hover_text = [], [], []
        for edge in G.edges(data=True):
            # Check if nodes exist in pos (robustness for layout failures)
            if edge[0] in pos and edge[1] in pos:
                 x0, y0 = pos[edge[0]]
                 x1, y1 = pos[edge[1]]
                 edge_x.extend([x0, x1, None])
                 edge_y.extend([y0, y1, None])
                 edge_hover_text.append(edge[2].get('shared_words_info', '')) # Use .get for safety
            else:
                 logging.debug(f"Skipping edge {edge[:2]} due to missing node position.")


        # Prepare node data for Plotly
        node_x = [pos[node][0] for node in G.nodes() if node in pos]
        node_y = [pos[node][1] for node in G.nodes() if node in pos]
        # Filter nodes for adjacencies and text based on those in pos
        valid_nodes = [node for node in G.nodes() if node in pos]
        node_adjacencies = [G.degree(node) for node in valid_nodes]
        node_text = [f"{node}<br>Connections: {adj}" for node, adj in zip(valid_nodes, node_adjacencies)]
        node_size = [10 + 3 * adj for adj in node_adjacencies]


        # Create Plotly traces
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'),
            hoverinfo='text', hovertext=edge_hover_text, mode='lines')

        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text,
            marker=dict(
                showscale=True, colorscale='YlGnBu', reversescale=True, color=node_adjacencies,
                size=node_size,
                # --- CORRECTED colorbar title property ---
                colorbar=dict(
                    thickness=15,
                    title=dict(
                        text='Node Connections', # Title text
                        side='right'            # Position of the title
                    )
                ),
                # --- END CORRECTION ---
                line_width=1))

        # Create Figure
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            # --- FIX: Correct way to set title font size ---
                            title=dict(
                                text='Topic Network Graph (Shared Keywords)',
                                font=dict(
                                    size=16 # Set size within title's font dict
                                )
                            ),
                            # --- END FIX ---
                            showlegend=False, hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            height=config.VIZ_HEIGHT_TALL, width=config.VIZ_WIDTH,
                            plot_bgcolor='white'
                            ))
        save_plot(fig, filename)
    except Exception as e:
        # Log full traceback for better debugging
        logging.error(f"Failed to generate topic network graph: {e}", exc_info=True)


def plot_engagement_per_topic(df, filename="engagement_per_topic.html"):
    """Plots total engagement sum per topic."""
    logging.info("Generating engagement per topic bar chart...")
    if not isinstance(df, pd.DataFrame) or df.empty:
        logging.warning("Input DataFrame is invalid or empty. Skipping engagement per topic plot.")
        return
    required_cols = ['topic_name', 'engagements']
    if not all(col in df.columns for col in required_cols):
        logging.warning(f"Missing required columns ({required_cols}). Skipping engagement per topic plot.")
        return
    try:
        # Ensure engagements are numeric before grouping
        df['engagements'] = pd.to_numeric(df['engagements'], errors='coerce')
        df_filtered = df.dropna(subset=['engagements', 'topic_name'])

        if df_filtered.empty:
             logging.warning("No valid data after filtering NAs for engagement plot.")
             return

        topic_engagement = df_filtered.groupby('topic_name', observed=True, as_index=False)['engagements'].sum()
        topic_engagement = topic_engagement.sort_values(by='engagements', ascending=False)

        fig = px.bar(
            topic_engagement, x='topic_name', y='engagements',
            title='Total Engagements per Topic',
            labels={'topic_name': 'Topic Name', 'engagements': 'Total Engagements'},
            color='topic_name', # Color by topic name might be too much if many topics
            height=config.VIZ_HEIGHT_STANDARD, width=config.VIZ_WIDTH
        )
        fig.update_layout(
            xaxis_title='Topic', yaxis_title='Total Engagements', xaxis_tickangle=-45,
            showlegend=False, # Legend is redundant if coloring by x-axis category
            xaxis={'categoryorder':'total descending'}
        )
        save_plot(fig, filename)
    except Exception as e:
        logging.error(f"Failed to generate engagement per topic plot: {e}", exc_info=True)


def plot_engagement_timeline(df, filename="engagement_timeline.html"):
    """Plots total monthly engagement per topic over time."""
    logging.info("Generating engagement timeline plot...")
    if not isinstance(df, pd.DataFrame) or df.empty:
        logging.warning("Input DataFrame is invalid or empty. Skipping engagement timeline.")
        return
    required_cols = ['topic_name', 'engagements', 'time']
    if not all(col in df.columns for col in required_cols):
        logging.warning(f"Missing required columns ({required_cols}). Skipping engagement timeline.")
        return
    try:
        df_copy = df.copy()
        # Ensure numeric engagements and valid time
        df_copy['engagements'] = pd.to_numeric(df_copy['engagements'], errors='coerce')
        df_copy['time'] = pd.to_datetime(df_copy['time'], errors='coerce')
        df_copy.dropna(subset=['engagements', 'time', 'topic_name'], inplace=True)
        if df_copy.empty:
             logging.warning("No valid data after filtering NAs for engagement timeline.")
             return

        # Suppress the specific UserWarning about timezone loss if desired
        with pd.option_context('mode.chained_assignment', None): # Suppress SettingWithCopyWarning if any
             import warnings
             warnings.filterwarnings("ignore", message="Converting to PeriodArray/Index representation will drop timezone information.")
             df_copy['Month'] = df_copy['time'].dt.to_period('M').dt.to_timestamp()
             warnings.resetwarnings() # Reset warnings filter


        engagement_timeline_data = df_copy.groupby(['Month', 'topic_name'], observed=True, as_index=False)['engagements'].sum()
        engagement_timeline_data = engagement_timeline_data.rename(columns={'engagements': 'total_monthly_engagements'})

        fig = px.area(
            engagement_timeline_data, x='Month', y='total_monthly_engagements', color='topic_name',
            title='Total Monthly Engagements per Topic Over Time',
            labels={'total_monthly_engagements': 'Total Engagements', 'topic_name': 'Topics', 'Month': 'Month'},
            height=config.VIZ_HEIGHT_STANDARD, width=config.VIZ_WIDTH
        )
        fig.update_layout(
            xaxis_title='Month', yaxis_title='Total Engagements per Month', legend_title='Topics',
            hovermode='x unified', xaxis=dict(tickformat='%b %Y', dtick='M1', ticklabelmode='period')
        )
        save_plot(fig, filename)
    except Exception as e:
        logging.error(f"Failed to generate engagement timeline plot: {e}", exc_info=True)


def plot_hourly_avg_engagement(df, filename="hourly_avg_engagement.html"):
    """Plots average hourly engagement per topic."""
    logging.info("Generating hourly average engagement line chart...")
    if not isinstance(df, pd.DataFrame) or df.empty:
        logging.warning("Input DataFrame is invalid or empty. Skipping hourly avg engagement plot.")
        return
    required_cols = ['topic_name', 'engagements', 'time']
    if not all(col in df.columns for col in required_cols):
        logging.warning(f"Missing required columns ({required_cols}). Skipping hourly avg engagement plot.")
        return
    try:
        df_copy = df.copy()
        # Ensure numeric engagements and valid time
        df_copy['engagements'] = pd.to_numeric(df_copy['engagements'], errors='coerce')
        df_copy['time'] = pd.to_datetime(df_copy['time'], errors='coerce')
        df_copy.dropna(subset=['engagements', 'time', 'topic_name'], inplace=True)
        if df_copy.empty:
             logging.warning("No valid data after filtering NAs for hourly avg engagement plot.")
             return

        df_copy['hour'] = df_copy['time'].dt.hour
        hourly_avg_engagement_data = df_copy.groupby(['hour', 'topic_name'], observed=True, as_index=False)['engagements'].mean()
        hourly_avg_engagement_data = hourly_avg_engagement_data.rename(columns={'engagements': 'avg_hourly_engagement'})

        fig = px.line(
            hourly_avg_engagement_data, x='hour', y='avg_hourly_engagement', color='topic_name',
            title='Average Hourly Engagement per Topic',
            labels={'avg_hourly_engagement': 'Average Engagement', 'topic_name': 'Topics', 'hour': 'Hour of Day'},
            markers=True, height=config.VIZ_HEIGHT_STANDARD, width=config.VIZ_WIDTH
        )
        fig.update_layout(
            xaxis=dict(tickmode='array', tickvals=list(range(24)), ticktext=[f"{h:02d}:00" for h in range(24)], title='Hour of Day (24h format)'),
            yaxis_title='Average Engagement per Post', legend_title='Topics', hovermode='x unified'
        )
        save_plot(fig, filename)
    except Exception as e:
        logging.error(f"Failed to generate hourly avg engagement plot: {e}", exc_info=True)


def plot_engagement_distribution(df, filename="engagement_distribution_boxplot.html"):
    """Plots box plot distribution of engagement scores per topic."""
    logging.info("Generating engagement distribution box plot...")
    if not isinstance(df, pd.DataFrame) or df.empty:
        logging.warning("Input DataFrame is invalid or empty. Skipping engagement distribution plot.")
        return
    required_cols = ['topic_name', 'engagements']
    if not all(col in df.columns for col in required_cols):
        logging.warning(f"Missing required columns ({required_cols}). Skipping engagement distribution plot.")
        return
    try:
        df_copy = df.copy()
        # Ensure numeric engagements
        df_copy['engagements'] = pd.to_numeric(df_copy['engagements'], errors='coerce')
        df_copy.dropna(subset=['engagements', 'topic_name'], inplace=True)
        if df_copy.empty:
             logging.warning("No valid data after filtering NAs for engagement distribution plot.")
             return

        fig = px.box(
            df_copy, x='topic_name', y='engagements',
            title='Distribution of Engagement Scores per Topic',
            labels={'topic_name': 'Topic', 'engagements': 'Engagement Score'},
            color='topic_name', points='outliers', # Show outliers
            height=config.VIZ_HEIGHT_STANDARD, width=config.VIZ_WIDTH
        )
        # Order topics by median engagement descending for better readability
        fig.update_layout(xaxis_tickangle=-45, showlegend=False, xaxis={'categoryorder':'median descending'})
        save_plot(fig, filename)
    except Exception as e:
        logging.error(f"Failed to generate engagement distribution plot: {e}", exc_info=True)


def plot_activity_heatmap(df, filename="activity_heatmap.html"):
    """Plots heatmap of post count by day of week and hour."""
    logging.info("Generating activity heatmap (Day vs Hour)...")
    if not isinstance(df, pd.DataFrame) or df.empty:
        logging.warning("Input DataFrame is invalid or empty. Skipping activity heatmap.")
        return
    required_cols = ['time']
    if not all(col in df.columns for col in required_cols):
        logging.warning(f"Missing required columns ({required_cols}). Skipping activity heatmap.")
        return
    try:
        df_copy = df.copy()
        df_copy['datetime'] = pd.to_datetime(df_copy['time'], errors='coerce')
        df_copy.dropna(subset=['datetime'], inplace=True)
        if df_copy.empty:
             logging.warning("No valid datetime data after conversion for activity heatmap.")
             return

        df_copy['day_of_week'] = df_copy['datetime'].dt.day_name()
        df_copy['hour'] = df_copy['datetime'].dt.hour
        # Define order for plotting
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        df_copy['day_of_week'] = pd.Categorical(df_copy['day_of_week'], categories=day_order, ordered=True)

        # Use observed=False if you want to ensure all day/hour slots appear even with 0 count
        activity_heatmap_data = df_copy.groupby(['day_of_week', 'hour'], observed=False).size().reset_index(name='post_count')

        fig = px.density_heatmap(
            activity_heatmap_data, x='hour', y='day_of_week', z='post_count',
            title='Hourly Activity Pattern by Day of Week (Post Count)',
            labels={'hour': 'Hour of Day (0-23)', 'day_of_week': 'Day of Week', 'post_count': 'Number of Posts'},
            color_continuous_scale='Viridis', # Example color scale
            height=config.VIZ_HEIGHT_STANDARD * 0.8, width=config.VIZ_WIDTH*0.8 # Adjust size if needed
        )
        fig.update_layout(
             xaxis=dict(tickmode='array', tickvals=list(range(0,24,2)), ticktext=[f"{h:02d}" for h in range(0,24,2)]), # Show every 2 hours
             yaxis={'categoryorder':'array', 'categoryarray':day_order} # Ensure correct day order
        )
        save_plot(fig, filename)
    except Exception as e:
        logging.error(f"Failed to generate activity heatmap: {e}", exc_info=True)


def plot_engagement_heatmap(df, filename="engagement_heatmap.html"):
    """Plots heatmap of average engagement by day of week and hour."""
    logging.info("Generating engagement heatmap (Day vs Hour)...")
    if not isinstance(df, pd.DataFrame) or df.empty:
        logging.warning("Input DataFrame is invalid or empty. Skipping engagement heatmap.")
        return
    required_cols = ['time', 'engagements']
    if not all(col in df.columns for col in required_cols):
        logging.warning(f"Missing required columns ({required_cols}). Skipping engagement heatmap.")
        return
    try:
        df_copy = df.copy()
        df_copy['datetime'] = pd.to_datetime(df_copy['time'], errors='coerce')
        df_copy['engagements'] = pd.to_numeric(df_copy['engagements'], errors='coerce')
        df_copy.dropna(subset=['datetime', 'engagements'], inplace=True)
        if df_copy.empty:
             logging.warning("No valid datetime/engagement data after conversion/filtering for engagement heatmap.")
             return

        df_copy['day_of_week'] = df_copy['datetime'].dt.day_name()
        df_copy['hour'] = df_copy['datetime'].dt.hour
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        df_copy['day_of_week'] = pd.Categorical(df_copy['day_of_week'], categories=day_order, ordered=True)

        # Calculate mean engagement, fill missing day/hour combos with NaN or 0
        engagement_heatmap_data = df_copy.groupby(['day_of_week', 'hour'], observed=False)['engagements'].mean().reset_index(name='avg_engagement')
        # Optional: fill NaN with 0 if you want 0 avg engagement shown for empty slots
        # engagement_heatmap_data['avg_engagement'] = engagement_heatmap_data['avg_engagement'].fillna(0)

        fig = px.density_heatmap(
            engagement_heatmap_data, x='hour', y='day_of_week', z='avg_engagement',
            title='Average Engagement by Hour and Day of Week',
            labels={'hour': 'Hour of Day (0-23)', 'day_of_week': 'Day of Week', 'avg_engagement': 'Average Engagement'},
            color_continuous_scale='Plasma', # Example color scale
            text_auto='.1f', # Display avg engagement value on cells, formatted
            height=config.VIZ_HEIGHT_STANDARD*0.8, width=config.VIZ_WIDTH*0.8
        )
        fig.update_layout(
             xaxis=dict(tickmode='array', tickvals=list(range(0,24,2)), ticktext=[f"{h:02d}" for h in range(0,24,2)]),
             yaxis={'categoryorder':'array', 'categoryarray':day_order}
        )
        # Update color scale midpoint if desired, e.g., center around overall mean
        # overall_mean_engagement = df_copy['engagements'].mean()
        # fig.update_layout(coloraxis_midpoint=overall_mean_engagement)
        save_plot(fig, filename)
    except Exception as e:
        logging.error(f"Failed to generate engagement heatmap: {e}", exc_info=True)


def plot_word_clouds(df, filename_prefix="wordcloud_topic_", num_topics=None):
    """Generates a combined plot of word clouds for each topic."""
    logging.info("Generating word clouds for each topic...")
    if not isinstance(df, pd.DataFrame) or df.empty:
        logging.warning("Input DataFrame is invalid or empty. Skipping word clouds.")
        return
    required_cols = ['topic_name', 'cleaned_text']
    if not all(col in df.columns for col in required_cols):
         logging.warning(f"Missing required columns ({required_cols}). Skipping word clouds.")
         return

    try:
        # Group text ensuring topic_name and cleaned_text are not NaN/None
        df_filtered = df.dropna(subset=['topic_name', 'cleaned_text'])
        # Aggregate non-empty cleaned text
        topic_texts = df_filtered[df_filtered['cleaned_text'].str.strip() != ''].groupby('topic_name', observed=True)['cleaned_text'].apply(lambda x: ' '.join(x))

        if topic_texts.empty:
             logging.warning("No text data found for any topic after grouping/filtering. Skipping word clouds.")
             return

        if num_topics is None:
            num_topics = len(topic_texts)
        else:
            num_topics = min(num_topics, len(topic_texts)) # Don't request more than available

        # Dynamically adjust layout
        cols = 3
        rows = (num_topics + cols - 1) // cols
        fig_height = max(5, rows * 4) # Adjust height based on rows
        fig_width = 15

        fig = plt.figure(figsize=(fig_width, fig_height))

        # Iterate through sorted topics for consistent order if desired
        sorted_topics = topic_texts.sort_index()

        plot_index = 1
        for topic_name, text in sorted_topics.items():
            if plot_index > num_topics: # Stop if we exceed requested/available topics
                 break
            # Skip if text is effectively empty after join (should be caught by earlier filter)
            if not text.strip():
                 logging.warning(f"Skipping word cloud for topic '{topic_name}' due to empty text.")
                 continue

            # Sanitize topic name for filename (optional, useful if saving individually)
            # safe_topic_name = re.sub(r'[^\w\-\s]+', '_', topic_name).replace(' ', '_') # Allow spaces/hyphens
            # individual_filename = f"{filename_prefix}{plot_index-1}_{safe_topic_name}.png"

            try:
                # Adjust WordCloud parameters if needed
                wc = WordCloud(width=600, height=300, background_color='white',
                               max_words=50, # Limit number of words
                               collocations=False, # Avoid bigrams being treated as single words
                               prefer_horizontal=0.9 # Prefer horizontal layout
                               ).generate(text)

                # Plotting on subplots
                ax = fig.add_subplot(rows, cols, plot_index)
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                # Truncate long topic names for title
                display_name = (topic_name[:40] + '...') if len(topic_name) > 40 else topic_name
                ax.set_title(f"{display_name}", fontsize=10)

                plot_index += 1

            except ValueError as ve:
                 # This can happen if text contains only stopwords or is too short
                 logging.warning(f"Could not generate word cloud for topic '{topic_name}': {ve}. Skipping.")
                 ax = fig.add_subplot(rows, cols, plot_index)
                 ax.text(0.5, 0.5, 'No words\nto display', ha='center', va='center', fontsize=9)
                 ax.axis('off')
                 ax.set_title(f"{topic_name} (Empty/Error)", fontsize=10)
                 plot_index += 1
            except Exception as inner_e:
                 logging.error(f"Unexpected error generating word cloud for topic '{topic_name}': {inner_e}", exc_info=True)
                 # Still add a placeholder subplot
                 ax = fig.add_subplot(rows, cols, plot_index)
                 ax.text(0.5, 0.5, 'Error', ha='center', va='center', fontsize=9)
                 ax.axis('off')
                 ax.set_title(f"{topic_name} (Error)", fontsize=10)
                 plot_index += 1


        # Adjust layout if fewer plots were made than grid size
        plt.tight_layout(pad=2.0)

        # Save the combined figure
        combined_filename = f"{filename_prefix}combined.png"
        # Pass the figure object itself to save_plot
        save_plot(fig, combined_filename)

    except Exception as e:
        logging.error(f"Failed during overall word cloud generation: {e}", exc_info=True)
        # Ensure figure is closed if created before error
        if 'fig' in locals() and isinstance(fig, plt.Figure):
             plt.close(fig)


def plot_correlation_heatmap(df, filename="topic_engagement_correlation.html"):
    """Plots correlation matrix heatmap based on daily topic engagements."""
    logging.info("Generating topic engagement correlation heatmap...")
    if not isinstance(df, pd.DataFrame) or df.empty:
        logging.warning("Input DataFrame is invalid or empty. Skipping correlation heatmap.")
        return
    required_cols = ['topic_name', 'engagements', 'time']
    if not all(col in df.columns for col in required_cols):
        logging.warning(f"Missing required columns ({required_cols}). Skipping correlation heatmap.")
        return
    try:
        df_copy = df.copy()
        # Ensure required columns have correct types and filter NAs
        df_copy['engagements'] = pd.to_numeric(df_copy['engagements'], errors='coerce')
        df_copy['time'] = pd.to_datetime(df_copy['time'], errors='coerce')
        df_copy.dropna(subset=['time', 'topic_name', 'engagements'], inplace=True)
        if df_copy.empty:
             logging.warning("No valid data after filtering NAs for correlation heatmap.")
             return

        # Aggregate engagement per topic per day
        df_copy['Date'] = df_copy['time'].dt.date
        daily_topic_engagement = df_copy.groupby(['Date', 'topic_name'], observed=True, as_index=False)['engagements'].sum()

        # Pivot data: topics as columns, dates as index
        engagement_pivot = daily_topic_engagement.pivot(index='Date', columns='topic_name', values='engagements')
        engagement_pivot_filled = engagement_pivot.fillna(0) # Fill days with no posts/engagement with 0

        # Need at least two topics (columns) to calculate correlation
        if engagement_pivot_filled.shape[1] < 2:
            logging.warning("Need at least two topics with data to calculate correlation. Skipping heatmap.")
            return

        correlation_matrix = engagement_pivot_filled.corr(method='pearson') # Specify method if needed

        fig = px.imshow(
            correlation_matrix,
            text_auto='.2f',  # Display correlation values formatted to 2 decimals
            aspect="auto",   # Adjust aspect ratio ('equal' makes it square)
            color_continuous_scale='RdBu_r', # Red-Blue diverging scale (-1 Blue, 0 White, +1 Red)
            range_color=[-1, 1], # Ensure color scale covers full range
            title='Topic Engagement Correlation Matrix (Daily Totals)'
        )
        fig.update_layout(
            height=max(600, engagement_pivot_filled.shape[1]*40), # Adjust height based on num topics
            width=max(700, engagement_pivot_filled.shape[1]*40 + 100), # Adjust width
            xaxis_title="Topics",
            yaxis_title="Topics",
            xaxis={'side': 'bottom', 'tickangle': -45}, # Move x-axis labels down and angle them
            yaxis={'tickangle': 0}
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        save_plot(fig, filename)
    except Exception as e:
        logging.error(f"Failed to generate correlation heatmap: {e}", exc_info=True)