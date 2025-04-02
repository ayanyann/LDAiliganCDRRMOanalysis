# analysis.py
import pandas as pd
import numpy as np
import logging
import config
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_topic_density(lda_model, document_topic_dist, train_df_topic_col):
    """Calculates top word concentration and average document assignment probability."""
    if lda_model is None:
        logging.error("LDA model is None. Cannot calculate density.")
        return None
    if document_topic_dist is None:
         logging.error("Document topic distribution is None. Cannot calculate density.")
         return None
    if train_df_topic_col is None:
         logging.error("Training DataFrame topic column is None. Cannot calculate density.")
         return None
    if not isinstance(train_df_topic_col, pd.Series):
         logging.error("train_df_topic_col must be a pandas Series.")
         return None

    logging.info("Calculating topic density metrics...")
    try:
        n_topics = lda_model.n_components
        n_top_words_for_metric = config.DENSITY_METRIC_TOP_WORDS

        # Metric 1: Top Word Probability Concentration
        topic_word_probs = lda_model.components_ / lda_model.components_.sum(axis=1)[:, np.newaxis]
        top_word_concentration = []
        for i in range(n_topics):
            # Sort probabilities in descending order for topic i
            sorted_probs = np.sort(topic_word_probs[i, :])[::-1]
            # Sum the probabilities of the top N words
            concentration_score = np.sum(sorted_probs[:n_top_words_for_metric])
            top_word_concentration.append(concentration_score)

        # Metric 2: Average Document Assignment Probability
        avg_doc_probs = []
        # Use the index from the passed Series, which should align with the original DataFrame
        doc_indices = train_df_topic_col.index

        # Check alignment: The number of rows in document_topic_dist MUST match
        # the number of documents used to generate it (usually len(train_bow.rows))
        # Direct comparison with len(train_df_topic_col.index) assumes the index hasn't been reset
        # A safer check might involve passing train_bow shape or re-transforming if unsure.
        # For now, assume document_topic_dist corresponds row-by-row to the input BOW matrix
        # from which train_df_topic_col was derived.

        if document_topic_dist.shape[0] != len(train_df_topic_col):
             # If lengths don't match, it's ambiguous which rows correspond.
             # This could happen if train_df was filtered *after* BOW creation but before this function.
             logging.error(f"Mismatch between document_topic_dist rows ({document_topic_dist.shape[0]}) "
                           f"and train_df_topic_col length ({len(train_df_topic_col)}). "
                           "Cannot reliably calculate avg assignment probability. Ensure alignment.")
             # Fill with NaN or return None, as calculation is unsafe
             avg_doc_probs = [np.nan] * n_topics
             # Or uncomment the line below to stop the process here:
             # return None
        else:
            # Alignment seems plausible, proceed
            topic_assignment_series = train_df_topic_col # Use the provided topic assignments

            for i in range(n_topics):
                # Find the indices (positions, 0 to N-1) in the document_topic_dist matrix
                # that correspond to documents assigned to topic 'i'
                # We use .iloc-based indexing on the underlying numpy array
                docs_assigned_mask = (topic_assignment_series.values == i)

                if not np.any(docs_assigned_mask):
                    # No documents assigned to this topic
                    avg_doc_probs.append(0.0)
                    continue

                # Select the full probability distributions for these documents
                doc_topic_probs_for_topic_docs = document_topic_dist[docs_assigned_mask, :]
                # Get the probability of belonging *specifically* to topic 'i' for these docs
                probs_for_topic_i = doc_topic_probs_for_topic_docs[:, i]

                # Calculate the average of these probabilities
                avg_prob = np.mean(probs_for_topic_i)
                avg_doc_probs.append(avg_prob)

        # Combine results
        topic_density_df = pd.DataFrame({
            'topic_id': list(range(n_topics)),
            f'top_{n_top_words_for_metric}_word_concentration': top_word_concentration,
            'avg_doc_assignment_prob': avg_doc_probs
        })
        # Fill only NaN values resulting from the calculation (e.g., if alignment failed)
        topic_density_df.fillna(0, inplace=True)
        logging.info("Topic density metrics calculated.")
        return topic_density_df

    except AttributeError as ae:
         logging.error(f"AttributeError during density calculation (check model/inputs): {ae}", exc_info=True)
         return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during density calculation: {e}", exc_info=True)
        return None


def combine_and_categorize_metrics(train_df, topic_density_df, topic_names):
    """Combines volume (post count) and density metrics, and categorizes them."""
    if not isinstance(train_df, pd.DataFrame) or train_df.empty:
        logging.error("Invalid or empty train_df provided for combining metrics.")
        return None
    if 'topic' not in train_df.columns:
         logging.error("Missing 'topic' column in train_df for combining metrics.")
         return None
    if topic_density_df is None or topic_density_df.empty:
        logging.error("Missing or empty topic_density_df for combining metrics.")
        return None
    if not topic_names or not isinstance(topic_names, list):
        logging.error("Missing or invalid topic_names list for combining metrics.")
        return None
    if len(topic_names) != topic_density_df['topic_id'].max() + 1:
         logging.warning("Mismatch between number of topic names and topic IDs in density df.")
         # Proceed cautiously, mapping might be incomplete

    logging.info("Combining volume and density metrics...")
    try:
        # Create a copy to avoid modifying the original train_df
        train_df_copy = train_df.copy()

        # Calculate Topic Counts using the assigned 'topic' column
        # Map numeric topic ID to name first using the provided list
        num_provided_names = len(topic_names)
        topic_map = {i: topic_names[i] if i < num_provided_names else f"Unknown Topic #{i}"
                     for i in range(int(train_df_copy['topic'].max()) + 1)}

        train_df_copy['topic_name_mapped'] = train_df_copy['topic'].map(topic_map)

        # Group by the mapped topic name to get counts
        topic_counts = train_df_copy.groupby('topic_name_mapped', observed=True).size().reset_index(name='post_count')
        # Rename column for merging clarity
        topic_counts.rename(columns={'topic_name_mapped': 'topic_name'}, inplace=True)

        # Add topic names to density_df for merging using the same map
        density_df_named = topic_density_df.copy()
        density_df_named['topic_name'] = density_df_named['topic_id'].map(topic_map)

        # Merge volume and density based on the generated topic_name
        combined_metrics_df = pd.merge(topic_counts, density_df_named, on='topic_name', how='left')

        if combined_metrics_df.empty:
             logging.warning("Merging volume and density resulted in an empty DataFrame.")
             return combined_metrics_df # Return empty DF

        # --- Categorize Volume ---
        logging.info("Categorizing volume...")
        volume_labels = ['Low Volume', 'Medium Volume', 'High Volume']
        # Use pd.cut for fixed bins or pd.qcut for quantile-based bins
        # Using qcut requires handling cases with few unique values or topics
        try:
            unique_counts = combined_metrics_df['post_count'].nunique()
            num_categories = min(3, unique_counts) # Max 3 categories
            if num_categories > 1:
                combined_metrics_df['volume_category'] = pd.qcut(
                    combined_metrics_df['post_count'], q=num_categories,
                    labels=volume_labels[:num_categories], duplicates='drop'
                )
            elif num_categories == 1:
                combined_metrics_df['volume_category'] = 'Medium Volume' # Assign default if only one count value
            else: # Should not happen if df is not empty, but handle anyway
                 combined_metrics_df['volume_category'] = 'Unknown Volume'
        except ValueError as e:
            logging.warning(f"Could not assign volume categories using qcut: {e}. Assigning default 'Medium Volume'.")
            combined_metrics_df['volume_category'] = 'Medium Volume' # Fallback

        # --- Categorize Density ---
        logging.info("Categorizing density...")
        density_labels = ['Low Density', 'Medium Density', 'High Density']
        density_col = 'avg_doc_assignment_prob'
        if density_col in combined_metrics_df.columns:
            try:
                 unique_densities = combined_metrics_df[density_col].nunique()
                 num_categories = min(3, unique_densities)
                 if num_categories > 1:
                      combined_metrics_df['density_category'] = pd.qcut(
                         combined_metrics_df[density_col], q=num_categories,
                         labels=density_labels[:num_categories], duplicates='drop'
                     )
                 elif num_categories == 1:
                      combined_metrics_df['density_category'] = 'Medium Density'
                 else:
                      combined_metrics_df['density_category'] = 'Unknown Density'
            except ValueError as e:
                logging.warning(f"Could not assign density categories using qcut: {e}. Assigning default 'Medium Density'.")
                combined_metrics_df['density_category'] = 'Medium Density' # Fallback
            except Exception as e:
                 logging.error(f"An unexpected error occurred during density categorization: {e}", exc_info=True)
                 combined_metrics_df['density_category'] = 'Unknown Density' # Placeholder
        else:
            logging.warning(f"'{density_col}' column not found. Skipping density categorization.")
            combined_metrics_df['density_category'] = 'Unknown Density'

        # No need to drop temp column as it was created on a copy

        logging.info("Volume and density categorization complete.")
        return combined_metrics_df

    except Exception as e:
         logging.error(f"Error during metric combination/categorization: {e}", exc_info=True)
         return None


def display_grouped_topics(combined_metrics_df):
    """Prints topics grouped by volume and density categories."""
    if not isinstance(combined_metrics_df, pd.DataFrame) or combined_metrics_df.empty:
        logging.warning("Cannot display grouped topics: DataFrame is invalid or empty.")
        return
    # Check if categorization columns exist
    if 'volume_category' not in combined_metrics_df.columns or 'density_category' not in combined_metrics_df.columns:
        logging.warning("Cannot display grouped topics due to missing category columns ('volume_category' or 'density_category').")
        return

    print("\n--- Topics Grouped by Volume and Density ---")
    # Define the desired display order
    volume_order = ['High Volume', 'Medium Volume', 'Low Volume']
    density_order = ['High Density', 'Medium Density', 'Low Density', 'Unknown Density'] # Include Unknown

    # Ensure category columns are treated as categorical with the defined order for proper sorting/iteration
    if isinstance(combined_metrics_df['volume_category'].dtype, pd.CategoricalDtype):
         combined_metrics_df['volume_category'] = combined_metrics_df['volume_category'].cat.set_categories(volume_order, ordered=True)
    else:
         combined_metrics_df['volume_category'] = pd.Categorical(combined_metrics_df['volume_category'], categories=volume_order, ordered=True)

    if isinstance(combined_metrics_df['density_category'].dtype, pd.CategoricalDtype):
          combined_metrics_df['density_category'] = combined_metrics_df['density_category'].cat.set_categories(density_order, ordered=True)
    else:
         combined_metrics_df['density_category'] = pd.Categorical(combined_metrics_df['density_category'], categories=density_order, ordered=True)


    # Iterate through the defined order
    for vol_cat in volume_order:
        # Check if this volume category actually exists in the data
        if vol_cat not in combined_metrics_df['volume_category'].cat.categories: continue
        for den_cat in density_order:
             # Check if this density category actually exists
            if den_cat not in combined_metrics_df['density_category'].cat.categories: continue

            # Filter for the specific combination
            filtered_topics = combined_metrics_df[
                (combined_metrics_df['volume_category'] == vol_cat) &
                (combined_metrics_df['density_category'] == den_cat)
            ].sort_values(by='post_count', ascending=False) # Sort by post count within group

            # Only print header if there are topics in this group OR if you want to show empty groups
            if not filtered_topics.empty:
                print(f"\n{vol_cat} / {den_cat}:")
                print("-" * (len(str(vol_cat)) + len(str(den_cat)) + 3)) # Adjust length based on actual category names
                for _, row in filtered_topics.iterrows():
                    # Use .get() for safe access in case columns were somehow dropped
                    post_count_str = str(row.get('post_count', 'N/A'))
                    density_score = row.get('avg_doc_assignment_prob', float('nan'))
                    # Format density score nicely
                    density_score_str = f"{density_score:.3f}" if pd.notna(density_score) else "N/A"
                    topic_name_str = row.get('topic_name', 'Unknown Topic')
                    print(f"- {topic_name_str} (Posts: {post_count_str}, Density: {density_score_str})")
            #else: # Uncomment this if you want to explicitly show empty categories
            #    print(f"\n{vol_cat} / {den_cat}:")
            #    print("-" * (len(str(vol_cat)) + len(str(den_cat)) + 3))
            #    print("(None in this category)")

def save_analysis_results(eval_df, labeled_topics_df, combined_metrics_df, result_dir):
    """Saves various analysis results to CSV files."""
    logging.info(f"Saving analysis results to directory: {result_dir}")
    os.makedirs(result_dir, exist_ok=True) # Ensure directory exists

    try:
        # Save evaluation metrics (coherence, perplexity)
        if eval_df is not None and isinstance(eval_df, pd.DataFrame) and not eval_df.empty:
            eval_path = os.path.join(result_dir, 'lda_model_evaluation.csv')
            eval_df.to_csv(eval_path, index=False)
            logging.info(f"LDA evaluation results saved to {eval_path}")
        elif eval_df is not None:
             logging.warning("Evaluation data provided but is not a valid DataFrame or is empty. Skipping save.")

        # Save labeled topics and keywords
        if labeled_topics_df is not None and isinstance(labeled_topics_df, pd.DataFrame) and not labeled_topics_df.empty:
            topics_path = os.path.join(result_dir, 'best_model_topics_labeled.csv')
            labeled_topics_df.to_csv(topics_path, index=False)
            logging.info(f"Labeled topics saved to {topics_path}")
        elif labeled_topics_df is not None:
             logging.warning("Labeled topics data provided but is not a valid DataFrame or is empty. Skipping save.")

        # Save combined volume/density metrics
        if combined_metrics_df is not None and isinstance(combined_metrics_df, pd.DataFrame) and not combined_metrics_df.empty:
            metrics_path = os.path.join(result_dir, 'topic_volume_density_metrics.csv')
            # Define desired columns and order for the output CSV
            density_metric_col_name = f'top_{config.DENSITY_METRIC_TOP_WORDS}_word_concentration'
            cols_to_save = [
                'topic_name', 'topic_id', 'post_count', 'volume_category',
                'avg_doc_assignment_prob', 'density_category',
                density_metric_col_name # Use the dynamically generated name
            ]
            # Select only columns that actually exist in the DataFrame
            existing_cols = [col for col in cols_to_save if col in combined_metrics_df.columns]
            # Sort before saving
            combined_metrics_df[existing_cols].sort_values(
                 by=['volume_category', 'density_category', 'post_count'],
                 ascending=[True, True, False] # Sort order for categories and count
            ).to_csv(metrics_path, index=False)
            logging.info(f"Combined volume/density metrics saved to {metrics_path}")
        elif combined_metrics_df is not None:
             logging.warning("Combined metrics data provided but is not a valid DataFrame or is empty. Skipping save.")

    except Exception as e:
        logging.error(f"Error saving analysis results: {e}", exc_info=True)


def save_intermediate_data(data_to_save, data_name, result_dir):
     """Saves intermediate DataFrames or lists of dictionaries to CSV files."""
     # --- Corrected checks and handling for list/DataFrame ---
     if data_to_save is None: # Check for None first
         logging.warning(f"Data '{data_name}' is None. Skipping save.")
         return

     is_empty = False
     if isinstance(data_to_save, list):
          if not data_to_save: # Check if list is empty
               is_empty = True
     elif isinstance(data_to_save, pd.DataFrame):
          if data_to_save.empty: # Check if DataFrame is empty
               is_empty = True
     # Add checks for other types if needed, e.g., dictionaries, numpy arrays
     else:
          # Assume it might be some other iterable, check its length? Or just log warning.
          try:
               if not data_to_save: # Check if generic iterable is empty
                    is_empty = True
          except TypeError: # Handle non-iterable types
               pass # If it's not iterable, it's likely not empty in a meaningful way here
          logging.debug(f"Data '{data_name}' is of type {type(data_to_save)}, not list or DataFrame.")

     if is_empty:
          logging.warning(f"Data '{data_name}' is empty. Skipping save.")
          return
     # --- END FIX for empty check ---

     try:
        # Ensure result directory exists
        os.makedirs(result_dir, exist_ok=True)
        filepath = os.path.join(result_dir, f"{data_name}.csv")
        logging.debug(f"Attempting to save '{data_name}' to {filepath}")

        # --- Corrected Specific formatting based on data_name ---
        if data_name == 'topic_prevalence_wide_format':
            # Expects 'data_to_save' to be the LONG format DataFrame for pivoting
            if not isinstance(data_to_save, pd.DataFrame):
                 logging.error(f"Expected DataFrame for '{data_name}', got {type(data_to_save)}. Skipping save.")
                 return
            required_pivot_cols = ['Month', 'topic_name', 'document_count']
            if not all(col in data_to_save.columns for col in required_pivot_cols):
                 logging.error(f"Missing required columns for pivoting '{data_name}': {required_pivot_cols}. Have: {list(data_to_save.columns)}. Skipping.")
                 return
            logging.debug(f"Pivoting data for '{data_name}'. Columns: {list(data_to_save.columns)}")
            df_pivot = data_to_save.pivot_table(index='Month', columns='topic_name', values='document_count', fill_value=0).reset_index()
            df_pivot.sort_values('Month').to_csv(filepath, index=False)

        elif data_name == 'hourly_topic_activity_transposed':
            # Expects 'data_to_save' to be the LONG format DataFrame for pivoting
            if not isinstance(data_to_save, pd.DataFrame):
                 logging.error(f"Expected DataFrame for '{data_name}', got {type(data_to_save)}. Skipping save.")
                 return
            required_pivot_cols = ['hour', 'topic_name', 'count']
            if not all(col in data_to_save.columns for col in required_pivot_cols):
                 logging.error(f"Missing required columns for pivoting '{data_name}': {required_pivot_cols}. Have: {list(data_to_save.columns)}. Skipping.")
                 return
            logging.debug(f"Pivoting data for '{data_name}'. Columns: {list(data_to_save.columns)}")
            df_pivot = data_to_save.pivot(index='topic_name', columns='hour', values='count').fillna(0)
            # Ensure column names are suitable for formatting (should be integers)
            try:
                 df_pivot.columns = [f"{int(h):02d}:00" for h in df_pivot.columns]
                 df_pivot = df_pivot.reset_index()
                 # Sort columns numerically after 'topic_name'
                 time_cols = sorted([col for col in df_pivot.columns if col != 'topic_name'], key=lambda x: int(x.split(':')[0])) # Sort numerically
                 df_pivot = df_pivot[['topic_name'] + time_cols]
                 df_pivot.to_csv(filepath, index=False)
            except ValueError as ve:
                 logging.error(f"Error formatting hour columns for '{data_name}' (are they numeric?): {ve}. Columns: {list(df_pivot.columns)}. Skipping save.", exc_info=True)


        elif data_name == 'topic_relationships':
             # --- Corrected: Handle list directly ---
             # Expects 'data_to_save' to be the list of dictionaries
             if not isinstance(data_to_save, list):
                  logging.error(f"Expected list for '{data_name}', got {type(data_to_save)}. Skipping save.")
                  return
             # Check if list contains dictionaries before converting
             if data_to_save and not all(isinstance(item, dict) for item in data_to_save):
                  logging.error(f"List for '{data_name}' does not contain only dictionaries. Skipping save.")
                  return
             pd.DataFrame(data_to_save).to_csv(filepath, index=False)
             # --- END FIX ---

        else:
             # Default save assumes data_to_save is a DataFrame
             if isinstance(data_to_save, pd.DataFrame):
                 data_to_save.to_csv(filepath, index=False)
             else:
                 # Add more handling here if other types like simple lists need saving
                 logging.warning(f"Default save logic skipped for '{data_name}'; unsupported data type: {type(data_to_save)}")

        # Check if file was actually created (optional sanity check)
        if os.path.exists(filepath):
             logging.info(f"Intermediate data '{data_name}' saved successfully to {filepath}")
        else:
             # This case might occur if an error happened silently above or permissions issue
             logging.warning(f"File path {filepath} not found after attempting to save '{data_name}'.")


     except KeyError as ke:
         logging.error(f"Error saving intermediate data '{data_name}' due to KeyError: {ke}. Check required columns for pivoting/saving.")
         if isinstance(data_to_save, pd.DataFrame):
              logging.debug(f"Columns in DataFrame for '{data_name}' when error occurred: {list(data_to_save.columns)}")
     except Exception as e:
          logging.error(f"Error saving intermediate data '{data_name}': {e}", exc_info=True)