# main.py
import pandas as pd
import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split
from itertools import combinations

# Import project modules
import config
import data_loader
import text_preprocessor
import topic_modeler
import gemini_labeler
import visualizer
import analysis

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
                    handlers=[logging.StreamHandler()]) # Output logs to console

def main():
    """Main function to run the topic modeling pipeline."""
    logging.info("--- Starting Iligan CDRRMO Topic Modeling Pipeline ---")

    # --- 1. Load and Prepare Data ---
    df = data_loader.load_and_prepare_data(config.RAW_DATA_PATH)
    if df is None or df.empty:
        logging.error("Failed to load or prepare data. Exiting.")
        return
    # Save initial prepared data (optional)
    # analysis.save_intermediate_data(df, "prepared_data", config.RESULT_DIR)


    # --- 2. Preprocess Text ---
    df = text_preprocessor.apply_preprocessing(df, text_column='caption')
    # Remove rows with empty text after cleaning
    df.dropna(subset=['cleaned_text'], inplace=True)
    df = df[df['cleaned_text'].str.strip() != ''] # Ensure no empty strings remain
    if df.empty:
        logging.error("No data remaining after text preprocessing. Exiting.")
        return
    logging.info(f"Data shape after preprocessing and cleaning: {df.shape}")


    # --- 3. Train/Test Split ---
    logging.info("Splitting data into training and testing sets...")
    try:
        # Ensure enough data for split
        if len(df) < 10: # Adjust threshold as needed
             logging.warning("Dataset too small for train/test split. Using all data for training.")
             train_df = df.copy()
             test_df = df.copy() # Use same data for test - less ideal but avoids error
        else:
             train_df, test_df = train_test_split(df, test_size=0.2, random_state=42) # 80/20 split

        logging.info(f"Train set size: {len(train_df)}, Test set size: {len(test_df)}")
    except Exception as e:
         logging.error(f"Error during train/test split: {e}. Exiting.")
         return


    # --- 4. Create Bag-of-Words ---
    logging.info("Creating Bag-of-Words representation...")
    # Fit on training data only
    train_bow, vectorizer = topic_modeler.create_bow(
        train_df['cleaned_text'],
        config.LDA_MAX_FEATURES,
        config.LDA_NGRAM_RANGE
    )
    # Transform test data using the *same* vectorizer
    test_bow, _ = topic_modeler.create_bow(
        test_df['cleaned_text'],
        config.LDA_MAX_FEATURES,
        config.LDA_NGRAM_RANGE,
        fit_vectorizer=vectorizer # Use the fitted vectorizer
    )

    if train_bow is None or test_bow is None or vectorizer is None:
         logging.error("Failed to create Bag-of-Words. Exiting.")
         return


    # --- 5. Find Optimal LDA Model ---
    logging.info("Finding the best LDA model...")
    # Prepare texts in list-of-lists format for coherence calculation
    train_texts_for_coherence = [doc.split() for doc in train_df['cleaned_text']]

    best_lda_model, best_topics_keywords, best_n_topics, eval_df = topic_modeler.find_best_lda_model(
        train_bow,
        test_bow,
        train_texts_for_coherence,
        config.LDA_TOPIC_RANGE,
        vectorizer
    )

    if best_lda_model is None:
        logging.error("Failed to find a suitable LDA model. Exiting.")
        return

    # Save evaluation results
    analysis.save_analysis_results(eval_df, None, None, config.RESULT_DIR) # Save eval part first

    # Plot coherence/perplexity
    visualizer.plot_coherence_perplexity(eval_df)


    # --- 6. Label Topics using Gemini ---
    logging.info("Labeling topics using Gemini API...")
    topic_names = gemini_labeler.label_topics_with_gemini(best_topics_keywords)

    # Create and save labeled topics dataframe
    labeled_topics_df = pd.DataFrame({
        'topic_id': list(range(best_n_topics)),
        'topic_name_generated': topic_names,
        'top_keywords': [', '.join(keywords) for keywords in best_topics_keywords]
    })
    analysis.save_analysis_results(None, labeled_topics_df, None, config.RESULT_DIR) # Update with labeled topics


    # --- 7. Save Best Model and Vectorizer ---
    logging.info("Saving the best LDA model and vectorizer...")
    topic_modeler.save_model_and_vectorizer(best_lda_model, vectorizer, config.MODEL_DIR)

    # --- OPTIONAL: Load Model (Example) ---
    # lda_loaded, vec_loaded = topic_modeler.load_model_and_vectorizer(config.MODEL_DIR)
    # if lda_loaded and vec_loaded:
    #     logging.info("Model and vectorizer loaded successfully (example).")


    # --- 8. Assign Topics to Training Data ---
    # We only analyze/visualize based on the training data split
    dominant_topic_train, doc_topic_dist_train = topic_modeler.assign_topics_to_documents(best_lda_model, train_bow)

    if dominant_topic_train is None:
        logging.error("Failed to assign topics to documents. Some analysis/visualizations will be skipped.")
    else:
        train_df['topic'] = dominant_topic_train
        topic_map = {i: name for i, name in enumerate(topic_names)}
        train_df['topic_name'] = train_df['topic'].map(topic_map)
        logging.info("Topics assigned to training documents.")
        # Save train_df with topics (optional)
        analysis.save_intermediate_data(train_df, "train_data_with_topics", config.RESULT_DIR)

    # --- 9. Generate Visualizations ---
    logging.info("--- Generating Visualizations ---")

    # pyLDAvis
    visualizer.visualize_pyldavis(best_lda_model, train_bow, vectorizer)

    if 'topic_name' in train_df.columns: # Check if topics were assigned successfully
        # Time Series Prevalence
        visualizer.plot_topic_prevalence_timeline(train_df)
        analysis.save_intermediate_data( # Save underlying data
             train_df.groupby([pd.to_datetime(train_df['time']).dt.to_period('M').dt.to_timestamp(), 'topic_name']).size().reset_index(name='document_count'),
             "topic_prevalence_timeline_data", config.RESULT_DIR
        )
        analysis.save_intermediate_data( # Save wide format
             train_df.groupby([pd.to_datetime(train_df['time']).dt.to_period('M').dt.to_timestamp(), 'topic_name']).size().reset_index(name='document_count'),
             "topic_prevalence_wide_format", config.RESULT_DIR
        )


        # Hourly Activity
        visualizer.plot_hourly_topic_activity(train_df)
        hourly_activity_data = train_df.groupby([pd.to_datetime(train_df['time']).dt.hour, 'topic_name']).size().reset_index(name='count')
        hourly_activity_data.rename(columns={'level_0': 'hour'}, inplace=True) # Rename grouped hour column if needed
        analysis.save_intermediate_data(hourly_activity_data, "hourly_topic_activity_data", config.RESULT_DIR)
        analysis.save_intermediate_data(hourly_activity_data, "hourly_topic_activity_transposed", config.RESULT_DIR)


        # Network Graph
        visualizer.plot_topic_network_graph(best_topics_keywords, topic_names)
        # Save network edge data
        network_edges = []
        topic_words_dict = {name: set(keywords) for name, keywords in zip(topic_names, best_topics_keywords)}
        for topic1, topic2 in combinations(topic_names, 2):
             shared = topic_words_dict[topic1].intersection(topic_words_dict[topic2])
             if shared:
                  network_edges.append({
                       'source_topic': topic1, 'target_topic': topic2,
                       'shared_word_count': len(shared),
                       'shared_words_sample': ', '.join(list(shared)[:5]) # Sample
                  })
        analysis.save_intermediate_data(network_edges, "topic_relationships", config.RESULT_DIR)


        # Engagement Analysis
        visualizer.plot_engagement_per_topic(train_df)
        visualizer.plot_engagement_timeline(train_df)
        visualizer.plot_hourly_avg_engagement(train_df)
        visualizer.plot_engagement_distribution(train_df)


        # Heatmaps
        visualizer.plot_activity_heatmap(train_df)
        visualizer.plot_engagement_heatmap(train_df)


        # Word Clouds
        visualizer.plot_word_clouds(train_df)


        # Correlation Heatmap
        visualizer.plot_correlation_heatmap(train_df)

    else:
        logging.warning("Skipping visualizations that depend on topic assignments.")


    # --- 10. Perform Further Analysis ---
    logging.info("--- Performing Further Analysis ---")
    if dominant_topic_train is not None and 'topic' in train_df.columns:
         # Topic Density
         topic_density_df = analysis.calculate_topic_density(best_lda_model, doc_topic_dist_train, train_df['topic'])

         # Combine Volume & Density
         combined_metrics_df = analysis.combine_and_categorize_metrics(train_df, topic_density_df, topic_names)

         if combined_metrics_df is not None:
              # Display grouped topics in console
              analysis.display_grouped_topics(combined_metrics_df)
              # Save the combined metrics table
              analysis.save_analysis_results(None, None, combined_metrics_df, config.RESULT_DIR) # Save combined metrics
         else:
              logging.warning("Could not generate combined volume/density metrics.")
    else:
         logging.warning("Skipping density/volume analysis due to missing topic assignments.")


    logging.info("--- Pipeline Finished ---")

# --- Entry Point ---
if __name__ == "__main__":
    main()