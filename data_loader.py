# data_loader.py
import pandas as pd
import logging
import config # Assuming config.py defines RAW_DATA_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_prepare_data(file_path=config.RAW_DATA_PATH): # Use path from config by default
    """Loads the raw CSV data and performs initial preparation."""
    logging.info(f"Loading data from: {file_path}")
    try:
        # --- FIX: Add low_memory=False ---
        df = pd.read_csv(file_path, low_memory=False)
        # --- END FIX ---
    except FileNotFoundError:
        logging.error(f"Error: Data file not found at {file_path}")
        return None
    except Exception as e: # Catch other potential reading errors
        logging.error(f"Error reading CSV file {file_path}: {e}")
        return None

    logging.info("Selecting relevant columns and calculating engagements.")
    # Select initial relevant columns
    columns_to_use = ['shares', 'sharedPost/text', 'text', 'time',
                      'sharedPost/pageName/name', 'likes','topReactionsCount']

    # Gracefully handle missing columns
    actual_columns_to_use = [col for col in columns_to_use if col in df.columns]
    missing_cols = [col for col in columns_to_use if col not in df.columns]
    if missing_cols:
        logging.warning(f"Missing expected columns, they will be skipped: {missing_cols}")
    if not actual_columns_to_use:
        logging.error("None of the required initial columns found. Cannot proceed.")
        return None

    df_subset = df[actual_columns_to_use].copy() # Use only available columns

    # Calculate engagements safely checking for column existence
    likes_col = 'likes' if 'likes' in df_subset.columns else None
    reactions_col = 'topReactionsCount' if 'topReactionsCount' in df_subset.columns else None

    if likes_col:
        df_subset['likes_numeric'] = pd.to_numeric(df_subset[likes_col], errors='coerce').fillna(0)
    else:
        df_subset['likes_numeric'] = 0

    if reactions_col:
         df_subset['topReactionsCount_numeric'] = pd.to_numeric(df_subset[reactions_col], errors='coerce').fillna(0)
    else:
        df_subset['topReactionsCount_numeric'] = 0

    df_subset['engagements'] = df_subset['likes_numeric'] + df_subset['topReactionsCount_numeric']

    logging.info("Merging text columns into 'caption'.")
    # Merge text columns, prioritizing 'text' if it exists
    text_col_present = 'text' in df_subset.columns
    shared_text_col_present = 'sharedPost/text' in df_subset.columns

    if text_col_present:
        df_subset['text_norm'] = df_subset['text'].replace('', None).fillna('') # Normalize Nones for combine_first if needed
    else:
        df_subset['text_norm'] = '' # Create empty column if 'text' is missing

    if shared_text_col_present:
        df_subset['shared_text_norm'] = df_subset['sharedPost/text'].replace('', None).fillna('')
    else:
        df_subset['shared_text_norm'] = ''

    # Combine available text fields
    df_subset['caption'] = df_subset['text_norm'].combine_first(df_subset['shared_text_norm'])
    # Ensure empty string if both were missing/empty
    df_subset['caption'] = df_subset['caption'].fillna('')


    # Drop rows where the combined caption is still empty or None AFTER combine_first
    df_subset.dropna(subset=['caption'], inplace=True)
    df_subset = df_subset[df_subset['caption'].str.strip() != ''] # Remove rows with only whitespace caption

    if df_subset.empty:
        logging.warning("No valid text data found after merging and cleaning.")
        return None

    # Select final columns for output, checking existence first
    final_column_candidates = ['caption', 'time', 'sharedPost/pageName/name', 'shares', 'engagements']
    final_columns = [col for col in final_column_candidates if col in df_subset.columns or col == 'caption' or col == 'engagements'] # caption/engagements are created

    df_final = df_subset[final_columns].copy()

    logging.info(f"Data loading and preparation complete. Shape: {df_final.shape}")
    return df_final