# text_preprocessor.py
import re
import unicodedata
import nltk
from nltk.corpus import stopwords
import logging
import config # Import config to access stopwords

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- NLTK Stopwords Download ---
def ensure_nltk_stopwords():
    """Downloads NLTK stopwords if not already present."""
    try:
        stopwords.words('english')
        logging.debug("NLTK stopwords already downloaded.")
    except LookupError:
        logging.info("NLTK 'stopwords' resource not found. Downloading...")
        nltk.download('stopwords')
        logging.info("NLTK stopwords download complete.")

ensure_nltk_stopwords() # Ensure they are downloaded when module is imported

# --- Preprocessing Function ---
def preprocess_text(text):
    """Cleans and preprocesses a single text string."""
    if not isinstance(text, str):
        return '' # Return empty string for non-string input

    text = str(text).lower() # 1. Lowercase

    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # 3. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # 4. Normalize unicode characters
    try:
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    except Exception as e:
        logging.warning(f"Unicode normalization failed for text snippet: {text[:50]}... Error: {e}")
        # Continue processing even if normalization fails for some chars

    # 5. Remove Emojis
    emoji_pattern = re.compile("["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    # 6. Replace hashtags (#word -> HASHTAG_word)
    text = re.sub(r'#([a-zA-Z0-9_]+)', r'HASHTAG_\1', text)

    # 7. Remove remaining punctuation (keeping letters, numbers, underscore, whitespace)
    text = re.sub(r'[^a-zA-Z0-9_\s]', '', text)

    # 8. Remove numbers (optional, depending on whether numbers are meaningful)
    text = re.sub(r'\d+', '', text)

    # 9. Define and remove stopwords
    english_stopwords = set(stopwords.words('english'))
    tagalog_stopwords_set = set(config.CUSTOM_TAGALOG_STOPWORDS)
    bisaya_stopwords_set = set(config.CUSTOM_BISAYA_STOPWORDS)
    all_stop_words = english_stopwords.union(tagalog_stopwords_set).union(bisaya_stopwords_set)
    all_stop_words.add('cdrrmo') # Add domain-specific words if needed
    all_stop_words.add('iligan')

    # 10. Tokenize and filter stopwords
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in all_stop_words and len(word) > 1] # Keep words > 1 char

    # 11. Join back into string
    return ' '.join(filtered_tokens)

def apply_preprocessing(df, text_column='caption'):
    """Applies preprocessing to a DataFrame column."""
    if text_column not in df.columns:
        logging.error(f"Column '{text_column}' not found in DataFrame for preprocessing.")
        return df
    logging.info(f"Applying preprocessing to '{text_column}' column...")
    df['cleaned_text'] = df[text_column].apply(preprocess_text)
    logging.info("Preprocessing complete.")
    # Log count of empty strings after preprocessing
    empty_count = (df['cleaned_text'] == '').sum()
    if empty_count > 0:
        logging.warning(f"{empty_count} rows resulted in empty text after preprocessing.")
    return df