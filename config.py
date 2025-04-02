# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- File Paths ---
RAW_DATA_PATH = 'data/cdrrmo-raw-data.csv' # Or adjust path relative to your project
OUTPUT_DIR = 'output'
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')
RESULT_DIR = os.path.join(OUTPUT_DIR, 'results')

# --- API Keys ---
# IMPORTANT: Best practice is to use environment variables.
# Avoid hardcoding keys directly in the script.
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not found in environment variables. Gemini labeling will fail.")
    # You could raise an error here or provide a default fallback if needed.

# --- LDA Model Parameters ---
LDA_MAX_FEATURES = 5000
LDA_NGRAM_RANGE = (1, 3) # Unigrams, bigrams, trigrams
LDA_TOPIC_RANGE = range(5, 31, 5) # Test fewer topics initially (e.g., 5 to 30) for faster iteration
LDA_N_TOP_WORDS = 20 # Number of top words to extract per topic

# --- Visualization Parameters ---
VIZ_WIDTH = 1200
VIZ_HEIGHT_STANDARD = 600
VIZ_HEIGHT_TALL = 800

# --- Analysis Parameters ---
DENSITY_METRIC_TOP_WORDS = 20 # For topic density calculation

# --- Gemini Labeling Parameters ---
GEMINI_MODEL_NAME = "gemini-2.5-pro-exp-03-25" # Or another suitable model like "gemini-pro"
GEMINI_MAX_RETRIES = 3
GEMINI_RETRY_DELAY = 5 # seconds

# --- Create Output Directories ---
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# --- Custom Stopwords (Keep these here or move to preprocessor if preferred) ---
CUSTOM_TAGALOG_STOPWORDS = [
    'ako', 'ikaw', 'siya', 'tayo', 'kami', 'kayo', 'sila', 'ito', 'iyan', 'iyon',
    'ang', 'ng', 'sa', 'mga', 'ay', 'at', 'na', 'pero', 'o', 'hindi', 'wala',
    'may', 'mayroon', 'po', 'opo', 'ho', 'oho', 'para', 'tungkol', 'mula',
    'hanggang', 'isang', 'dalawa', 'ba', 'kasi', 'kaya', 'dahil', 'pa', 'naman',
    'nga', 'din', 'rin', 'bang'
]
CUSTOM_BISAYA_STOPWORDS = [
    'ako', 'ikaw', 'siya', 'kita', 'kami', 'kamo', 'sila', 'kini', 'kana', 'kadto',
    'ang', 'og', 'sa', 'mga', 'ug', 'apan', 'o', 'dili', 'wala', 'naa', 'adunay',
    'para', 'bahin', 'gikan', 'hangtod', 'usa', 'duha', 'ba', 'kay', 'mao',
    'tungod', 'pa', 'man', 'pud', 'sab', 'kaayo', 'nga'
]