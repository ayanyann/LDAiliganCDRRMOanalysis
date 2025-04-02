# topic_modeler.py
import pandas as pd
import numpy as np
import pickle
import os
import logging
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_bow(texts, max_features, ngram_range, fit_vectorizer=None):
    """Creates Bag-of-Words matrix. Fits a new vectorizer or uses an existing one."""
    if fit_vectorizer:
        logging.info("Using existing vectorizer to transform texts.")
        vectorizer = fit_vectorizer
        bow_matrix = vectorizer.transform(texts)
    else:
        logging.info(f"Creating and fitting new CountVectorizer (max_features={max_features}, ngram_range={ngram_range}).")
        vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range)
        bow_matrix = vectorizer.fit_transform(texts)
        logging.info(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    return bow_matrix, vectorizer

def evaluate_lda(n_topics, train_bow, test_bow, train_texts, dictionary, vectorizer):
    """Trains LDA for a given number of topics and evaluates coherence and perplexity."""
    logging.debug(f"Training LDA model with n_topics={n_topics}...")
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, n_jobs=-1) # Use all CPU cores
    lda.fit(train_bow)

    # Extract top words for coherence calculation
    feature_names = vectorizer.get_feature_names_out()
    topics_keywords = []
    for topic_idx, topic_weights in enumerate(lda.components_):
        # Get top N word indices
        top_word_indices = topic_weights.argsort()[:-config.LDA_N_TOP_WORDS-1:-1]
        top_words = [feature_names[i] for i in top_word_indices]
        topics_keywords.append(top_words)

    # Compute coherence (using gensim's CoherenceModel)
    if not train_texts or not dictionary:
         logging.warning("Cannot compute coherence: train_texts or dictionary is empty.")
         coherence = np.nan
    else:
        try:
            cm = CoherenceModel(topics=topics_keywords, texts=train_texts, dictionary=dictionary, coherence='c_v')
            coherence = cm.get_coherence()
        except Exception as e:
            logging.error(f"Error calculating coherence for n_topics={n_topics}: {e}")
            coherence = np.nan # Assign NaN if coherence calculation fails

    # Compute perplexity on test data
    try:
        perplexity = lda.perplexity(test_bow)
    except Exception as e:
         logging.error(f"Error calculating perplexity for n_topics={n_topics}: {e}")
         perplexity = np.nan # Assign NaN if perplexity calculation fails


    logging.debug(f"n_topics={n_topics}, Coherence={coherence:.4f}, Perplexity={perplexity:.4f}")
    return lda, coherence, perplexity, topics_keywords

def find_best_lda_model(train_bow, test_bow, train_texts, topic_range, vectorizer):
    """Iterates through topic numbers, evaluates models, and selects the best one."""
    logging.info(f"Evaluating LDA models for topics in range: {list(topic_range)}")

    # Prepare for coherence calculation
    if not train_texts:
        logging.error("train_texts is empty. Cannot create dictionary for coherence.")
        return None, None, None, None

    try:
        dictionary = Dictionary(train_texts)
        # Filter dictionary (optional but good practice)
        dictionary.filter_extremes(no_below=5, no_above=0.7)
    except Exception as e:
        logging.error(f"Error creating or filtering Gensim dictionary: {e}")
        return None, None, None, None


    coherence_scores = []
    perplexity_scores = []
    models = []
    all_topics_keywords = [] # Store keywords for each model

    for n_topics in topic_range:
        try:
            lda, coherence, perplexity, topics_kws = evaluate_lda(
                n_topics, train_bow, test_bow, train_texts, dictionary, vectorizer
            )
            coherence_scores.append(coherence)
            perplexity_scores.append(perplexity)
            models.append(lda)
            all_topics_keywords.append(topics_kws)
        except Exception as e:
            logging.error(f"Failed to evaluate model for n_topics={n_topics}: {e}")
            coherence_scores.append(np.nan)
            perplexity_scores.append(np.nan)
            models.append(None)
            all_topics_keywords.append([])


    eval_results = pd.DataFrame({
        'n_topics': list(topic_range),
        'coherence': coherence_scores,
        'perplexity': perplexity_scores
    })
    eval_results.dropna(subset=['coherence'], inplace=True) # Drop rows where coherence failed

    if eval_results.empty:
        logging.error("No models could be successfully evaluated (coherence calculation failed).")
        return None, None, None, eval_results

    # Select best model based on highest coherence score
    best_idx_eval = eval_results['coherence'].idxmax() # Index in the eval_results DataFrame
    # Find the corresponding index in the original topic_range list
    best_n_topics = eval_results.loc[best_idx_eval, 'n_topics']
    best_model_original_idx = list(topic_range).index(best_n_topics)

    best_lda_model = models[best_model_original_idx]
    best_topics_keywords = all_topics_keywords[best_model_original_idx]
    best_coherence = eval_results.loc[best_idx_eval, 'coherence']
    best_perplexity = eval_results.loc[best_idx_eval, 'perplexity']


    logging.info(f"Best model selected: {best_n_topics} topics "
                 f"(Coherence={best_coherence:.4f}, Perplexity={best_perplexity:.4f})")

    return best_lda_model, best_topics_keywords, best_n_topics, eval_results

def save_model_and_vectorizer(lda_model, vectorizer, model_dir):
    """Saves the trained LDA model and CountVectorizer."""
    model_path = os.path.join(model_dir, 'lda_model.pkl')
    vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')

    try:
        with open(model_path, 'wb') as f:
            pickle.dump(lda_model, f)
        logging.info(f"LDA model saved to {model_path}")

        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        logging.info(f"Vectorizer saved to {vectorizer_path}")
    except Exception as e:
        logging.error(f"Error saving model or vectorizer: {e}")

def load_model_and_vectorizer(model_dir):
    """Loads a previously saved LDA model and CountVectorizer."""
    model_path = os.path.join(model_dir, 'lda_model.pkl')
    vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
    lda_model = None
    vectorizer = None

    try:
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                lda_model = pickle.load(f)
            logging.info(f"LDA model loaded from {model_path}")
        else:
            logging.error(f"Model file not found at {model_path}")

        if os.path.exists(vectorizer_path):
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
            logging.info(f"Vectorizer loaded from {vectorizer_path}")
        else:
            logging.error(f"Vectorizer file not found at {vectorizer_path}")

    except Exception as e:
        logging.error(f"Error loading model or vectorizer: {e}")

    return lda_model, vectorizer

def assign_topics_to_documents(lda_model, bow_matrix):
    """Assigns the most likely topic to each document."""
    if lda_model is None or bow_matrix is None:
        logging.error("LDA model or BoW matrix is None. Cannot assign topics.")
        return None, None
    logging.info("Assigning dominant topic to documents...")
    document_topic_dist = lda_model.transform(bow_matrix)
    dominant_topic = np.argmax(document_topic_dist, axis=1)
    logging.info("Topic assignment complete.")
    return dominant_topic, document_topic_dist