# gemini_labeler.py
import google.generativeai as genai
import time
import logging
import config # To get API key and model name
import re



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_prompt(keywords):
    """Creates the prompt for the Gemini API."""
    keyword_str = ", ".join(keywords)
    # Fine-tune this prompt for better results!
    prompt = (
        "You are an expert in analyzing social media content related to disaster risk reduction and management (DRRM) in the Philippines context (Iligan City). "
        f"Based on the following keywords representing a topic: '{keyword_str}'.\n"
        "Suggest a concise and descriptive label (2-5 words) that accurately summarizes this topic. "
        "The label should be easily understandable in the DRRM context.\n"
        "Examples: 'Flood Warnings', 'Earthquake Preparedness', 'Emergency Hotlines', 'Road Closure Advisory', 'Volunteer Activities'.\n"
        "Respond with ONLY the label name, nothing else."
    )
    return prompt

def label_topics_with_gemini(topics_keywords_list):
    """Uses the Gemini API to generate labels for a list of topic keywords."""
    if not config.GEMINI_API_KEY:
        logging.error("Gemini API key not configured. Skipping topic labeling.")
        num_topics = len(topics_keywords_list)
        return [f"Unlabeled Topic #{i}" for i in range(num_topics)]

    logging.info(f"Configuring Gemini client with model: {config.GEMINI_MODEL_NAME}")
    try:
        genai.configure(api_key=config.GEMINI_API_KEY)
        model = genai.GenerativeModel(config.GEMINI_MODEL_NAME)
        logging.info("Gemini client configured successfully.")
    except Exception as e:
        logging.error(f"Failed to configure Gemini client: {e}")
        num_topics = len(topics_keywords_list)
        return [f"Unlabeled Topic #{i}" for i in range(num_topics)]

    topic_labels = []
    logging.info(f"Generating labels for {len(topics_keywords_list)} topics using Gemini...")

    for i, keywords in enumerate(topics_keywords_list):
        prompt = generate_prompt(keywords)
        retries = 0
        success = False
        while retries < config.GEMINI_MAX_RETRIES and not success:
            try:
                logging.debug(f"Sending prompt for topic {i} (keywords: {', '.join(keywords[:5])}...)")
                # Updated API call structure for google-generativeai
                response = model.generate_content(prompt)

                # Accessing the text response correctly
                if response and response.text:
                    generated_label = response.text.strip().replace('"', '').replace("'", "").replace("*","")
                    # Basic cleaning: remove potential markdown like "**Label**" -> "Label"
                    generated_label = re.sub(r'^\*+\s*(.*?)\s*\*+$', r'\1', generated_label)
                    # Limit length if needed
                    generated_label = " ".join(generated_label.split()[:5]) # Keep max 5 words

                    logging.info(f"Generated label for topic {i}: '{generated_label}'")
                    topic_labels.append(generated_label)
                    success = True
                else:
                    # Handle cases where response is empty or doesn't have 'text'
                    error_detail = f"Response: {response}" if response else "Empty response"
                    logging.warning(f"Gemini API returned unexpected response for topic {i}. {error_detail}. Using default label.")
                    topic_labels.append(f"API Error Topic #{i}")
                    success = True # Count as success to avoid infinite loop if API consistently fails


            except Exception as e:
                retries += 1
                logging.warning(f"Gemini API error for topic {i} (attempt {retries}/{config.GEMINI_MAX_RETRIES}): {e}")
                if retries < config.GEMINI_MAX_RETRIES:
                    logging.info(f"Retrying after {config.GEMINI_RETRY_DELAY} seconds...")
                    time.sleep(config.GEMINI_RETRY_DELAY)
                else:
                    logging.error(f"Max retries reached for topic {i}. Using default label.")
                    topic_labels.append(f"API Fail Topic #{i}")
                    # No need to set success=True here, the loop will terminate

        # Safety net in case loop finishes without success (shouldn't happen with current logic)
        if len(topic_labels) <= i:
             logging.error(f"Failed to get label for topic {i} after all retries. Using default.")
             topic_labels.append(f"Unknown Topic #{i}")

        # Optional: Add a small delay between requests to avoid hitting rate limits
        time.sleep(5) # Sleep for 1 second between calls


    logging.info("Gemini topic labeling complete.")
    # Sanity check: ensure number of labels matches number of topics
    if len(topic_labels) != len(topics_keywords_list):
        logging.error("Mismatch between number of topics and generated labels!")
        # Fallback strategy
        num_topics = len(topics_keywords_list)
        return [f"Labeling Mismatch #{i}" for i in range(num_topics)]

    return topic_labels