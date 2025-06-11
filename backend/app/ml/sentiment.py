from transformers import pipeline
from langdetect import detect
import torch
from typing import Dict, Any
from .translation import translate_text
from .sarcasm_detection import detect_sarcasm
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Optimize for Mac M1/M2 hardware
if torch.backends.mps.is_available():
    device = 0  # MPS on Mac
    logger.info("Using device: MPS")
elif torch.cuda.is_available():
    device = 0  # CUDA
    logger.info("Using device: CUDA")
else:
    device = -1  # CPU
    logger.info("Using device: CPU")

try:
    SENTIMENT_MODEL = pipeline(
        task="text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device
    )
    logger.info("Sentiment model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load sentiment model: {str(e)}")
    raise

def analyze_sentiment(text: str, company_id: int = None, db_session=None) -> Dict[str, Any]:
    try:
        # Handle translation if needed
        translated = False
        if detect(text) != 'en':
            # Debugging.
            print("Debugging: Translation has operated!!!")
            logger.info(f"Translating non-English text: {text[:50]}...")
            text = translate_text(text)
            translated = True
        
        # Debugging.
        print("Debugging: This Line of code!!!")
        # Get sentiment prediction
        base_result = SENTIMENT_MODEL(text)[0]
        # Debugging.
        print("Debugging: It's working now!!!")
        
        sentiment = "Like" if base_result['label'] == "POSITIVE" else "Dislike"
        confidence = float(base_result['score'])

        # Check for sarcasm
        # sarcasm_detected = False
        sarcasm = detect_sarcasm(text)
        if sarcasm['is_sarcastic'] and sarcasm['confidence'] > 0.70:
            # Debugging.
            print("Debugging: Sarcasm has detected!!!")
            logger.info(f"Sarcasm detected with confidence {sarcasm['confidence']}")
            # Flip the sentiment for sarcastic comments
            sentiment = "Dislike" if sentiment == "Like" else "Like"
            confidence = max(confidence, sarcasm['confidence'])

        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 3),
            "translated": translated,
            "is_sarcastic": sarcasm['is_sarcastic'],  # Return boolean
            "device": device
        }

    except Exception as e:
        logger.error(f"Sentiment analysis failed: {str(e)}")
        raise