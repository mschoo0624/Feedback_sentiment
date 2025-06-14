from transformers import pipeline
from langdetect import detect
import torch
from typing import Dict, Any
from .translation import translate_text
from .sarcasm_detection import AdvancedSarcasmDetector  # Import class directly
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

# Initialize models
try:
    logger.info("Loading sentiment model...")
    SENTIMENT_MODEL = pipeline(
        task="text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device,
        truncation=True,
        max_length=128
    )
    logger.info("Sentiment model loaded successfully")
    
    logger.info("Initializing sarcasm detector...")
    SARCASM_DETECTOR = AdvancedSarcasmDetector()
    logger.info("Sarcasm detector initialized successfully")
except Exception as e:
    logger.error(f"Failed to load models: {str(e)}")
    raise

def analyze_sentiment(text: str, company_id: int = None, db_session=None) -> Dict[str, Any]:
    try:
        # Handle translation if needed
        translated = False
        if detect(text) != 'en':
            logger.info(f"Translating non-English text: {text[:50]}...")
            text = translate_text(text)
            translated = True
        
        # Get sentiment prediction
        base_result = SENTIMENT_MODEL(text[:512])[0]  # Truncate for safety
        base_sentiment = "Like" if base_result['label'] == "POSITIVE" else "Dislike"
        confidence = float(base_result['score'])

        # Check for sarcasm
        sarcasm_result = SARCASM_DETECTOR.detect_sarcasm(text)
        is_sarcastic = sarcasm_result['is_sarcastic'] and sarcasm_result['confidence'] > 0.65
        
        if is_sarcastic:
            print("Debugging: is_sarcastic has worked!!!")
            logger.info(f"Sarcasm detected with confidence {sarcasm_result['confidence']}")
            # Flip the sentiment for sarcastic comments
            final_sentiment = "Dislike" if base_sentiment == "Like" else "Like"
            # Boost confidence when sarcasm is detected
            confidence = max(confidence, sarcasm_result['confidence'])
        else:
            final_sentiment = base_sentiment

        return {
            "sentiment": final_sentiment,
            "confidence": round(confidence, 3),
            "translated": translated,
            "is_sarcastic": is_sarcastic,
            "device": device
        }

    except Exception as e:
        logger.error(f"Sentiment analysis failed: {str(e)}")
        return {
            "sentiment": "Error",
            "confidence": 0.0,
            "translated": False,
            "is_sarcastic": False,
            "error": str(e)
        }