from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from langdetect import detect
import torch
from typing import Dict, Any
from .translation import translate_text
from .sarcasm_detection import AdvancedSarcasmDetector as CustomSarcasmDetector
import logging
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# Optimize for Mac M1/M2 hardware
def _get_device():
    if torch.backends.mps.is_available():
        device = 0  # MPS on Mac
        logger.info("Using device: MPS")
    elif torch.cuda.is_available():
        device = 0  # CUDA
        logger.info("Using device: CUDA")
    else:
        device = -1  # CPU
        logger.info("Using device: CPU")
        
device = _get_device()

# Initialize models
try:
    # Dedicated sentiment model (3-class: negative, neutral, positive)
    # Debugging.
    print("Loading sentiment model...")
    
    SENTIMENT_MODEL = pipeline(
        task="sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=device,
        truncation=True,
        max_length=128
    )
    
    logger.info("Sentiment model loaded successfully")
    
    SARCASM_DETECTOR = CustomSarcasmDetector()  # Use your custom model
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
        
        print(f"DEBUG: About to analyze text: '{text[:50]}...'")
        
        # Get sentiment prediction using dedicated model
        base_result = SENTIMENT_MODEL(text[:512])[0]  # Truncate for safety
        
        # Changes
        # Map 3-class sentiment to our 2-class system
        if base_result['label'] == 'LABEL_2':  # Positive
            base_sentiment = "Like"
        elif base_result['label'] == 'LABEL_0':  # Negative
            base_sentiment = "Dislike"
        else:  # Neutral (LABEL_1)
            # Treat neutral as negative by default
            base_sentiment = "Dislike"
        
        # label = base_result['label']
        # if label == "LABEL_2":  # POSITIVE
        #     base_sentiment = "Like"
        # elif label == "LABEL_0":  # NEGATIVE
        #     base_sentiment = "Dislike"
        # else:
        #     base_sentiment = "Dislike"  # Treat neutral as Dislike (or create 3-class)
            
        confidence = float(base_result['score'])
        
        print(f"DEBUG: Base sentiment: {base_sentiment} (confidence: {confidence})")

        # Check for sarcasm using your custom model
        print(f"DEBUG: Checking for sarcasm...")
        sarcasm_result = SARCASM_DETECTOR.detect_sarcasm(text)
        print(f"DEBUG: Sarcasm result: {sarcasm_result}")
        
        # Handle numpy types for confidence values
        sarcasm_confidence = float(sarcasm_result['confidence'])
        # Changes
        # is_sarcastic = sarcasm_result['is_sarcastic'] and sarcasm_confidence > 0.4
        is_sarcastic = sarcasm_result['is_sarcastic'] and sarcasm_confidence > 0.7
        
        print(f"DEBUG: is_sarcastic = {is_sarcastic}")
        
        if is_sarcastic:
            # Flip sentiment for sarcastic comments
            final_sentiment = "Dislike" if base_sentiment == "Like" else "Like"
            # Combine confidences
            combined_confidence = (confidence + sarcasm_confidence) / 2
        else:
            final_sentiment = base_sentiment
            combined_confidence = confidence

        result = {
            "sentiment": final_sentiment,
            "confidence": round(combined_confidence, 3),
            "translated": translated,
            "is_sarcastic": is_sarcastic,
            "base_sentiment": base_sentiment,
            "sarcasm_confidence": round(sarcasm_confidence, 3)
        }
        
        print(f"DEBUG: Final result: {result}")
        return result

    except Exception as e:
        logger.error(f"Sentiment analysis failed: {str(e)}")
        return {
            "sentiment": "Error",
            "confidence": 0.0,
            "translated": False,
            "is_sarcastic": False,
            "error": str(e)
        }