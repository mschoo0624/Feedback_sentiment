from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from langdetect import detect
import torch
from typing import Dict, Any
from .translation import translate_text
from .sarcasm_detection import AdvancedSarcasmDetector as CustomSarcasmDetector
import logging
import numpy as np
from pathlib import Path

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
    # initializes a sentiment analysis pipeline which is pre-trained. 
    SENTIMENT_MODEL = pipeline(
        task="sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=device,
        truncation=True,
        max_length=128
    )
    print("✅Sentiment model loaded successfully")
    
    print("🔄 Loading sarcasm detector...")
    SARCASM_DETECTOR = CustomSarcasmDetector()
    
    # model_path = Path("./improved_sentiment_model").resolve()
    model_path = Path(__file__).resolve().parent / "improved_sentiment_model"
    assert model_path.exists(), f"Model path not found: {model_path}"

    # Load your improved sarcasm-aware sentiment model
    print("🔄 Loading improved sarcasm-aware sentiment model...")
    CUSTOM_MODEL_TOKENIZER = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    print("DEBUGGING 1")
    CUSTOM_MODEL = AutoModelForSequenceClassification.from_pretrained(str(model_path), local_files_only=True)
    print("DEBUGGING 2")
    CUSTOM_MODEL.to(device)
    print("DEBUGGING 3")
    CUSTOM_MODEL.eval()
    print("✅ Loading improved sarcasm-aware sentiment model has completed!!!")
    
except Exception as e:
    logger.error(f"Failed to load models: {str(e)}")
    raise

# Run inference with improved model
def predict_custom_sentiment(text: str) -> Dict[str, Any]:
    tokens = CUSTOM_MODEL_TOKENIZER(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        output = CUSTOM_MODEL(**tokens)
        probs = torch.softmax(output.logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()
        label = "Like" if pred_idx == 1 else "Dislike"
        confidence = probs[0][pred_idx].item()

    return {
        "label": label,
        "confidence": round(confidence, 3),
        "probabilities": {
            "Dislike": round(probs[0][0].item(), 3),
            "Like": round(probs[0][1].item(), 3)
        }
    }

def analyze_sentiment(text: str, company_id: int = None, db_session=None) -> Dict[str, Any]:
    try:
        # 1. Translate if needed
        translated = False
        if detect(text) != 'en':
            logger.info("🌐 Translating non-English text")
            text = translate_text(text)
            translated = True

        print(f"📝 Input: '{text[:80]}...'")

        # 2. Run general sentiment analysis
        base_result = SENTIMENT_MODEL(text[:512])[0]
        label_map = {
            "LABEL_2": "Like",     # Positive
            "LABEL_1": "Neutral",  # Neutral
            "LABEL_0": "Dislike"   # Negative
        }
        base_sentiment = label_map.get(base_result['label'], "Dislike")
        confidence = float(base_result['score'])

        print(f"🔍 Base sentiment: {base_sentiment} ({confidence:.3f})")

        # 3. Run sarcasm detection
        sarcasm_result = SARCASM_DETECTOR.detect_sarcasm(text)
        is_sarcastic = sarcasm_result["is_sarcastic"] and sarcasm_result["confidence"] > 0.7
        sarcasm_confidence = float(sarcasm_result["confidence"])

        print(f"🎭 Sarcasm detected? {is_sarcastic} (confidence: {sarcasm_confidence:.3f})")

        # 4. If sarcastic → override using your improved model
        if is_sarcastic:
            print("Debugging: Sarcasm has detected!!!")
            custom_result = predict_custom_sentiment(text)
            final_sentiment = custom_result["label"]
            final_confidence = (custom_result["confidence"] + sarcasm_confidence) / 2
        else:
            final_sentiment = base_sentiment
            final_confidence = confidence

        # 5. Assemble response
        result = {
            "sentiment": final_sentiment,
            "confidence": round(final_confidence, 3),
            "translated": translated,
            "is_sarcastic": is_sarcastic,
            "base_sentiment": base_sentiment,
            "sarcasm_confidence": round(sarcasm_confidence, 3)
        }

        print(f"✅ Final Result: {result}")
        return result

    except Exception as e:
        logger.error(f"❌ Sentiment analysis failed: {e}")
        return {
            "sentiment": "Error",
            "confidence": 0.0,
            "translated": False,
            "is_sarcastic": False,
            "error": str(e)
        }