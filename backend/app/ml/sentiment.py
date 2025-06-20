from transformers import AutoModelForSequenceClassification, AutoTokenizer
from langdetect import detect
import torch
from typing import Dict, Any
from .translation import translate_text
from app.ml.sarcasm_detection import AdvancedSentimentClassifier as CustomSarcasmDetector
import logging
import numpy as np
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Detect & set device
def _get_device():
    if torch.backends.mps.is_available():
        logger.info("Using device: MPS")
        return torch.device("mps")
    elif torch.cuda.is_available():
        logger.info("Using device: CUDA")
        return torch.device("cuda")
    else:
        logger.info("Using device: CPU")
        return torch.device("cpu")

device = _get_device()

# Initialize models
try:
    logger.info("🔄 Loading sarcasm detector...")
    SARCASM_DETECTOR = CustomSarcasmDetector()

    logger.info("🔄 Loading improved sentiment model...")
    # model_path = Path(__file__).resolve().parent / "improved_sentiment_model"
    # Improved Version 2
    model_path = Path(__file__).resolve().parent / "improved_sentiment_model_V2"
    assert model_path.exists(), f"Model path not found: {model_path}"

    CUSTOM_MODEL_TOKENIZER = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    CUSTOM_MODEL = AutoModelForSequenceClassification.from_pretrained(str(model_path), local_files_only=True)
    CUSTOM_MODEL.to(device)
    CUSTOM_MODEL.eval()
    logger.info("✅ Sentiment model loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load models: {str(e)}")
    raise

# Predict using your ML model
def predict_custom_sentiment(text: str) -> Dict[str, Any]:
    inputs = CUSTOM_MODEL_TOKENIZER(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = CUSTOM_MODEL(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
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

# Unified analysis function
def analyze_sentiment(text: str, company_id: int = None, db_session=None) -> Dict[str, Any]:
    try:
        translated = False

        # 🔠 Translate if not English
        if detect(text) != 'en':
            logger.info("🌐 Translating non-English text")
            text = translate_text(text)
            translated = True

        print(f"📝 Input: '{text[:80]}...'")

        # Step 1: Predict sentiment using improved model
        base_result = predict_custom_sentiment(text)
        base_sentiment = base_result["label"]
        base_confidence = base_result["confidence"]
        print(f"🔍 Base sentiment: {base_sentiment} ({base_confidence:.3f})")

        # Step 2: Sarcasm detection
        sarcasm_result = SARCASM_DETECTOR.detect_sarcasm(text)
        is_sarcastic = sarcasm_result.get("is_sarcastic", False)
        sarcasm_confidence = sarcasm_result.get("confidence", 0.0)
        print(f"🎭 Sarcasm detected? {is_sarcastic} (confidence: {sarcasm_confidence:.3f})")

        # Step 3: If sarcastic, reprocess
        if is_sarcastic:
            print("Debugging: Sarcasm has detected!!!")
            result = predict_custom_sentiment(text)
            final_sentiment = result["label"]
            final_confidence = (result["confidence"] + sarcasm_confidence) / 2
        else:
            final_sentiment = base_sentiment
            final_confidence = base_confidence

        output = {
            "sentiment": final_sentiment,
            "confidence": round(final_confidence, 3),
            "translated": translated,
            "is_sarcastic": is_sarcastic,
            "base_sentiment": base_sentiment,
            "sarcasm_confidence": round(sarcasm_confidence, 3)
        }

        print(f"✅ Final Result: {output}")
        return output

    except Exception as e:
        logger.error(f"❌ Sentiment analysis failed: {e}")
        return {
            "sentiment": "Error",
            "confidence": 0.0,
            "translated": False,
            "is_sarcastic": False,
            "base_sentiment": "N/A",
            "sarcasm_confidence": 0.0,
            "error": str(e)
        }
