import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class AdvancedSentimentClassifier:
    def __init__(self):
        self.device = self._get_device()
        # self.model_path = Path(__file__).resolve().parent / "improved_sentiment_model"
        #  impoved Version 2 
        self.model_path = Path(__file__).resolve().parent / "improved_sentiment_model_V2"

        try:
            logger.info(f"🔄 Loading model from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path)).to(self.device)
            self.model.eval()
            logger.info("✅ Sentiment model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise

    def _get_device(self):
        if torch.backends.mps.is_available():
            logger.info("Using device: MPS")
            return torch.device("mps")
        elif torch.cuda.is_available():
            logger.info("Using device: CUDA")
            return torch.device("cuda")
        else:
            logger.info("Using device: CPU")
            return torch.device("cpu")

    def detect_sarcasm(self, text: str) -> dict:
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

            predicted_class = int(np.argmax(probs))
            confidence = float(np.max(probs))

            label_map = {0: "Dislike", 1: "Like"}
            predicted_label = label_map[predicted_class]
            is_sarcastic = predicted_label == "Dislike" and confidence > 0.85
            
            return {
            "predicted_label": predicted_label,
            "confidence": round(confidence, 3),
            "is_sarcastic": is_sarcastic  # ✅ REQUIRED by sentiment.py
        }

        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            return {
                "predicted_label": "Error",
                "confidence": 0.0,
                "is_sarcastic": False
            }
