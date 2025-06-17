import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import logging

# Set up logging
logger = logging.getLogger(__name__)

class AdvancedSarcasmDetector:
    """
    A sarcasm-aware sentiment classifier using a custom fine-tuned transformer model.
    """

    def __init__(self, model_path="./improved_sentiment_model"):
        self.model_path = model_path
        self.device = self._get_device()

        try:
            logger.info(f"Loading sarcasm-aware model from {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
            self.model.eval()
            logger.info("✅ Sarcasm-aware model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load sarcasm-aware model: {str(e)}")
            raise

    def _get_device(self):
        if torch.backends.mps.is_available():
            device = 0  # MPS on Mac
            logger.info("Using device: MPS")
        elif torch.cuda.is_available():
            device = 0  # CUDA
            logger.info("Using device: CUDA")
        else:
            device = -1  # CPU
            logger.info("Using device: CPU")

    def detect_sarcasm(self, text: str) -> dict:
        """
        Predict sentiment and determine if sentiment reversal is likely sarcasm.
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True
            ).to(self.device)

            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

            # Get prediction
            predicted_class = int(np.argmax(probs))
            confidence = float(np.max(probs))

            # Our label mapping
            label_map = {0: "Dislike", 1: "Like"}
            predicted_label = label_map[predicted_class]

            # Naive sarcasm heuristic: extremely positive tone, but actual intent seems negative (or vice versa)
            # NOTE: You can improve this logic based on linguistic cues or metadata
            sarcastic = (
                "great" in text.lower() or
                "perfect" in text.lower() or
                "amazing" in text.lower() or
                "love" in text.lower()
            ) and predicted_label == "Dislike" and confidence > 0.85

            return {
                "predicted_label": predicted_label,
                "confidence": round(confidence, 3),
                "is_sarcastic": sarcastic
            }

        except Exception as e:
            logger.error(f"Error in sarcasm detection: {str(e)}")
            return {
                "predicted_label": "Error",
                "confidence": 0.0,
                "is_sarcastic": False
            }
