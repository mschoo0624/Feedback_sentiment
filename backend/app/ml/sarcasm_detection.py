import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import logging
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

class AdvancedSarcasmDetector:
    """
    A sarcasm-aware sentiment classifier using a custom fine-tuned transformer model.
    """

    def __init__(self):
        self.device = self._get_device()

        # Resolve path to local model directory
        model_path = Path(__file__).resolve().parent / "improved_sentiment_model"

        try:
            logger.info(f"🔄 Loading sarcasm-aware model from {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(model_path)).to(self.device)
            self.model.eval()
            logger.info("✅ Sarcasm-aware model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load sarcasm-aware model: {str(e)}")
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

            label_map = {0: "Dislike", 1: "Like"}
            predicted_label = label_map[predicted_class]

            # Heuristic sarcasm detector: overly positive words but negative intent
            sarcastic = (
                any(phrase in text.lower() for phrase in ["great", "perfect", "amazing", "love"])
                and predicted_label == "Dislike"
                and confidence > 0.85
            )

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
