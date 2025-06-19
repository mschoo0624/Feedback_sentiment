from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pathlib import Path
import sys
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use MPS/CPU/CUDA as available
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

DEVICE = _get_device()


def test_sentiment_model():
    """Test the standalone improved sentiment model (Dislike vs Like)"""
    logger.info("🧪 Testing standalone sentiment model...")
    model_path = Path("./improved_sentiment_model").resolve()

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path), local_files_only=True)
    model.to(DEVICE)
    model.eval()

    test_cases = [
        ("I love this product!", "Like"),
        ("Worst experience ever", "Dislike"),
        ("Excellent service and great support", "Like"),
        ("I hate this product", "Dislike"),
        ("I would never use this product ever again.", "Dislike"),
        ("Just like what I wanted it.", "Like"),
        ("It feels good to type on the keyboard and the price-performance ratio is good.", "Like"),
        ("What a brilliant idea - not!", "Dislike"),
        ("Perfect, this product is the least thing I wanted it", "Dislike"),
        ("So helpful, they ignored all my questions", "Dislike"),
    ]

    for text, expected in test_cases:
        tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64).to(DEVICE)

        with torch.no_grad():
            outputs = model(**tokens)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_idx = torch.argmax(probs).item()
            prediction = "Like" if pred_idx == 1 else "Dislike"
            confidence = probs[0][pred_idx].item()

        result_icon = "✅" if prediction == expected else "❌"
        print("")
        print(f"{result_icon} Text: {text}")
        print(f"  Expected: {expected}, Predicted: {prediction} (confidence: {confidence:.3f})")
        print(f"  Raw probs: {probs.tolist()}")
        print("-" * 80)

def test_full_pipeline():
    """Test the full sentiment + sarcasm pipeline"""
    try:
        # Ensure correct import path to ml package
        current_dir = Path(__file__).parent.resolve()
        project_root = current_dir.parent
        sys.path.insert(0, str(project_root))

        # 🧠 Import the full analyzer
        from ml.sentiment import analyze_sentiment

        logger.info("🧪 Testing full sarcasm-aware sentiment pipeline...")

        test_cases = [
            ("I love this product!", "Like", False),
            ("Worst experience ever", "Dislike", False),
            ("Excellent service and great support", "Like", False),
            ("I hate this product", "Dislike", False),
            ("I would never use this product ever again.", "Dislike", False),
            ("Just like what I wanted it.", "Like", False),
            ("It feels good to type on the keyboard and the price-performance ratio is good.", "Like", False),
            ("What a brilliant idea - not!", "Dislike", True),
            ("Perfect, just what I didn't need", "Dislike", True),
            ("So helpful, they ignored all my questions", "Dislike", True),
        ]

        for text, expected_sentiment, expected_sarcasm in test_cases:
            print("🔍 Processing:", text)
            result = analyze_sentiment(text)

            sentiment_match = result["sentiment"] == expected_sentiment
            sarcasm_match = result["is_sarcastic"] == expected_sarcasm
            result_icon = "✅" if sentiment_match and sarcasm_match else "❌"
            
            print("")
            print(f"{result_icon} Text: {text}")
            print(f"  Expected: {expected_sentiment} (sarcasm: {expected_sarcasm})")
            print(f"  Result: {result['sentiment']} (sarcasm: {result['is_sarcastic']})")
            print(f"  Confidence: {result['confidence']}, Sarcasm confidence: {result.get('sarcasm_confidence', 0.0)}")
            print(f"  Base sentiment: {result.get('base_sentiment', 'N/A')}")
            print(f"  Translated: {result['translated']}")
            print("-" * 80)

    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.info("📁 Make sure to run this from the *project root* (where `app/` lives)")
    except Exception as e:
        logger.error(f"❌ Pipeline test failed: {e}")


if __name__ == "__main__":
    test_sentiment_model()
    test_full_pipeline()
