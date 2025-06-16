from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pathlib import Path
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_sentiment_model():
    """Test the standalone sentiment model"""
    logger.info("Testing standalone sentiment model...")
    
    # ✅ Load your trained model (absolute path to avoid HF validation error)
    model_path = Path("./improved_sentiment_model").resolve()

    # ✅ Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    
    # Test cases
    test_cases = [
        ("I love this product!", "Like"),
        ("Worst experience ever", "Dislike"),
        ("This is exactly what I wanted", "Like"),
        ("The quality is exceptional", "Like"),
        ("Would not recommend to anyone", "Dislike"),
        ("Complete waste of money", "Dislike"),
        ("Oh great, another amazing product!", "Like"),  # Should be literal like
        ("Perfect, just what I didn't need", "Dislike")  # Negative due to "didn't need"
    ]

    logger.info("\nTesting standalone sentiment model predictions:")
    for text, expected in test_cases:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            prediction = "Like" if torch.argmax(probs) == 1 else "Dislike"
            confidence = torch.max(probs).item()
            
        result = "✅" if prediction == expected else "❌"
        print(f"{result} Text: {text}")
        print(f"  Expected: {expected}, Predicted: {prediction} (confidence: {confidence:.3f})")
        print("-" * 80)

def test_full_pipeline():
    """Test the full sentiment analysis pipeline with sarcasm detection"""
    try:
        # Add parent directory to path for module import
        current_dir = Path(__file__).parent
        parent_dir = current_dir.parent
        sys.path.insert(0, str(parent_dir))
        
        from app.ml.sentiment import analyze_sentiment
        logger.info("\nTesting full sentiment analysis pipeline...")
        
        # Test cases with expected outcomes
        test_cases = [
            ("I love this product!", "Like", False),
            ("Worst experience ever", "Dislike", False),
            ("This is exactly what I wanted", "Like", False),
            ("Would not recommend to anyone", "Dislike", False),
            ("Oh great, another amazing product!", "Dislike", True),  # Sarcastic
            ("Perfect, just what I didn't need", "Dislike", True),    # Sarcastic
            ("Fantastic service, if you enjoy waiting 3 hours", "Dislike", True),  # Sarcastic
            ("What a brilliant idea - not!", "Like", True),  # Sarcastic (flipped)
            ("I'm thrilled it arrived broken", "Like", True)  # Sarcastic (flipped)
        ]

        logger.info("\nTesting full pipeline predictions:")
        for text, expected_sentiment, expected_sarcasm in test_cases:
            result = analyze_sentiment(text)
            
            sentiment_match = result["sentiment"] == expected_sentiment
            sarcasm_match = result["is_sarcastic"] == expected_sarcasm
            result_icon = "✅" if sentiment_match and sarcasm_match else "❌"
            
            print(f"{result_icon} Text: {text}")
            print(f"  Expected: {expected_sentiment} (sarcasm: {expected_sarcasm})")
            print(f"  Result: {result['sentiment']} (sarcasm: {result['is_sarcastic']})")
            print(f"  Confidence: {result['confidence']}, Sarcasm confidence: {result.get('sarcasm_confidence', 0.0)}")
            print(f"  Base sentiment: {result.get('base_sentiment', 'N/A')}")
            print(f"  Translated: {result['translated']}")
            print("-" * 80)
            
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.info("Make sure to run this from the project root directory")
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")

if __name__ == "__main__":
    # Test standalone sentiment model
    test_sentiment_model()
    
    # Test full pipeline including sarcasm detection
    test_full_pipeline()