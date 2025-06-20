import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import logging
import numpy as np
"""
Testing only the Machine Learning Model.
"""
# Setup logging
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

device = _get_device()

# Load model
# MODEL_DIR = Path("./improved_sentiment_model").resolve()
#  impoved Version 2 
MODEL_DIR = Path("./improved_sentiment_model_V2").resolve()
logger.info(f"Loading model from {MODEL_DIR}")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).to(device)
model.eval()

# Label mapping (make sure it matches training)
id2label = {0: "Dislike", 1: "Like"}

# Test examples
test_cases = [
    ("I love this product!", "Like"),
    ("Worst experience ever", "Dislike"),
    ("Excellent service and great support", "Like"),
    ("I hate this product", "Dislike"),
    ("I would never use this product ever again.", "Dislike"),
    ("Just like what I wanted it.", "Like"),
    ("It feels good to type on the keyboard and the price-performance ratio is good.", "Like"),
    ("What a brilliant idea - not!", "Dislike"),      # sarcastic
    ("Perfect, this product is the least thing I wanted it", "Dislike"),  # sarcastic
    ("So helpful, they ignored all my questions", "Dislike"),  # sarcastic
]

# Run test cases 
count = 1
for text, expected in test_cases:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        pred_id = int(np.argmax(probs))
        pred_label = id2label[pred_id]  
        confidence = probs[pred_id]

    result = "✅" if pred_label == expected else "❌"
    print("")
    print(f"{result} Text ({count}): {text}")
    print(f"  Expected: {expected}, Predicted: {pred_label} (confidence: {confidence:.3f})")
    print(f"  Raw probs: {probs}")
    print("-" * 80)
    count += 1
