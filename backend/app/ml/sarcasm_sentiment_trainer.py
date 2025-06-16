# Standard libraries
import os
import random

# Data science & visualization
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# PyTorch and evaluation
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Hugging Face Transformers and datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import load_dataset, Dataset
import logging

# Ensure proper memory fallback on Apple Silicon
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Detect and set device
if torch.backends.mps.is_available():
    torch_device = torch.device("mps")
    logger.info("Using device: MPS (Apple Silicon)")
elif torch.cuda.is_available():
    torch_device = torch.device("cuda")
    logger.info("Using device: CUDA")
else:
    torch_device = torch.device("cpu")
    logger.info("Using device: CPU")

# Visualization config
sns.set(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 8)

# Stage 1: Load and combine diverse labeled datasets
def create_diverse_dataset(max_samples_per_source=2000):
    """Load sentiment datasets and add custom sarcastic examples, then balance them"""
    data = []

    # Amazon Reviews
    logger.info("Loading Amazon reviews...")
    amazon_ds = load_dataset("amazon_polarity")
    positive_amazon = [ex for ex in amazon_ds["train"] if ex["label"] == 1][:max_samples_per_source]
    negative_amazon = [ex for ex in amazon_ds["train"] if ex["label"] == 0][:max_samples_per_source]

    # Append Amazon data with labels
    for example in positive_amazon:
        data.append({"text": example["content"], "label": "Like", "source": "amazon"})
    for example in negative_amazon:
        data.append({"text": example["content"], "label": "Dislike", "source": "amazon"})

    # IMDB Reviews
    logger.info("Loading IMDB reviews...")
    imdb_ds = load_dataset("imdb")
    positive_imdb = [ex for ex in imdb_ds["train"] if ex["label"] == 1][:max_samples_per_source]
    negative_imdb = [ex for ex in imdb_ds["train"] if ex["label"] == 0][:max_samples_per_source]

    # Truncate long reviews to avoid exceeding max token length
    for example in positive_imdb:
        text = example["text"][:500] + "..." if len(example["text"]) > 500 else example["text"]
        data.append({"text": text, "label": "Like", "source": "imdb"})
    for example in negative_imdb:
        text = example["text"][:500] + "..." if len(example["text"]) > 500 else example["text"]
        data.append({"text": text, "label": "Dislike", "source": "imdb"})

    # Stanford Sentiment Treebank (SST-2)
    logger.info("Loading Stanford Sentiment...")
    sst_ds = load_dataset("sst2")
    sst_samples = sst_ds["train"].shuffle(seed=42).select(range(max_samples_per_source * 2))
    for example in sst_samples:
        label = "Like" if example["label"] == 1 else "Dislike"
        data.append({"text": example["sentence"], "label": label, "source": "sst"})

    # Manually added sarcastic examples
    logger.info("Adding custom sarcastic examples...")
    # sarcastic_examples = [ ... ]  # (omitted here for brevity, your list is great)
    sarcastic_examples = [
        ("Oh great, another amazing product!", "Dislike"),
        ("Perfect, just what I didn't need", "Dislike"),
        ("Fantastic service, if you enjoy waiting 3 hours", "Dislike"),
        ("What a brilliant idea - not!", "Dislike"),
        ("I'm thrilled it arrived broken", "Dislike"),
        ("Wonderful customer service, they hung up on me", "Dislike"),
        ("Amazing quality, it broke immediately", "Dislike"),
        ("Love how it doesn't work at all", "Dislike"),
        ("Great design, really user-friendly", "Dislike"),
        ("So helpful, they ignored all my questions", "Dislike"),
        ("I absolutely love this product", "Like"),
        ("Excellent quality and fast shipping", "Like"),
        ("Highly recommend to everyone", "Like"),
        ("Outstanding customer service", "Like"),
        ("Perfect for my needs", "Like"),
        ("Great value for money", "Like"),
        ("Works exactly as described", "Like"),
        ("Very satisfied with purchase", "Like"),
        ("Top quality materials", "Like"),
        ("Exceeded my expectations", "Like"),
        ("Terrible product, complete waste of money", "Dislike"),
        ("Worst purchase I've ever made", "Dislike"),
        ("Broke after one day", "Dislike"),
        ("Would not recommend to anyone", "Dislike"),
        ("Poor quality and overpriced", "Dislike"),
        ("Completely useless", "Dislike"),
        ("Customer service was horrible", "Dislike"),
        ("Doesn't work as advertised", "Dislike"),
        ("Cheaply made and overpriced", "Dislike"),
        ("Save your money, buy something else", "Dislike"),
    ]

    for text, label in sarcastic_examples:
        data.append({"text": text, "label": label, "source": "custom"})

    # Shuffle and balance dataset to max 5000 per class
    random.shuffle(data)
    df = pd.DataFrame(data)
    max_per_class = min(5000, df['label'].value_counts().min())
    like_samples = df[df['label'] == 'Like'].sample(n=max_per_class, random_state=42)
    dislike_samples = df[df['label'] == 'Dislike'].sample(n=max_per_class, random_state=42)
    balanced_df = pd.concat([like_samples, dislike_samples])
    balanced_data = balanced_df.to_dict('records')
    random.shuffle(balanced_data)

    # Log and visualize data distribution
    logger.info(f"Final dataset: {len(balanced_data)} samples")
    logger.info("Source distribution:")
    print(balanced_df['source'].value_counts())

    # Plot distribution
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    sns.countplot(data=balanced_df, x='label')
    plt.title('Label Distribution')

    plt.subplot(1, 3, 2)
    sns.countplot(data=balanced_df, x='source')
    plt.title('Source Distribution')
    plt.xticks(rotation=45)

    plt.subplot(1, 3, 3)
    sns.countplot(data=balanced_df, x='source', hue='label')
    plt.title('Source vs Label Distribution')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig("dataset_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

    return balanced_data

# Label mapping
LABEL_MAP = {"Dislike": 0, "Like": 1}

# Tokenization logic with cleaning
def tokenize_function(examples, tokenizer, max_length=64):
    """Clean text and tokenize using provided tokenizer"""
    cleaned_texts = [' '.join(text.split()) for text in examples['text']]
    encoding = tokenizer(
        cleaned_texts,
        truncation=True,
        padding=False,
        max_length=max_length,
        return_attention_mask=True
    )
    encoding['labels'] = [LABEL_MAP[label] for label in examples['label']]
    return encoding

# Evaluation metrics for validation/testing
def compute_metrics(eval_pred):
    """Calculate evaluation metrics: accuracy and F1 scores"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_weighted": f1_score(labels, predictions, average='weighted'),
        "f1_macro": f1_score(labels, predictions, average='macro')
    }

# Main training function
def train_improved_model(model_name="distilbert-base-uncased"):
    """Train sentiment classifier using diverse data and a transformer model"""
    
    logger.info("🔄 Creating diverse dataset...")
    raw_data = create_diverse_dataset(max_samples_per_source=2000)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Handle tokenizers missing pad_token

    # Apply tokenization
    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length=64),
        batched=True,
        remove_columns=['text', 'source', 'label']
    )

    # Train/val/test split (70/15/15)
    train_test = dataset.train_test_split(test_size=0.3, seed=42)
    val_test = train_test['test'].train_test_split(test_size=0.5, seed=42)
    train_dataset = train_test['train']
    val_dataset = val_test['train']
    test_dataset = val_test['test']

    logger.info("🤖 Loading pre-trained model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "Dislike", 1: "Like"},
        label2id={"Dislike": 0, "Like": 1},
        ignore_mismatched_sizes=True  # Avoid crash if head size doesn't match
    )
    model.to(torch_device)

    # Training configuration
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        # fp16=torch_device.type == "cuda",  # Enable mixed precision only on CUDA
        # bf16=torch_device.type == "mps",   # Enable bfloat16 only on M1/M2
        save_total_limit=2,
        fp16=torch_device.type == "cuda",
        bf16=False,  # <- Force disabled
    )

    # Trainer initialization
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer, padding=True)
    )

    # Start training
    logger.info("🚀 Starting training...")
    trainer.train()

    # Final evaluation
    logger.info("📈 Final evaluation on test set...")
    test_results = trainer.evaluate(test_dataset)
    logger.info(f"Test Results: {test_results}")

    # Generate classification report
    predictions = trainer.predict(test_dataset)
    y_true = predictions.label_ids
    y_pred = np.argmax(predictions.predictions, axis=-1)
    print("\n📝 Detailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Dislike", "Like"]))

    # Confusion matrix visualization
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Dislike", "Like"], yticklabels=["Dislike", "Like"])
    plt.title("Confusion Matrix - Improved Model")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig("improved_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Save model locally
    logger.info("💾 Saving improved model...")
    model.save_pretrained("./improved_sentiment_model")
    tokenizer.save_pretrained("./improved_sentiment_model")

    logger.info("✅ Training completed successfully!")
    logger.info("📁 Model saved to: ./improved_sentiment_model")
    logger.info("🧪 Use the separate test script to evaluate specific examples")
    return model, tokenizer

# Entry point
if __name__ == "__main__":
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    try:
        # First attempt with distilbert
        train_improved_model(model_name="distilbert-base-uncased")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.info("Trying with an even smaller model...")
        train_improved_model(model_name="albert-base-v2")