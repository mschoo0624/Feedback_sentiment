import os
import random
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import load_dataset, Dataset
import logging

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if torch.backends.mps.is_available():
    torch_device = torch.device("mps")
    logger.info("Using device: MPS (Apple Silicon)")
elif torch.cuda.is_available():
    torch_device = torch.device("cuda")
    logger.info("Using device: CUDA")
else:
    torch_device = torch.device("cpu")
    logger.info("Using device: CPU")

# Configure visualization
sns.set(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 8)

# Stage 1: Enhanced dataset preparation
def pre_labeled_datasets(max_samples=None):
    """Load sarcasm and sentiment datasets, balance classes"""
    # Load datasets
    sarcasm_ds = load_dataset("tweet_eval", "irony")
    sentiment_ds = load_dataset("amazon_polarity")
    
    data = []
    
    # Process sarcasm dataset
    for example in sarcasm_ds["train"]:
        label = "Dislike" if example["label"] == 1 else "Like"
        data.append({"text": example["text"], "label": label})
    
    # Balance sentiment dataset
    positive = [ex for ex in sentiment_ds["train"] if ex["label"] == 1][:3000]
    negative = [ex for ex in sentiment_ds["train"] if ex["label"] == 0][:3000]
    
    for example in positive + negative:
        label = "Like" if example["label"] == 1 else "Dislike"
        data.append({"text": example["content"], "label": label})
    
    random.shuffle(data)
    
    # Apply sampling if needed for fallback
    if max_samples and len(data) > max_samples:
        data = random.sample(data, max_samples)
    
    # Analyze class distribution
    df = pd.DataFrame(data) 
    print("\n📈 Class Distribution:")
    print(df["label"].value_counts())
    
    # Visualization
    plt.figure()
    sns.countplot(x="label", data=df)
    plt.title("Class Distribution")
    plt.savefig("class_distribution.png")
    plt.close()
    
    return data

# Label mapping
LABEL_MAP = {"Dislike": 0, "Like": 1}

# Tokenization
def tokenize(examples, tokenizer, max_length=64):
    """Tokenize text and map labels with adjustable length"""
    texts = examples['text']
    
    encoding = tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=max_length
    )
    encoding['label'] = [LABEL_MAP[label] for label in examples['label']]
    return encoding

# Compute metrics
def compute_metrics(eval_pred):
    """Calculate accuracy and F1 score"""
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted")
    }

# 🧠 CPU fallback training (for extreme low-memory cases)
def train_model_cpu_fallback():
    """Low-memory CPU training with tiny model"""
    print("\n⚠️ Switching to low-memory CPU fallback mode")
    torch_device = torch.device("cpu")
    model_name = "prajjwal1/bert-tiny"
    max_samples = 2000
    
    # Prepare dataset
    raw_data = pre_labeled_datasets(max_samples=max_samples)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create dataset with shorter sequences
    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(
        lambda x: tokenize(x, tokenizer, max_length=32), 
        batched=True
    )
    dataset = dataset.remove_columns(['text'])
    
    # Train-test split
    dataset = dataset.train_test_split(test_size=0.2)
    
    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2,
        id2label={0: "Dislike", 1: "Like"}
    )
    model.to(torch_device)
    
    # Training arguments for CPU fallback
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        report_to="none",
        load_best_model_at_end=True,
        save_total_limit=1,
        no_cuda=True,
        use_mps_device=False,
        fp16=False
    )
    
    # Create Trainer
    trainer_obj = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer)
    )
    
    # Train model
    print("\n🚀 Starting CPU fallback training...")
    trainer_obj.train()
    
    # Evaluate
    print("\n📈 Evaluating model...")
    metrics = trainer_obj.evaluate()
    print("\n✅ Evaluation Results (CPU Fallback):", metrics)
    
    # Get predictions
    predictions = trainer_obj.predict(dataset["test"])
    y_true = predictions.label_ids
    y_pred = np.argmax(predictions.predictions, axis=-1)
    
    # Classification report
    print("\n📝 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Dislike", "Like"]))
    
    # Confusion matrix
    plt.figure()
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Dislike", "Like"],
                yticklabels=["Dislike", "Like"])
    plt.title("Confusion Matrix (CPU Fallback)")
    plt.savefig("confusion_matrix_fallback.png")
    
    # Sanity checks
    print("\n🧪 Sanity Checks:")
    test_samples = [
        ("I love this product!", "Like"),
        ("Worst experience ever", "Dislike"),
        ("Perfect, just what I didn't need", "Dislike"),
        ("Great job breaking it", "Dislike")
    ]
    
    for text, expected in test_samples:
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_label = "Like" if torch.argmax(probs) == 1 else "Dislike"
        print(f"'{text}'\n  Expected: {expected}, Predicted: {pred_label}")
    
    # Save model
    print("\n💾 Saving final model...")
    model.save_pretrained("./sarcasm_sentiment_model")
    tokenizer.save_pretrained("./sarcasm_sentiment_model")

# Main training function
def trainer(model_name: str = "distilroberta-base", use_fallback=False):
    """End-to-end training with fallback option"""
    if use_fallback:
        return train_model_cpu_fallback()
        
    print("\n🔄 Loading and preprocessing dataset...")
    try:
        raw_data = pre_labeled_datasets()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Create dataset
        dataset = Dataset.from_list(raw_data)
        dataset = dataset.map(
            lambda x: tokenize(x, tokenizer, max_length=64), 
            batched=True
        )
        dataset = dataset.remove_columns(['text'])
        
        # Train-test split
        print("\n📊 Splitting dataset...")
        dataset = dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
        
        # Initialize model
        print("\n🤖 Initializing model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2,
            id2label={0: "Dislike", 1: "Like"}
        )
        model.to(torch_device)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            eval_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=200,
            learning_rate=2e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=2,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=50,
            report_to="none",
            load_best_model_at_end=True,
            fp16=torch_device.type != "cpu",
        )
        
        # Create Trainer
        trainer_obj = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            data_collator=DataCollatorWithPadding(tokenizer)
        )
        
        # Train model
        print("\n🚀 Starting training...")
        train_results = trainer_obj.train()
        
        # Evaluate
        print("\n📈 Evaluating model...")
        metrics = trainer_obj.evaluate()
        print("\n✅ Evaluation Results:", metrics)
        
        # Get predictions
        predictions = trainer_obj.predict(eval_dataset)
        y_true = predictions.label_ids
        y_pred = np.argmax(predictions.predictions, axis=-1)
        
        # Classification report
        print("\n📝 Classification Report:")
        print(classification_report(y_true, y_pred, target_names=["Dislike", "Like"]))
        
        # Confusion matrix
        plt.figure()
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Dislike", "Like"],
                    yticklabels=["Dislike", "Like"])
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png")
        
        # Sanity checks
        print("\n🧪 Sanity Checks:")
        test_samples = [
            ("I love this product!", "Like"),
            ("Worst experience ever", "Dislike"),
            ("Perfect, just what I didn't need", "Dislike"),
            ("Great job breaking it", "Dislike")
        ]
        
        for text, expected in test_samples:
            inputs = tokenizer(text, return_tensors="pt", truncation=True).to(torch_device)
            with torch.no_grad():
                outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_label = "Like" if torch.argmax(probs) == 1 else "Dislike"
            print(f"'{text}'\n  Expected: {expected}, Predicted: {pred_label}")
        
        # Save model
        print("\n💾 Saving final model...")
        model.save_pretrained("./sarcasm_sentiment_model")
        tokenizer.save_pretrained("./sarcasm_sentiment_model")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "MPS backend" in str(e):
            print(f"\n⚠️ Memory error: {str(e)[:200]}...")
            print("Switching to CPU fallback training...")
            train_model_cpu_fallback()
        else:
            raise e

# Main execution
if __name__ == "__main__":
    # Set seeds for reproducibility
    os.environ['PYTHONHASHSEED'] = '42'
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # First try standard training
    try:
        trainer(model_name="distilroberta-base")
    except Exception as e:
        print(f"\n⚠️ Training failed: {str(e)[:200]}...")
        print("Attempting CPU fallback training...")
        trainer(use_fallback=True)