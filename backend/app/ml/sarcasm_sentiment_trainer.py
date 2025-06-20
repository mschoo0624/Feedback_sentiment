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

# Visualization config for the dataset distributions or confusion matrices.
sns.set(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 8)

# Label map (Like or Dislike)
LABEL_MAP = {"Dislike": 0, "Like": 1}

# Custom sarcasm examples (from Reddit/news/tweets, abbreviated)
custom_sarcasm_examples = [
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
        ("Save your money, buy something else", "Dislike")
]

# Tokenize it so, that transformer model can understand it. 
def tokenize_function(examples, tokenizer, max_length=64):
    cleaned = [' '.join(t.split()) for t in examples['text']]
    enc = tokenizer(cleaned, truncation=True, padding=False, max_length=max_length)
    enc['labels'] = [LABEL_MAP[label] for label in examples['label']]
    return enc

# Evaluation for understanding how well this ML machine sentiment classifier is performing. 
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_weighted": f1_score(labels, preds, average='weighted'),
        "f1_macro": f1_score(labels, preds, average='macro')
    }

# Dataset builder
def create_diverse_dataset(max_per_source=1000):
    logger.info("📦 Loading datasets...")
    # Empty data set to store the pre-trained datasets. 
    ds_list = []

    def wrap(data, label, source):
        return [{"text": d, "label": label, "source": source} for d in data]

    # Amazon
    amazon = load_dataset("amazon_polarity", split="train")
    print("Debugging: Amazon DataSets have loaded!!!")
    ds_list.extend(wrap([x["content"] for x in amazon if x["label"] == 1][:max_per_source], "Like", "amazon"))
    ds_list.extend(wrap([x["content"] for x in amazon if x["label"] == 0][:max_per_source], "Dislike", "amazon"))

    # IMDB
    imdb = load_dataset("imdb", split="train")
    print("Debugging: IMDB DataSets have loaded!!!")
    ds_list.extend(wrap([x["text"][:500] for x in imdb if x["label"] == 1][:max_per_source], "Like", "imdb"))
    ds_list.extend(wrap([x["text"][:500] for x in imdb if x["label"] == 0][:max_per_source], "Dislike", "imdb"))

    # SST-2
    sst = load_dataset("sst2", split="train")
    print("Debugging: SST DataSets have loaded!!!")
    ds_list.extend([{"text": x["sentence"], "label": "Like" if x["label"] else "Dislike", "source": "sst"} for x in sst.select(range(max_per_source * 2))])

    # Adding the customized sarcastic examples. 
    ds_list.extend([{"text": t, "label": l, "source": "custom"} for t, l in custom_sarcasm_examples])

    # Making sure the "Like" and "Dislike" datasets are balenced out and shuffled it before training. 
    df = pd.DataFrame(ds_list)
    min_count = df.label.value_counts().min()
    df = pd.concat([
        df[df.label == "Like"].sample(n=min_count, random_state=42),
        df[df.label == "Dislike"].sample(n=min_count, random_state=42)
    ])
    df = df.sample(frac=1).reset_index(drop=True)

    # Plot distribution
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    sns.countplot(data=df, x='label')
    plt.title('Label Distribution')

    plt.subplot(1, 3, 2)
    sns.countplot(data=df, x='source')
    plt.title('Source Distribution')
    plt.xticks(rotation=45)

    plt.subplot(1, 3, 3)
    sns.countplot(data=df, x='source', hue='label')
    plt.title('Source vs Label Distribution')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig("dataset_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

    return df.to_dict("records")

# Training logic
def train_model(model_name="distilbert-base-uncased"):
    # it will fine-tune a DistilBERT model unless you pass something else
    logger.info("🔄 Preparing data...")
    raw = create_diverse_dataset()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    data = Dataset.from_list(raw)
    data = data.map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=["text", "label", "source"])
    split = data.train_test_split(test_size=0.3, seed=42)
    val_test = split["test"].train_test_split(test_size=0.5, seed=42)
    train, val, test = split["train"], val_test["train"], val_test["test"]

    logger.info("🤖 Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "Dislike", 1: "Like"},
        label2id={"Dislike": 0, "Like": 1},
        ignore_mismatched_sizes=True
    ).to(torch_device)

    args = TrainingArguments(
        output_dir="./results",
        do_eval=True,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        save_total_limit=2,
        fp16=torch_device.type == "cuda",
        bf16=False
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer)
    )

    logger.info("🚀 Training...")
    trainer.train()

    logger.info("📊 Evaluating...")
    preds = trainer.predict(test)
    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=-1)

    print("\n📝 Report:")
    print(classification_report(y_true, y_pred, target_names=["Dislike", "Like"]))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Dislike", "Like"], yticklabels=["Dislike", "Like"])
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

    logger.info("💾 Saving model...")
    model.save_pretrained("./improved_sentiment_model")
    tokenizer.save_pretrained("./improved_sentiment_model")

    return model, tokenizer

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    try:
        train_model("distilbert-base-uncased")
    except Exception as e:
        logger.error(f"Primary model failed: {e}")
        logger.info("Fallback: trying ALBERT")
        train_model("albert-base-v2")
