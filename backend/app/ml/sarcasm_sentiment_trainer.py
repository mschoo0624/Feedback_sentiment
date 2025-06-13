import random
import json
from typing import List, Dict
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
import torch
from sklearn.metrics import accuracy_score, f1_score

# Stage 1 (Load and combine datasets.)
def pre_labeled_datasets() -> List[Dict]:
    # This ds contains tweets labeled for sarcasm detection. 
    sarcasm_ds = load_dataset("tweet_eval", "irony")
    # This pre_labeled dataset contains Amazon product reviews with positive/negative labels
    sentiment_ds = load_dataset("amazon_polarity")
    
    # Empty List. 
    data = []
    
    for example in sarcasm_ds["train"]:
        text = example["text"]
        # Label 1 = sarcastic (assumed negative) -> "Dislike"
        # Label 0 = not sarcastic (assumed positive) -> "Like"
        label = "Dislike" if example["label"] == 1 else "Like"  # Fixed: lowercase 'label'
        data.append({"text": text, "label": label})
    
    for example in sentiment_ds["train"][:10000]:
        text = example["content"]  # Fixed: Amazon polarity uses 'content', not 'text'
        # Label 0 = negative -> "Dislike", Label 1 = positive -> "Like"
        label = "Like" if example["label"] == 1 else "Dislike"  # Fixed: correct mapping and lowercase 'label'
        data.append({"text": text, "label": label})
    
    random.shuffle(data)
    return data

# Stage 2 (Creating a map for the label)
LABEL_MAP = {"Dislike": 0, "Like": 1}

# Stage 3 (Tokenization)
def tokenize(example, tokenizer):
    encoding = tokenizer(
        example['text'],
        truncation=True,
        padding=False,
        max_length=256
    )
    encoding['label'] = LABEL_MAP[example['label']]
    return encoding

# Stage 4 (Evaluate metrics)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # This line converts the raw logits into actual predicted class labels.
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    # Measures how many predictions were exactly right.
    acc = accuracy_score(labels, predictions)
    # Balances precision and recall 
    f1 = f1_score(labels, predictions)
    
    return {"accuracy": acc, "f1": f1}

# Stage 5 (Train Function) 
def trainer(model_name: str = "roberta-base"): # Pretrained model to fine-tune.
    print("\n🔄 Loading and preprocessing dataset...")
    raw_data = pre_labeled_datasets()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(lambda x: tokenize(x, tokenizer))
    
    # Remove unused columns to avoid conflicts during training
    dataset = dataset.remove_columns(['text'])
    
    print("\n📊 Splitting dataset into training and testing...")
    dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    
    print("\n🤖 Initializing model...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        report_to="none",
        load_best_model_at_end=True,  # Added: Load best model based on eval metrics
    )
    
    print("\n🚀 Starting training...")
    trainer_obj = Trainer(  # Renamed to avoid confusion with function name
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer)
    )

    trainer_obj.train()

    print("\n📈 Evaluating model...")
    metrics = trainer_obj.evaluate()
    print("\n✅ Evaluation Results:", metrics)

    print("\n💾 Saving final model to ./sarcasm_sentiment_model")
    model.save_pretrained("./sarcasm_sentiment_model")
    tokenizer.save_pretrained("./sarcasm_sentiment_model")
    
if __name__ == "__main__":
    trainer()