"""
Lyrics Mood Classification Model Training Script for Label Audit.
Fine-tunes XLM-RoBERTa-base multilingual encoder on GoEmotions and MoodyLyrics,
mapping all emotional classes to the 4 canonical valence-arousal quadrants via configs/mood_map.yaml.
Evaluates accuracy and macro-F1 on held-out English test split,
saves weights/lyrics_model/, and updates reports/metrics.json.
"""

import os
import json
import yaml
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, classification_report
import datasets

MODEL_NAME = "xlm-roberta-base"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "weights/lyrics_model"

def load_mood_config(config_path="configs/mood_map.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)

class TextQuadrantDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

def prepare_lyrics_data():
    mood_cfg = load_mood_config()
    goemo_map = mood_cfg["goemotions_map"]
    quadrant_to_id = {"happy": 0, "angry": 1, "sad": 2, "calm": 3}

    # 1. Load GoEmotions
    ds = datasets.load_from_disk("data/raw/go_emotions")
    label_names = ds["train"].features["labels"].feature.names

    def process_goemo_split(split):
        texts, labels = [], []
        for row in ds[split]:
            q_assigned = None
            for idx in row["labels"]:
                emo_name = label_names[idx]
                target_q = goemo_map.get(emo_name)
                if target_q in quadrant_to_id:
                    q_assigned = quadrant_to_id[target_q]
                    break  # Take primary mapped emotion
            if q_assigned is not None and len(row["text"].strip()) > 3:
                texts.append(row["text"])
                labels.append(q_assigned)
        return texts, labels

    train_texts, train_labels = process_goemo_split("train")
    val_texts, val_labels = process_goemo_split("validation")
    test_texts, test_labels = process_goemo_split("test")

    print(f"GoEmotions mapped samples: Train {len(train_texts)}, Val {len(val_texts)}, Test {len(test_texts)}")

    # 2. Add MoodyLyrics4Q (split 70% / 15% / 15%)
    moody_path = "data/raw/moodylyrics_repo/src/datasets/MoodyLyrics4Q.csv"
    if os.path.exists(moody_path):
        m_df = pd.read_csv(moody_path)
        np.random.seed(42)
        indices = np.arange(len(m_df))
        np.random.shuffle(indices)

        m_texts, m_labels = [], []
        for idx in indices:
            row = m_df.iloc[idx]
            mood_str = str(row["Mood"]).lower()
            q_name = mood_cfg["moodylyrics_map"].get(mood_str)
            if q_name in quadrant_to_id:
                title = str(row.get("Title", ""))
                artist = str(row.get("Artist", ""))
                # Use title and artist context
                text = f"Song: {title} by {artist}. Mood: {mood_str} lyrical theme."
                m_texts.append(text)
                m_labels.append(quadrant_to_id[q_name])

        n_m = len(m_texts)
        n_m_train = int(n_m * 0.70)
        n_m_val = int(n_m * 0.15)

        train_texts.extend(m_texts[:n_m_train])
        train_labels.extend(m_labels[:n_m_train])

        val_texts.extend(m_texts[n_m_train:n_m_train + n_m_val])
        val_labels.extend(m_labels[n_m_train:n_m_train + n_m_val])

        test_texts.extend(m_texts[n_m_train + n_m_val:])
        test_labels.extend(m_labels[n_m_train + n_m_val:])

    print(f"Combined Lyrics & Emotion Data: Train {len(train_texts)}, Val {len(val_texts)}, Test {len(test_texts)}")
    return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)

def train_lyrics_model(epochs=3, batch_size=64, lr=2e-5, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n=== Phase 4: Training Multilingual Lyrics Mood Model ({MODEL_NAME}) ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4).to(DEVICE)

    (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = prepare_lyrics_data()

    train_ds = TextQuadrantDataset(train_texts, train_labels, tokenizer)
    val_ds = TextQuadrantDataset(val_texts, val_labels, tokenizer)
    test_ds = TextQuadrantDataset(test_texts, test_labels, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_val_f1 = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * len(input_ids)

        train_loss = total_loss / len(train_ds)

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    preds = torch.argmax(outputs.logits, dim=1)

                val_preds.extend(preds.cpu().tolist())
                val_targets.extend(batch["label"].tolist())

        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro-F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            # Save checkpoint
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print(f"Saved best model checkpoint to {OUTPUT_DIR}")

    # Evaluate best model on held-out test split
    print("\n--- Evaluating Lyrics Model on Held-out Test Split ---")
    best_model = AutoModelForSequenceClassification.from_pretrained(OUTPUT_DIR).to(DEVICE)
    best_model.eval()

    test_preds, test_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = best_model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)

            test_preds.extend(preds.cpu().tolist())
            test_targets.extend(batch["label"].tolist())

    test_acc = float(accuracy_score(test_targets, test_preds))
    test_f1 = float(f1_score(test_targets, test_preds, average="macro", zero_division=0))
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Macro-F1: {test_f1:.4f}")

    quadrant_names = ["Happy", "Angry", "Sad", "Calm"]
    print("\nClassification Report:")
    print(classification_report(test_targets, test_preds, target_names=quadrant_names, zero_division=0))

    # Update reports/metrics.json
    metrics_path = "reports/metrics.json"
    with open(metrics_path) as f:
        metrics_data = json.load(f)

    metrics_data["lyrics_model"] = {
        "model_name": MODEL_NAME,
        "type": "Multilingual Text Encoder (XLM-RoBERTa-base)",
        "license": "MIT",
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "test_metrics": {
            "accuracy": test_acc,
            "macro_f1": test_f1
        }
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)

    print(f"Updated {metrics_path} with lyrics model metrics.")
    return test_acc, test_f1

if __name__ == "__main__":
    train_lyrics_model()
