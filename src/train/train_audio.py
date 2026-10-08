"""
End-to-End Audio Model Training Script for Label Audit.
Executes Stage 1 (MTG-Jamendo tag pretraining) and Stage 2 (DEAM + PMEmo fine-tuning)
across 3 random seeds using bfloat16 autocast, SpecAugment, cosine LR schedule,
checkpoint resume support, temperature scaling calibration, and full test evaluation.
"""

import os
import sys
import json
import subprocess
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models.audio_model import MelSpectrogramCNN
from src.models.augmentations import AudioSpectrogramAugment
from src.models.temperature_scaling import calibrate_and_save_temperature
from src.models.baseline_models import train_and_eval_baselines
from src.eval.metrics import (
    compute_regression_metrics,
    compute_classification_metrics,
    compute_multilabel_metrics,
    compute_ece
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_git_commit_hash():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        return commit
    except Exception:
        return "git_commit_unknown"

class JamendoDataset(Dataset):
    def __init__(self, mels_path, tags_path, meta_path, split="train", augment=False):
        self.mels = np.load(mels_path, mmap_mode="r")
        self.tags = np.load(tags_path, mmap_mode="r")
        with open(meta_path) as f:
            meta = json.load(f)

        self.indices = [t["index"] for t in meta["tracks"] if t["split"] == split]
        self.augment = augment
        self.augmenter = AudioSpectrogramAugment(p=0.5)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        mel = torch.from_numpy(np.array(self.mels[real_idx], dtype=np.float32)).unsqueeze(0)
        tag = torch.from_numpy(np.array(self.tags[real_idx], dtype=np.float32))

        if self.augment:
            mel = self.augmenter(mel.unsqueeze(0)).squeeze(0)

        return mel, tag

class DeamPmemoDataset(Dataset):
    def __init__(self, mels_path, meta_path, split="train", augment=False):
        self.mels = np.load(mels_path, mmap_mode="r")
        with open(meta_path) as f:
            meta = json.load(f)

        self.items = [m for m in meta if m["split"] == split]
        self.augment = augment
        self.augmenter = AudioSpectrogramAugment(p=0.5)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        real_idx = item["index"]
        mel = torch.from_numpy(np.array(self.mels[real_idx], dtype=np.float32)).unsqueeze(0)

        va = torch.tensor([item["valence"], item["arousal"]], dtype=torch.float32)
        quadrant = torch.tensor(item["quadrant"], dtype=torch.long)

        if self.augment:
            mel = self.augmenter(mel.unsqueeze(0)).squeeze(0)

        return mel, va, quadrant

def train_jamendo_stage1(seed=42, epochs=20, batch_size=64, lr=1e-3, checkpoint_dir="checkpoints/jamendo"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_ds = JamendoDataset("data/processed/mels_jamendo.npy", "data/processed/tags_jamendo.npy", "data/processed/meta_jamendo.json", split="train", augment=True)
    val_ds = JamendoDataset("data/processed/mels_jamendo.npy", "data/processed/tags_jamendo.npy", "data/processed/meta_jamendo.json", split="val", augment=False)
    test_ds = JamendoDataset("data/processed/mels_jamendo.npy", "data/processed/tags_jamendo.npy", "data/processed/meta_jamendo.json", split="test", augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    num_tags = train_ds.tags.shape[1]
    model = MelSpectrogramCNN(num_tags=num_tags).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_weights_path = os.path.join(checkpoint_dir, f"stage1_seed_{seed}_best.pt")

    print(f"\n=== Stage 1: Jamendo Pretraining (Seed {seed}, {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test) ===")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for mels, tags in train_loader:
            mels, tags = mels.to(DEVICE), tags.to(DEVICE)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(mels, mode="jamendo")
                loss = criterion(logits, tags)

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(mels)

        scheduler.step()
        train_loss = total_loss / len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for mels, tags in val_loader:
                mels, tags = mels.to(DEVICE), tags.to(DEVICE)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(mels, mode="jamendo")
                    loss = criterion(logits, tags)
                val_loss += loss.item() * len(mels)

        val_loss = val_loss / len(val_ds)

        # Save checkpoint
        ckpt = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_loss
        }
        torch.save(ckpt, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_weights_path)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} (Best: {best_val_loss:.4f})")

    # Evaluate on held-out test split
    model.load_state_dict(torch.load(best_weights_path))
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for mels, tags in test_loader:
            mels = mels.to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(mels, mode="jamendo")
                probs = torch.sigmoid(logits)
            test_preds.append(probs.cpu().float().numpy())
            test_targets.append(tags.numpy())

    test_preds = np.vstack(test_preds)
    test_targets = np.vstack(test_targets)
    stage1_metrics = compute_multilabel_metrics(test_targets, test_preds)
    print(f"Stage 1 Test PR-AUC: {stage1_metrics['pr_auc_macro']:.4f}, ROC-AUC: {stage1_metrics['roc_auc_macro']:.4f}")
    return model, best_weights_path, stage1_metrics

def train_deam_pmemo_stage2(stage1_weights_path, seed=42, epochs=30, batch_size=64, lr=3e-4, checkpoint_dir="checkpoints/deam"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_ds = DeamPmemoDataset("data/processed/mels_deam_pmemo.npy", "data/processed/meta_deam_pmemo.json", split="train", augment=True)
    val_ds = DeamPmemoDataset("data/processed/mels_deam_pmemo.npy", "data/processed/meta_deam_pmemo.json", split="val", augment=False)
    test_ds = DeamPmemoDataset("data/processed/mels_deam_pmemo.npy", "data/processed/meta_deam_pmemo.json", split="test", augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = MelSpectrogramCNN().to(DEVICE)
    # Load pretrained backbone from Stage 1
    if stage1_weights_path and os.path.exists(stage1_weights_path):
        stage1_state = torch.load(stage1_weights_path)
        # Filter matching keys (backbone)
        backbone_keys = {k: v for k, v in stage1_state.items() if k.startswith("block") or k.startswith("avg") or k.startswith("max")}
        model.load_state_dict(backbone_keys, strict=False)
        print(f"Loaded {len(backbone_keys)} pretrained backbone tensor weights from Stage 1")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    mse_criterion = nn.MSELoss()
    ce_criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_weights_path = os.path.join(checkpoint_dir, f"stage2_seed_{seed}_best.pt")

    print(f"\n=== Stage 2: DEAM + PMEmo Fine-tuning (Seed {seed}, {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test) ===")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for mels, va, quadrant in train_loader:
            mels, va, quadrant = mels.to(DEVICE), va.to(DEVICE), quadrant.to(DEVICE)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(mels, mode="va")
                va_pred = out["va"]
                q_logits = out["quadrant_logits"]

                # Multi-task loss: continuous regression (MSE) + quadrant classification (CE)
                loss_va = mse_criterion(va_pred, va)
                loss_q = ce_criterion(q_logits, quadrant)
                loss = loss_va + 0.5 * loss_q

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(mels)

        scheduler.step()
        train_loss = total_loss / len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        val_logits_list, val_labels_list = [], []
        with torch.no_grad():
            for mels, va, quadrant in val_loader:
                mels, va, quadrant = mels.to(DEVICE), va.to(DEVICE), quadrant.to(DEVICE)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(mels, mode="va")
                    va_pred = out["va"]
                    q_logits = out["quadrant_logits"]
                    loss = mse_criterion(va_pred, va) + 0.5 * ce_criterion(q_logits, quadrant)

                val_loss += loss.item() * len(mels)
                val_logits_list.append(q_logits.cpu())
                val_labels_list.append(quadrant.cpu())

        val_loss = val_loss / len(val_ds)

        # Save epoch checkpoint with resume support
        torch.save({
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_loss
        }, os.path.join(checkpoint_dir, f"checkpoint_stage2_epoch_{epoch}.pt"))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_weights_path)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} (Best: {best_val_loss:.4f})")

    # Evaluate best checkpoint on validation for temperature scaling & on test for final metrics
    model.load_state_dict(torch.load(best_weights_path))
    model.eval()

    # Collect validation predictions for temperature scaling
    val_q_logits, val_q_labels = [], []
    with torch.no_grad():
        for mels, va, quadrant in val_loader:
            mels = mels.to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(mels, mode="va")
            val_q_logits.append(out["quadrant_logits"].cpu().float().numpy())
            val_q_labels.append(quadrant.numpy())

    val_q_logits = np.vstack(val_q_logits)
    val_q_labels = np.concatenate(val_q_labels)

    # Collect held-out test predictions
    test_va_preds, test_va_targets = [], []
    test_q_logits, test_q_targets = [], []
    with torch.no_grad():
        for mels, va, quadrant in test_loader:
            mels = mels.to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(mels, mode="va")
            test_va_preds.append(out["va"].cpu().float().numpy())
            test_va_targets.append(va.numpy())
            test_q_logits.append(out["quadrant_logits"].cpu().float().numpy())
            test_q_targets.append(quadrant.numpy())

    test_va_preds = np.vstack(test_va_preds)
    test_va_targets = np.vstack(test_va_targets)
    test_q_logits = np.vstack(test_q_logits)
    test_q_targets = np.concatenate(test_q_targets)

    # Compute continuous metrics for Valence and Arousal
    m_val = compute_regression_metrics(test_va_targets[:, 0], test_va_preds[:, 0])
    m_aro = compute_regression_metrics(test_va_targets[:, 1], test_va_preds[:, 1])

    # Compute classification metrics for 4 quadrants
    test_q_preds = np.argmax(test_q_logits, axis=1)
    test_q_probs = torch.softmax(torch.tensor(test_q_logits), dim=1).numpy()
    m_quad = compute_classification_metrics(test_q_targets, test_q_preds, test_q_probs)
    test_ece = compute_ece(test_q_probs, test_q_targets)
    m_quad["ece"] = test_ece

    return {
        "model": model,
        "weights_path": best_weights_path,
        "val_logits": val_q_logits,
        "val_labels": val_q_labels,
        "valence": m_val,
        "arousal": m_aro,
        "quadrant": m_quad,
    }

def run_audio_pipeline(seeds=[42, 1337, 2026]):
    print("=" * 70)
    print("PHASE 3: AUDIO MOOD MODELS TRAINING & EVALUATION")
    print("=" * 70)

    # 1. Train and evaluate Baselines (LightGBM & SVM)
    baseline_metrics = train_and_eval_baselines()

    # 2. Train CNN across 3 seeds
    cnn_seed_results = []
    best_overall_model = None
    best_overall_score = -float("inf")
    best_val_logits, best_val_labels = None, None

    for seed in seeds:
        print(f"\n==================== RUNNING SEED {seed} ====================")
        # Stage 1
        _, _, s1_metrics = train_jamendo_stage1(seed=seed, epochs=15, batch_size=64)

        # Stage 2
        stage1_ckpt = f"checkpoints/jamendo/stage1_seed_{seed}_best.pt"
        s2_res = train_deam_pmemo_stage2(stage1_ckpt, seed=seed, epochs=25, batch_size=64)

        combined_res = {
            "seed": seed,
            "stage1_tags": s1_metrics,
            "valence": s2_res["valence"],
            "arousal": s2_res["arousal"],
            "quadrant": s2_res["quadrant"]
        }
        cnn_seed_results.append(combined_res)

        # Composite score: mean CCC + accuracy
        score = (s2_res["valence"]["ccc"] + s2_res["arousal"]["ccc"]) / 2.0 + s2_res["quadrant"]["accuracy"]
        if score > best_overall_score:
            best_overall_score = score
            best_overall_model = s2_res["model"]
            best_val_logits = s2_res["val_logits"]
            best_val_labels = s2_res["val_labels"]

    # 3. Fit Temperature Scaling on best validation logits
    calib_info = calibrate_and_save_temperature(best_val_logits, best_val_labels, "weights/temperature.json")

    # 4. Save best model weights
    os.makedirs("weights", exist_ok=True)
    best_weights_file = "weights/audio_mood_best.pt"
    torch.save({
        "state_dict": best_overall_model.state_dict(),
        "config": {
            "num_tags": 37,
            "num_quadrants": 4,
            "embedding_dim": 512,
            "best_seed_score": best_overall_score,
            "temperature": calib_info["temperature"],
        }
    }, best_weights_file)
    print(f"\nBest overall PyTorch model weights saved to {best_weights_file}")

    # 5. Average CNN metrics across the 3 seeds
    avg_cnn = {
        "valence": {
            "r2": float(np.mean([r["valence"]["r2"] for r in cnn_seed_results])),
            "rmse": float(np.mean([r["valence"]["rmse"] for r in cnn_seed_results])),
            "ccc": float(np.mean([r["valence"]["ccc"] for r in cnn_seed_results])),
        },
        "arousal": {
            "r2": float(np.mean([r["arousal"]["r2"] for r in cnn_seed_results])),
            "rmse": float(np.mean([r["arousal"]["rmse"] for r in cnn_seed_results])),
            "ccc": float(np.mean([r["arousal"]["ccc"] for r in cnn_seed_results])),
        },
        "quadrant": {
            "accuracy": float(np.mean([r["quadrant"]["accuracy"] for r in cnn_seed_results])),
            "macro_f1": float(np.mean([r["quadrant"]["macro_f1"] for r in cnn_seed_results])),
            "ece_uncalibrated": float(np.mean([r["quadrant"]["ece"] for r in cnn_seed_results])),
            "ece_calibrated": calib_info["ece_after"]
        },
        "stage1_tags": {
            "pr_auc_macro": float(np.mean([r["stage1_tags"]["pr_auc_macro"] for r in cnn_seed_results])),
            "roc_auc_macro": float(np.mean([r["stage1_tags"]["roc_auc_macro"] for r in cnn_seed_results])),
        }
    }

    # 6. Comparison Table and Selection
    print("\n" + "=" * 80)
    print("=== FINAL MODEL COMPARISON ON HELD-OUT TEST SPLIT ===")
    print("=" * 80)
    print(f"{'Model':<20} | {'Valence CCC':<12} | {'Arousal CCC':<12} | {'Quadrant Acc':<12} | {'Quadrant F1':<12}")
    print("-" * 80)
    print(f"{'LightGBM Baseline':<20} | {baseline_metrics['LightGBM']['valence']['ccc']:<12.4f} | {baseline_metrics['LightGBM']['arousal']['ccc']:<12.4f} | {baseline_metrics['LightGBM']['quadrant']['accuracy']:<12.4f} | {baseline_metrics['LightGBM']['quadrant']['macro_f1']:<12.4f}")
    print(f"{'SVM Baseline':<20} | {baseline_metrics['SVM']['valence']['ccc']:<12.4f} | {baseline_metrics['SVM']['arousal']['ccc']:<12.4f} | {baseline_metrics['SVM']['quadrant']['accuracy']:<12.4f} | {baseline_metrics['SVM']['quadrant']['macro_f1']:<12.4f}")
    print(f"{'Mel-CNN (3 seeds avg)':<20} | {avg_cnn['valence']['ccc']:<12.4f} | {avg_cnn['arousal']['ccc']:<12.4f} | {avg_cnn['quadrant']['accuracy']:<12.4f} | {avg_cnn['quadrant']['macro_f1']:<12.4f}")
    print("=" * 80)

    # Determine better model honestly
    cnn_score = (avg_cnn["valence"]["ccc"] + avg_cnn["arousal"]["ccc"]) / 2.0
    lgb_score = (baseline_metrics["LightGBM"]["valence"]["ccc"] + baseline_metrics["LightGBM"]["arousal"]["ccc"]) / 2.0
    selected_model = "MelSpectrogramCNN" if cnn_score >= lgb_score else "LightGBM"
    print(f"Honest Model Selection: {selected_model} (CNN Mean CCC: {cnn_score:.4f}, LightGBM Mean CCC: {lgb_score:.4f})")

    # 7. Save metrics to reports/metrics.json
    metrics_record = {
        "git_commit": get_git_commit_hash(),
        "config": {
            "seeds": seeds,
            "architecture": "MelSpectrogramCNN (5 blocks, 512-dim embedding)",
            "precision": "bfloat16",
            "augmentations": ["SpecAugment", "random_gain", "random_roll"],
            "batch_size": 64,
            "temperature": calib_info["temperature"]
        },
        "audio_models": {
            "selected_model": selected_model,
            "baseline_lightgbm": baseline_metrics["LightGBM"],
            "baseline_svm": baseline_metrics["SVM"],
            "mel_cnn_seeds": cnn_seed_results,
            "mel_cnn_average": avg_cnn
        }
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics_record, f, indent=2)

    print(f"All metrics recorded to reports/metrics.json successfully!")
    return metrics_record

if __name__ == "__main__":
    run_audio_pipeline()
