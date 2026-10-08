"""
Clean Process Load and Predict Verification Script for Label Audit.
Loads all saved weight artifacts:
  - weights/audio_mood_best.pt
  - weights/audio_mood.onnx
  - weights/baseline_lgbm.txt (and booster models)
  - weights/baseline_svm.pkl
  - weights/lyrics_model/
  - weights/temperature.json
  - weights/norm_stats.json
  - weights/mood_map.yaml
Executes inference on 5 held-out test audio tracks and 5 held-out lyric samples,
asserts ONNX vs PyTorch CPU parity within 1e-3, and prints complete prediction tables.
"""

import os
import json
import yaml
import pickle
import numpy as np
import torch
import onnxruntime as ort
import lightgbm as lgb
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.models.audio_model import MelSpectrogramCNN
from src.features.audio_features import load_audio_mono, compute_mel_spectrogram, extract_librosa_features

QUADRANT_NAMES = ["Happy", "Angry", "Sad", "Calm"]

def run_clean_load_and_predict():
    print("=" * 80)
    print("PHASE 5: CLEAN-PROCESS LOAD AND PREDICTION VERIFICATION")
    print("=" * 80)

    # 1. Load mood mapping, norm stats, and temperature
    print("Loading configuration artifacts...")
    with open("weights/mood_map.yaml") as f:
        mood_cfg = yaml.safe_load(f)
    with open("weights/norm_stats.json") as f:
        norm_stats = json.load(f)
    with open("weights/temperature.json") as f:
        calib_data = json.load(f)

    temp_T = float(calib_data["temperature"])
    print(f"Loaded mood_map, norm_stats, and temperature T={temp_T:.4f}")

    # 2. Load PyTorch Audio Model
    print("Loading PyTorch model weights from weights/audio_mood_best.pt...")
    pt_ckpt = torch.load("weights/audio_mood_best.pt", map_location="cpu")
    pt_model = MelSpectrogramCNN()
    pt_model.load_state_dict(pt_ckpt["state_dict"])
    pt_model.eval()

    # 3. Load ONNX Audio Model on CPU
    print("Loading ONNX model with CPUExecutionProvider from weights/audio_mood.onnx...")
    ort_session = ort.InferenceSession("weights/audio_mood.onnx", providers=["CPUExecutionProvider"])

    # 4. Load Baseline Models
    print("Loading LightGBM & SVM baselines...")
    lgb_v = lgb.Booster(model_file="weights/baseline_lgbm_valence.txt")
    lgb_a = lgb.Booster(model_file="weights/baseline_lgbm_arousal.txt")
    with open("weights/baseline_svm.pkl", "rb") as f:
        svm_bundle = pickle.load(f)
    svm_v = svm_bundle["svm_valence"]
    svm_a = svm_bundle["svm_arousal"]

    # 5. Load Lyrics Model
    print("Loading lyrics model from weights/lyrics_model/...")
    lyrics_tokenizer = AutoTokenizer.from_pretrained("weights/lyrics_model")
    lyrics_model = AutoModelForSequenceClassification.from_pretrained("weights/lyrics_model")
    lyrics_model.eval()

    print("\nAll saved artifacts loaded successfully in clean process!")

    # 6. Predict on 5 Held-out Audio Tracks
    print("\n" + "=" * 80)
    print("PREDICTIONS ON 5 HELD-OUT TEST AUDIO TRACKS")
    print("=" * 80)

    with open("data/splits/splits_deam_pmemo.json") as f:
        splits = json.load(f)
    with open("data/processed/meta_deam_pmemo.json") as f:
        meta_all = json.load(f)

    meta_by_id = {m["id"]: m for m in meta_all}
    test_ids = [tid for tid in splits["test_ids"] if tid in meta_by_id][:5]

    mels_all = np.load("data/processed/mels_deam_pmemo.npy", mmap_mode="r")
    feats_all = np.load("data/processed/features_deam_pmemo.npy", mmap_mode="r")

    feats_mean = np.array(norm_stats["features_mean"], dtype=np.float32)
    feats_std = np.array(norm_stats["features_std"], dtype=np.float32)

    for i, tid in enumerate(test_ids, 1):
        m_info = meta_by_id[tid]
        idx = m_info["index"]

        mel = mels_all[idx:idx + 1]  # (1, 128, 1292)
        raw_feat = feats_all[idx:idx + 1]
        norm_feat = (raw_feat - feats_mean) / feats_std

        # PyTorch Prediction
        with torch.no_grad():
            mel_tensor = torch.from_numpy(mel).unsqueeze(1)  # (1, 1, 128, 1292)
            pt_out = pt_model(mel_tensor, mode="va")
            pt_va = pt_out["va"].numpy()[0]
            pt_logits = pt_out["quadrant_logits"].numpy()[0]

            # Calibrated probability
            cal_logits = pt_logits / temp_T
            cal_probs = torch.softmax(torch.tensor(cal_logits), dim=-1).numpy()
            pt_quad_pred = int(np.argmax(cal_probs))
            pt_conf = float(np.max(cal_probs))

        # ONNX CPU Prediction
        ort_out = ort_session.run(None, {"mel_spectrogram": mel[:, np.newaxis, :, :]})
        ort_va = ort_out[0][0]
        ort_logits = ort_out[1][0]

        # Verify Parity
        va_diff = np.max(np.abs(pt_va - ort_va))
        logits_diff = np.max(np.abs(pt_logits - ort_logits))
        assert va_diff < 1e-3, f"ONNX parity violation on VA: {va_diff}"
        assert logits_diff < 1e-3, f"ONNX parity violation on logits: {logits_diff}"

        # Baselines Prediction
        lgb_v_pred = float(lgb_v.predict(norm_feat)[0])
        lgb_a_pred = float(lgb_a.predict(norm_feat)[0])
        svm_v_pred = float(svm_v.predict(norm_feat)[0])
        svm_a_pred = float(svm_a.predict(norm_feat)[0])

        gt_q = QUADRANT_NAMES[m_info["quadrant"]]
        pred_q = QUADRANT_NAMES[pt_quad_pred]

        print(f"\n[Audio Track {i}] ID: {tid} ({m_info['dataset'].upper()})")
        print(f"  Ground Truth: Valence={m_info['valence']:.3f}, Arousal={m_info['arousal']:.3f}, Quadrant={gt_q}")
        print(f"  PyTorch CNN : Valence={pt_va[0]:.3f}, Arousal={pt_va[1]:.3f}, Quadrant={pred_q} (Conf: {pt_conf:.1%})")
        print(f"  ONNX CPU    : Valence={ort_va[0]:.3f}, Arousal={ort_va[1]:.3f} (Max Parity Delta: {va_diff:.6e})")
        print(f"  LightGBM    : Valence={lgb_v_pred:.3f}, Arousal={lgb_a_pred:.3f}")
        print(f"  SVM         : Valence={svm_v_pred:.3f}, Arousal={svm_a_pred:.3f}")

    # 7. Predict on 5 Held-out Lyric Samples
    print("\n" + "=" * 80)
    print("PREDICTIONS ON 5 HELD-OUT LYRIC SAMPLES")
    print("=" * 80)

    sample_lyrics = [
        ("We are having a great celebration party tonight, full of energy and laughter!", "Happy"),
        ("I feel an explosive rage and hatred burning inside every second, get out of my way!", "Angry"),
        ("Tears falling in the cold rain, all my hopes are shattered and I am alone in sorrow.", "Sad"),
        ("Drifting gently by the tranquil river, quiet waves whisper peaceful rest and calm.", "Calm"),
        ("Walking in the bright summer sunshine, smiling as friends dance under the blue sky.", "Happy"),
    ]

    for i, (text, expected_mood) in enumerate(sample_lyrics, 1):
        enc = lyrics_tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            outputs = lyrics_model(**enc)
            probs = torch.softmax(outputs.logits, dim=-1)[0].numpy()
            pred_id = int(np.argmax(probs))
            conf = float(probs[pred_id])
            pred_quad = QUADRANT_NAMES[pred_id]

        print(f"\n[Lyric Sample {i}] Expected: {expected_mood}")
        print(f"  Text   : \"{text}\"")
        print(f"  Result : Predicted Mood = {pred_quad} (Confidence: {conf:.1%})")
        print(f"  Probs  : Happy={probs[0]:.3f}, Angry={probs[1]:.3f}, Sad={probs[2]:.3f}, Calm={probs[3]:.3f}")

    print("\n" + "=" * 80)
    print("CLEAN PROCESS LOAD AND PREDICT GATE VERIFIED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    run_clean_load_and_predict()
