"""
Threshold Calibration Engine for Label Audit Mismatch Comparator.
Evaluates exclusively on the validation split (data/splits/splits_deam_pmemo.json['val_ids']).
Creates synthetic mismatches by swapping labels on a random 15% of validation tracks (seed=42).
Sweeps thresholds to maximize F1-score and serializes the complete curve and metadata
to backend/calibration/threshold.json.
Guarantees zero test-split data leakage.
"""

import os
import json
import numpy as np

from backend.app.inference.audio import AudioPredictor, QUADRANT_NAMES
from backend.app.inference.label import QUADRANT_CENTROIDS

def calibrate_mismatch_threshold(seed=42, swap_rate=0.15, output_file="backend/calibration/threshold.json"):
    print("=" * 80)
    print("PHASE 2: MISMATCH COMPARATOR THRESHOLD CALIBRATION")
    print("=" * 80)

    # 1. Load splits and verify validation data presence
    splits_path = "data/splits/splits_deam_pmemo.json"
    meta_path = "data/processed/meta_deam_pmemo.json"
    mels_path = "data/processed/mels_deam_pmemo.npy"

    if not os.path.exists(splits_path) or not os.path.exists(meta_path) or not os.path.exists(mels_path):
        raise FileNotFoundError("Validation datasets or splits missing! Cannot calibrate threshold.")

    with open(splits_path) as f:
        splits = json.load(f)
    with open(meta_path) as f:
        meta_all = json.load(f)

    val_ids = set(splits["val_ids"])
    test_ids = set(splits["test_ids"])

    # Strict isolation check
    assert len(val_ids & test_ids) == 0, "FATAL: Overlap between validation and test split!"
    print(f"Validation tracks available: {len(val_ids)} (Test tracks strictly isolated: {len(test_ids)})")

    meta_by_id = {m["id"]: m for m in meta_all}
    val_items = [meta_by_id[vid] for vid in splits["val_ids"] if vid in meta_by_id]
    mels_all = np.load(mels_path, mmap_mode="r")

    # 2. Run Audio Model Inference on Validation Tracks
    audio_predictor = AudioPredictor()
    print("Running ONNX audio predictions on validation tracks...")
    
    val_preds = []
    for item in val_items:
        idx = item["index"]
        mel = mels_all[idx]
        pred = audio_predictor.predict_mel_array(mel)
        val_preds.append({
            "id": item["id"],
            "true_quadrant_idx": int(item["quadrant"]),
            "true_quadrant_name": QUADRANT_NAMES[int(item["quadrant"])],
            "pred_v": pred["valence"],
            "pred_a": pred["arousal"],
            "quad_probs": pred["quadrant_probs"]
        })

    n_samples = len(val_preds)
    n_swap = int(round(n_samples * swap_rate))
    print(f"Total validation samples: {n_samples}. Synthetic swaps (15%): {n_swap}")

    # 3. Create synthetic mismatches with fixed seed
    np.random.seed(seed)
    swap_indices = set(np.random.choice(n_samples, size=n_swap, replace=False))

    assigned_labels = []
    ground_truth_mismatch = []

    for i, item in enumerate(val_preds):
        true_q = item["true_quadrant_name"]
        if i in swap_indices:
            # Swap to a different quadrant
            candidate_quads = [q for q in QUADRANT_NAMES if q != true_q]
            swapped_q = np.random.choice(candidate_quads)
            assigned_labels.append(swapped_q)
            ground_truth_mismatch.append(1)  # Mismatch present
        else:
            assigned_labels.append(true_q)
            ground_truth_mismatch.append(0)  # Correct match

    ground_truth_mismatch = np.array(ground_truth_mismatch)

    # 4. Compute Mismatch Scores
    scores = []
    max_dist = np.sqrt(8.0)

    for i, item in enumerate(val_preds):
        label_q = assigned_labels[i]
        label_centroid = QUADRANT_CENTROIDS[label_q]

        # Euclidean distance normalized to [0, 1]
        dist_euclid = np.sqrt(
            (item["pred_v"] - label_centroid[0]) ** 2 +
            (item["pred_a"] - label_centroid[1]) ** 2
        )
        d_norm = float(dist_euclid / max_dist)

        # Calibrated probability disagreement
        prob_target = float(item["quad_probs"].get(label_q, 0.0))
        d_prob = float(1.0 - prob_target)

        # Combined score
        s_mismatch = 0.5 * d_norm + 0.5 * d_prob
        scores.append(s_mismatch)

    scores = np.array(scores)

    # 5. Sweep Thresholds to Maximize F1
    threshold_candidates = np.arange(0.10, 0.90, 0.01)
    f1_curve = []
    best_f1 = -1.0
    best_tau = 0.50
    best_prec = 0.0
    best_rec = 0.0

    for tau in threshold_candidates:
        tau = round(float(tau), 4)
        pred_mismatch = (scores >= tau).astype(int)

        tp = np.sum((pred_mismatch == 1) & (ground_truth_mismatch == 1))
        fp = np.sum((pred_mismatch == 1) & (ground_truth_mismatch == 0))
        fn = np.sum((pred_mismatch == 0) & (ground_truth_mismatch == 1))

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        f1_curve.append({
            "threshold": tau,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4)
        })

        if f1 > best_f1:
            best_f1 = f1
            best_tau = tau
            best_prec = prec
            best_rec = rec

    print(f"\nOptimal Validation Threshold: tau = {best_tau:.4f}")
    print(f"  Best F1-Score : {best_f1:.4f}")
    print(f"  Precision     : {best_prec:.4f}")
    print(f"  Recall        : {best_rec:.4f}")

    # 6. Save results to backend/calibration/threshold.json
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    calibration_data = {
        "threshold": best_tau,
        "best_f1": round(best_f1, 4),
        "precision_at_threshold": round(best_prec, 4),
        "recall_at_threshold": round(best_rec, 4),
        "sample_size": n_samples,
        "swapped_count": n_swap,
        "swap_rate": swap_rate,
        "seed": seed,
        "split_used": "validation",
        "proxy_disclaimer": "Synthetic mismatches are an empirical proxy for decision boundary calibration, not real ground truth.",
        "f1_curve": f1_curve
    }

    with open(output_file, "w") as f:
        json.dump(calibration_data, f, indent=2)

    print(f"Saved calibration results to {output_file} successfully!")
    return calibration_data

if __name__ == "__main__":
    calibrate_mismatch_threshold()
