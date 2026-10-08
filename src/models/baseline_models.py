"""
Baseline Audio Models for Label Audit.
Trains LightGBM Regressors and Support Vector Regressors (SVR) on 76 Librosa features
to predict Valence and Arousal, and derive 4-quadrant mood classifications.
Saves weights/baseline_lgbm.txt and weights/baseline_svm.pkl.
"""

import os
import json
import pickle
import numpy as np
import lightgbm as lgb
from sklearn.svm import SVR
from src.eval.metrics import compute_regression_metrics, compute_classification_metrics

def va_to_quadrant(v, a):
    """
    Maps continuous valence and arousal (normalized in [0, 1]) to 4 quadrants:
    0: Happy (V >= 0.5, A >= 0.5)
    1: Angry (V < 0.5, A >= 0.5)
    2: Sad   (V < 0.5, A < 0.5)
    3: Calm  (V >= 0.5, A < 0.5)
    """
    quadrants = np.zeros(len(v), dtype=int)
    for i in range(len(v)):
        if v[i] >= 0.5 and a[i] >= 0.5:
            quadrants[i] = 0
        elif v[i] < 0.5 and a[i] >= 0.5:
            quadrants[i] = 1
        elif v[i] < 0.5 and a[i] < 0.5:
            quadrants[i] = 2
        else:
            quadrants[i] = 3
    return quadrants

def train_and_eval_baselines(features_path="data/processed/features_deam_pmemo.npy",
                             meta_path="data/processed/meta_deam_pmemo.json",
                             norm_stats_path="weights/norm_stats.json",
                             output_dir="weights"):
    os.makedirs(output_dir, exist_ok=True)
    with open(meta_path) as f:
        meta = json.load(f)

    with open(norm_stats_path) as f:
        norm_stats = json.load(f)

    feats_all = np.load(features_path)
    mean = np.array(norm_stats["features_mean"], dtype=np.float32)
    std = np.array(norm_stats["features_std"], dtype=np.float32)

    # Standardize features using train statistics only
    feats_norm = (feats_all - mean) / std

    train_idx = [m["index"] for m in meta if m["split"] == "train"]
    val_idx = [m["index"] for m in meta if m["split"] == "val"]
    test_idx = [m["index"] for m in meta if m["split"] == "test"]

    X_train, X_val, X_test = feats_norm[train_idx], feats_norm[val_idx], feats_norm[test_idx]

    y_v_all = np.array([m["valence"] for m in meta], dtype=np.float32)
    y_a_all = np.array([m["arousal"] for m in meta], dtype=np.float32)
    y_q_all = np.array([m["quadrant"] for m in meta], dtype=int)

    y_v_train, y_v_val, y_v_test = y_v_all[train_idx], y_v_all[val_idx], y_v_all[test_idx]
    y_a_train, y_a_val, y_a_test = y_a_all[train_idx], y_a_all[val_idx], y_a_all[test_idx]
    y_q_test = y_q_all[test_idx]

    print(f"Dataset splits: Train {len(train_idx)}, Val {len(val_idx)}, Test {len(test_idx)}")

    # 1. LightGBM Baselines
    print("\n--- Training LightGBM Baseline Regressors ---")
    lgb_v = lgb.LGBMRegressor(
        n_estimators=150, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=4, verbose=-1
    )
    lgb_v.fit(X_train, y_v_train)

    lgb_a = lgb.LGBMRegressor(
        n_estimators=150, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=4, verbose=-1
    )
    lgb_a.fit(X_train, y_a_train)

    pred_v_lgb = lgb_v.predict(X_test)
    pred_a_lgb = lgb_a.predict(X_test)
    pred_q_lgb = va_to_quadrant(pred_v_lgb, pred_a_lgb)

    metrics_v_lgb = compute_regression_metrics(y_v_test, pred_v_lgb)
    metrics_a_lgb = compute_regression_metrics(y_a_test, pred_a_lgb)
    metrics_q_lgb = compute_classification_metrics(y_q_test, pred_q_lgb)

    # Save LightGBM models
    lgb_v.booster_.save_model(os.path.join(output_dir, "baseline_lgbm_valence.txt"))
    lgb_a.booster_.save_model(os.path.join(output_dir, "baseline_lgbm_arousal.txt"))
    # Also write combined baseline_lgbm.txt marker
    with open(os.path.join(output_dir, "baseline_lgbm.txt"), "w") as f:
        f.write("LightGBM Baseline Models: baseline_lgbm_valence.txt, baseline_lgbm_arousal.txt\n")

    # 2. Support Vector Machine (SVR) Baselines
    print("\n--- Training SVM Baseline Regressors ---")
    svm_v = SVR(kernel="rbf", C=1.0, epsilon=0.05)
    svm_v.fit(X_train, y_v_train)

    svm_a = SVR(kernel="rbf", C=1.0, epsilon=0.05)
    svm_a.fit(X_train, y_a_train)

    pred_v_svm = svm_v.predict(X_test)
    pred_a_svm = svm_a.predict(X_test)
    pred_q_svm = va_to_quadrant(pred_v_svm, pred_a_svm)

    metrics_v_svm = compute_regression_metrics(y_v_test, pred_v_svm)
    metrics_a_svm = compute_regression_metrics(y_a_test, pred_a_svm)
    metrics_q_svm = compute_classification_metrics(y_q_test, pred_q_svm)

    with open(os.path.join(output_dir, "baseline_svm.pkl"), "wb") as f:
        pickle.dump({"svm_valence": svm_v, "svm_arousal": svm_a}, f)

    results = {
        "LightGBM": {
            "valence": metrics_v_lgb,
            "arousal": metrics_a_lgb,
            "quadrant": metrics_q_lgb
        },
        "SVM": {
            "valence": metrics_v_svm,
            "arousal": metrics_a_svm,
            "quadrant": metrics_q_svm
        }
    }

    print("\n=== Baseline Results on Held-out Test Split ===")
    print(f"LightGBM -> Valence: R2={metrics_v_lgb['r2']:.4f}, RMSE={metrics_v_lgb['rmse']:.4f}, CCC={metrics_v_lgb['ccc']:.4f}")
    print(f"LightGBM -> Arousal: R2={metrics_a_lgb['r2']:.4f}, RMSE={metrics_a_lgb['rmse']:.4f}, CCC={metrics_a_lgb['ccc']:.4f}")
    print(f"LightGBM -> Quadrant: Acc={metrics_q_lgb['accuracy']:.4f}, Macro-F1={metrics_q_lgb['macro_f1']:.4f}")
    print(f"SVM      -> Valence: R2={metrics_v_svm['r2']:.4f}, RMSE={metrics_v_svm['rmse']:.4f}, CCC={metrics_v_svm['ccc']:.4f}")
    print(f"SVM      -> Arousal: R2={metrics_a_svm['r2']:.4f}, RMSE={metrics_a_svm['rmse']:.4f}, CCC={metrics_a_svm['ccc']:.4f}")
    print(f"SVM      -> Quadrant: Acc={metrics_q_svm['accuracy']:.4f}, Macro-F1={metrics_q_svm['macro_f1']:.4f}")

    return results

if __name__ == "__main__":
    train_and_eval_baselines()
