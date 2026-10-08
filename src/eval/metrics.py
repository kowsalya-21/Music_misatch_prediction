"""
Evaluation Metrics Module for Label Audit.
Implements Lin's Concordance Correlation Coefficient (CCC), RMSE, R2,
Classification Accuracy, Macro-F1, PR-AUC, ROC-AUC, and Expected Calibration Error (ECE).
"""

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score, roc_auc_score, average_precision_score

def compute_ccc(y_true, y_pred):
    """
    Lin's Concordance Correlation Coefficient (CCC).
    Measures agreement between two continuous variables. Range: [-1, 1].
    """
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()

    if len(y_true) < 2:
        return 0.0

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    covar = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

    if denominator <= 1e-8:
        return 0.0

    ccc = (2.0 * covar) / denominator
    return float(np.clip(ccc, -1.0, 1.0))

def compute_regression_metrics(y_true, y_pred):
    """
    Computes R2, RMSE, and CCC for continuous predictions.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    ccc = compute_ccc(y_true, y_pred)

    return {
        "r2": r2,
        "rmse": rmse,
        "ccc": ccc
    }

def compute_classification_metrics(y_true, y_pred, y_probs=None):
    """
    Computes Accuracy, Macro-F1, and multi-class ROC-AUC/PR-AUC if probabilities provided.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    res = {
        "accuracy": acc,
        "macro_f1": f1
    }

    if y_probs is not None and y_probs.shape[1] > 1:
        try:
            res["roc_auc_ovr"] = float(roc_auc_score(y_true, y_probs, multi_class="ovr"))
        except Exception:
            res["roc_auc_ovr"] = None

    return res

def compute_multilabel_metrics(y_true, y_probs):
    """
    Computes macro PR-AUC and ROC-AUC for multi-label tag prediction.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=np.float64)

    # Filter out columns with only one class present
    valid_cols = [c for c in range(y_true.shape[1]) if len(np.unique(y_true[:, c])) > 1]
    if not valid_cols:
        return {"pr_auc_macro": 0.0, "roc_auc_macro": 0.0}

    y_true_sub = y_true[:, valid_cols]
    y_probs_sub = y_probs[:, valid_cols]

    pr_aucs = []
    roc_aucs = []

    for c in range(y_true_sub.shape[1]):
        pr_aucs.append(average_precision_score(y_true_sub[:, c], y_probs_sub[:, c]))
        try:
            roc_aucs.append(roc_auc_score(y_true_sub[:, c], y_probs_sub[:, c]))
        except Exception:
            pass

    return {
        "pr_auc_macro": float(np.mean(pr_aucs)) if pr_aucs else 0.0,
        "roc_auc_macro": float(np.mean(roc_aucs)) if roc_aucs else 0.0
    }

def compute_ece(probs, labels, n_bins=10):
    """
    Expected Calibration Error (ECE) for multi-class classification.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)
