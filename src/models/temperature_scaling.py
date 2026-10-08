"""
Temperature Scaling for Probability Calibration.
Fits temperature parameter T on validation logits to minimize Negative Log Likelihood,
computes Expected Calibration Error (ECE), and saves weights/temperature.json.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.eval.metrics import compute_ece

class ModelWithTemperature(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, val_logits, val_labels, max_iter=50, lr=0.01):
        """
        Fits temperature parameter T using L-BFGS on validation logits and labels.
        """
        device = val_logits.device
        self.to(device)

        nll_criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.forward(val_logits)
            loss = nll_criterion(scaled_logits, val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        # Ensure temperature stays positive
        with torch.no_grad():
            self.temperature.clamp_(min=0.01, max=10.0)

        t_val = float(self.temperature.item())
        return t_val

def calibrate_and_save_temperature(val_logits, val_labels, output_path="weights/temperature.json"):
    val_logits_t = torch.tensor(val_logits, dtype=torch.float32)
    val_labels_t = torch.tensor(val_labels, dtype=torch.long)

    # Initial probabilities and ECE
    orig_probs = torch.softmax(val_logits_t, dim=1).numpy()
    ece_before = compute_ece(orig_probs, val_labels)

    calibrator = ModelWithTemperature()
    best_temp = calibrator.fit(val_logits_t, val_labels_t)

    # Calibrated probabilities and ECE
    with torch.no_grad():
        calibrated_logits = calibrator(val_logits_t)
        cal_probs = torch.softmax(calibrated_logits, dim=1).numpy()

    ece_after = compute_ece(cal_probs, val_labels)

    calib_data = {
        "temperature": best_temp,
        "ece_before": ece_before,
        "ece_after": ece_after
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(calib_data, f, indent=2)

    print(f"\n--- Temperature Scaling Calibration ---")
    print(f"Optimal Temperature T: {best_temp:.4f}")
    print(f"Validation ECE: {ece_before:.4f} -> {ece_after:.4f} (reduction: {max(0, ece_before - ece_after):.4f})")
    print(f"Saved to {output_path}")

    return calib_data
