"""
Mismatch Comparator for Label Audit.
Computes Euclidean distance on Russell's circumplex plane [-1, 1]^2,
probability-weighted quadrant disagreement, and compares against the calibrated threshold
from backend/calibration/threshold.json.
"""

import os
import json
import numpy as np

MAX_EUCLIDEAN_DIST = np.sqrt(8.0) # ~2.8284

class MismatchComparator:
    def __init__(self, threshold_path="backend/calibration/threshold.json"):
        self.threshold = 0.6000 # default fallback
        self.disclaimer = "Synthetic mismatches are an empirical proxy for decision boundary calibration, not real ground truth."

        if os.path.exists(threshold_path):
            with open(threshold_path) as f:
                data = json.load(f)
                self.threshold = float(data.get("threshold", 0.6000))
                self.disclaimer = data.get("proxy_disclaimer", self.disclaimer)

    def compare(self, audio_result, label_result, lyrics_result=None):
        """
        Compares audio predictions with human label and optional lyrics.
        Returns a structured mismatch evaluation dict.
        """
        # If label is not provided or unknown, mismatch is null
        if not label_result or not label_result.get("is_known", False):
            return {
                "score": None,
                "threshold": self.threshold,
                "flagged": False,
                "reason": "Label is missing, ambiguous, or unrecognized. Mismatch score is not applicable.",
                "distance_audio_label": None,
                "distance_audio_lyrics": None,
                "disclaimer": self.disclaimer
            }

        v_audio = audio_result["valence"]
        a_audio = audio_result["arousal"]
        v_label = label_result["valence"]
        a_label = label_result["arousal"]
        label_quad = label_result["quadrant"]

        # Quadrants & Centroid Distance
        audio_quad = audio_result.get("primary_quadrant")
        raw_dist = np.sqrt((v_audio - v_label) ** 2 + (a_audio - a_label) ** 2)

        # 1. Normalized Euclidean Distance [0, 1]
        # Diagonal between opposite centroids (e.g. Happy (0.6, 0.6) to Sad (-0.6, -0.6)) is ~1.697
        CENTROID_DIAGONAL = np.sqrt((1.2) ** 2 + (1.2) ** 2) # ~1.697
        d_norm = min(1.0, float(raw_dist / CENTROID_DIAGONAL))

        # 2. Probability Disagreement [0, 1]
        quad_probs = audio_result.get("quadrant_probs", {})
        prob_target = float(quad_probs.get(label_quad, 0.0))
        d_prob = float(1.0 - prob_target)

        # 3. Combined Score [0, 1]
        score = float(round(0.5 * d_norm + 0.5 * d_prob, 4))

        # 4. Cross-Quadrant Contradiction Check:
        # If the model's primary predicted quadrant directly opposes or differs from the human label,
        # ensure the score properly flags a mismatch so the Verification Agent can investigate.
        OPPOSITE_QUADRANTS = {
            "happy": "sad",
            "sad": "happy",
            "angry": "calm",
            "calm": "angry"
        }
        is_polar_opposite = (OPPOSITE_QUADRANTS.get(audio_quad) == label_quad)
        if is_polar_opposite and score < self.threshold:
            score = float(round(max(score, self.threshold + 0.05), 4))

        flagged = bool(score >= self.threshold or (audio_quad != label_quad and prob_target < 0.25))

        # Distance between audio and lyrics if lyrics available
        dist_audio_lyrics = None
        if lyrics_result and lyrics_result.get("status") == "success" and lyrics_result.get("valence") is not None:
            v_lyr = lyrics_result["valence"]
            a_lyr = lyrics_result["arousal"]
            dist_audio_lyrics = float(round(np.sqrt((v_audio - v_lyr) ** 2 + (a_audio - a_lyr) ** 2), 4))

        if flagged:
            reason = f"Mismatch score {score:.4f} exceeds calibrated threshold {self.threshold:.4f} (Audio predicts '{audio_quad}', label claims '{label_quad}')."
        else:
            reason = f"Mismatch score {score:.4f} is within acceptable threshold {self.threshold:.4f}."

        return {
            "score": score,
            "threshold": self.threshold,
            "flagged": flagged,
            "reason": reason,
            "distance_audio_label": float(round(raw_dist, 4)),
            "distance_audio_lyrics": dist_audio_lyrics,
            "disclaimer": self.disclaimer
        }
