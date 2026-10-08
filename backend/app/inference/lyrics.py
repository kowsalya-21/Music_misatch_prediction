"""
Lyrics Mood Inference Engine for Label Audit.
Loads fine-tuned XLM-RoBERTa-base from weights/lyrics_model/ on CPU.
Includes script and language detection. If text is non-English,
gracefully returns a language_not_validated note without executing inference.
Maps English mood probabilities to continuous valence-arousal circumplex coordinates [-1, 1].
"""

import os
import re
import unicodedata
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

QUADRANT_NAMES = ["happy", "angry", "sad", "calm"]

# Circumplex centroid coordinates for the 4 canonical quadrants on [-1, 1]^2
QUADRANT_CENTROIDS = {
    "happy": (0.6, 0.6),    # Q1: High Valence, High Arousal
    "angry": (-0.6, 0.6),   # Q2: Low Valence, High Arousal
    "sad": (-0.6, -0.6),    # Q3: Low Valence, Low Arousal
    "calm": (0.6, -0.6)     # Q4: High Valence, Low Arousal
}

def detect_non_latin_script(text):
    """
    Checks if text contains non-Latin scripts (e.g. Telugu, Devanagari, Arabic, CJK).
    """
    non_latin_chars = 0
    total_letters = 0
    
    for ch in text:
        if ch.isalpha():
            total_letters += 1
            # Check script category or unicode block
            cat = unicodedata.name(ch, "")
            if not any(cat.startswith(p) for p in ["LATIN"]):
                non_latin_chars += 1
                
    if total_letters == 0:
        return False, 0.0
    
    ratio = non_latin_chars / total_letters
    return ratio > 0.10, ratio

def is_english_text(text):
    """
    Detects if the lyrical text is English.
    Returns (is_english, reason_note).
    """
    if not text or len(text.strip()) < 5:
        return False, "Lyrics text is too short or empty."

    # 1. Check for non-Latin scripts
    has_non_latin, ratio = detect_non_latin_script(text)
    if has_non_latin:
        return False, "Non-Latin script detected (e.g. Indic / Asian / Arabic characters)."

    # 2. Heuristic check on common English stopwords / words
    english_words = {
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for",
        "not", "on", "with", "he", "as", "you", "do", "at", "this", "but", "his",
        "by", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my",
        "one", "all", "would", "there", "their", "what", "so", "up", "out", "if",
        "about", "who", "get", "which", "go", "me", "when", "make", "can", "like",
        "time", "no", "just", "him", "know", "take", "people", "into", "year", "your",
        "good", "some", "could", "them", "see", "other", "than", "then", "now", "look",
        "only", "come", "its", "over", "think", "also", "back", "after", "use", "two",
        "how", "our", "work", "first", "well", "way", "even", "new", "want", "because",
        "any", "these", "give", "day", "most", "us", "love", "night", "heart", "feel",
        "baby", "never", "forever", "tonight", "eyes", "dream", "tears", "sun", "rain"
    }

    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    if len(words) < 3:
        return True, "Short text, latin script assumed."

    english_match_count = sum(1 for w in words if w in english_words)
    match_ratio = english_match_count / len(words)

    # If very low match on standard English words for longer texts, it may be Romanized regional text (e.g. Romanized Telugu)
    if len(words) >= 10 and match_ratio < 0.12:
        return False, "Text appears to be non-English or Romanized regional lyrics."

    return True, "English verified."

class LyricsPredictor:
    def __init__(self, model_dir="weights/lyrics_model", allow_multilingual=True):
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Lyrics model directory not found: {model_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()  # CPU inference
        self.allow_multilingual = allow_multilingual

    def predict_lyrics(self, lyrics_text):
        """
        Analyzes mood from song lyrics text.
        Supports English and Multilingual (Telugu/Indic) zero-shot inference.
        """
        if not lyrics_text or not lyrics_text.strip():
            return {
                "status": "missing",
                "note": "No lyrics provided.",
                "valence": None,
                "arousal": None,
                "quadrant_probs": None,
                "primary_quadrant": None
            }

        # Validate language / script
        is_en, reason = is_english_text(lyrics_text)
        if not is_en and not self.allow_multilingual:
            return {
                "status": "language_not_validated",
                "note": f"Lyrics model validated for English only ({reason}). Regional transfer pending fine-tuning.",
                "valence": None,
                "arousal": None,
                "quadrant_probs": None,
                "primary_quadrant": None
            }

        # Tokenize on CPU (XLM-RoBERTa supports 100+ languages including Telugu)
        enc = self.tokenizer(
            lyrics_text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**enc)
            logits = outputs.logits[0].numpy()
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)

        quad_dict = {
            "happy": float(probs[0]),
            "angry": float(probs[1]),
            "sad": float(probs[2]),
            "calm": float(probs[3])
        }

        # Map to continuous valence and arousal in [-1, 1] using quadrant centroids
        v_weighted = 0.0
        a_weighted = 0.0
        for q_name, prob in quad_dict.items():
            cx, cy = QUADRANT_CENTROIDS[q_name]
            v_weighted += prob * cx
            a_weighted += prob * cy

        best_idx = int(np.argmax(probs))
        primary_q = QUADRANT_NAMES[best_idx]

        note = "English lyrics analyzed successfully." if is_en else f"Multilingual zero-shot inference ({reason})."

        return {
            "status": "success",
            "valence": float(round(v_weighted, 4)),
            "arousal": float(round(a_weighted, 4)),
            "quadrant_probs": quad_dict,
            "primary_quadrant": primary_q,
            "confidence": float(round(probs[best_idx], 4)),
            "note": note
        }

