"""
Human Mood Label Parser for Label Audit.
Maps free-text human-assigned mood labels to canonical quadrants
using configs/mood_map.yaml.
If a label is ambiguous or not in the taxonomy, it returns 'unknown'
and guarantees mismatch calculation is null. Never guesses.
"""

import os
import yaml

QUADRANT_CENTROIDS = {
    "happy": (0.6, 0.6),
    "angry": (-0.6, 0.6),
    "sad": (-0.6, -0.6),
    "calm": (0.6, -0.6)
}

class LabelParser:
    def __init__(self, mood_map_path="configs/mood_map.yaml"):
        self.quadrant_map = {}
        self.excluded_tags = set()

        if os.path.exists(mood_map_path):
            with open(mood_map_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)

            # Direct quadrant names
            for q in ["happy", "angry", "sad", "calm"]:
                self.quadrant_map[q] = q

            # Synonyms & aliases
            self.quadrant_map["relaxed"] = "calm"
            self.quadrant_map["peaceful"] = "calm"
            self.quadrant_map["chill"] = "calm"
            self.quadrant_map["joy"] = "happy"
            self.quadrant_map["joyful"] = "happy"
            self.quadrant_map["depressed"] = "sad"
            self.quadrant_map["melancholy"] = "sad"
            self.quadrant_map["melancholic"] = "sad"
            self.quadrant_map["rage"] = "angry"
            self.quadrant_map["dark"] = "angry"
            self.quadrant_map["aggressive"] = "angry"

            # Jamendo Tag Map
            jam_map = cfg.get("jamendo_tag_map", {})
            for tag, quad in jam_map.items():
                self.quadrant_map[tag.lower()] = quad
                clean_tag = tag.replace("mood/theme---", "").lower()
                self.quadrant_map[clean_tag] = quad

            # MoodyLyrics Map
            moody_map = cfg.get("moodylyrics_map", {})
            for tag, quad in moody_map.items():
                self.quadrant_map[tag.lower()] = quad

            # GoEmotions Map
            goemo_map = cfg.get("goemotions_map", {})
            for tag, quad in goemo_map.items():
                if quad is not None:
                    self.quadrant_map[tag.lower()] = quad

            # Telugu Mood Map
            telugu_map = cfg.get("telugu_mood_map", {})
            for tag, quad in telugu_map.items():
                if quad is not None:
                    self.quadrant_map[str(tag).strip().lower()] = quad

            # Excluded Ambiguous Tags
            for tag in cfg.get("excluded_ambiguous_tags", []):
                self.excluded_tags.add(tag.lower())
                clean_tag = tag.replace("mood/theme---", "").lower()
                self.excluded_tags.add(clean_tag)

    def parse_label(self, raw_label):
        """
        Parses free-text human-assigned mood label.
        Returns a dict with quadrant, centroid coordinates, and is_known flag.
        Never guesses unknown or ambiguous labels.
        """
        if raw_label is None or not str(raw_label).strip():
            return {
                "raw_label": None,
                "quadrant": "unknown",
                "is_known": False,
                "valence": None,
                "arousal": None,
                "note": "No human label provided."
            }

        label_clean = str(raw_label).strip().lower()

        # Check if explicitly excluded as ambiguous
        if label_clean in self.excluded_tags or f"mood/theme---{label_clean}" in self.excluded_tags:
            return {
                "raw_label": raw_label,
                "quadrant": "unknown",
                "is_known": False,
                "valence": None,
                "arousal": None,
                "note": f"Label '{raw_label}' is an excluded ambiguous tag in mood taxonomy."
            }

        # Check mapped quadrants
        quadrant = self.quadrant_map.get(label_clean)
        if quadrant is None:
            # Check with prefix
            quadrant = self.quadrant_map.get(f"mood/theme---{label_clean}")

        if quadrant is None or quadrant not in QUADRANT_CENTROIDS:
            return {
                "raw_label": raw_label,
                "quadrant": "unknown",
                "is_known": False,
                "valence": None,
                "arousal": None,
                "note": f"Label '{raw_label}' not recognized in mood taxonomy."
            }

        centroid = QUADRANT_CENTROIDS[quadrant]
        return {
            "raw_label": raw_label,
            "quadrant": quadrant,
            "is_known": True,
            "valence": centroid[0],
            "arousal": centroid[1],
            "note": f"Mapped to canonical {quadrant} quadrant."
        }
