import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys

from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)
audio_file = "C:/Users/manin/Downloads/Power House.mp3"

print("=================================================================")
print("RUNNING END-TO-END TELUGU PIPELINE AUDIT")
print("=================================================================")

# TEST 1: Matching Telugu Song & Label (Upbeat Audio + Telugu 'ఆనందం' label + Telugu lyrics)
with open(audio_file, "rb") as f:
    r1 = client.post(
        "/api/analyze",
        files={"file": ("power_house.mp3", f, "audio/mpeg")},
        data={
            "label": "ఆనందం",
            "title": "Power House",
            "artist": "Telugu Singer",
            "lyrics": "నాటు నాటు నాటు పచ్చి మిరపకాయలాంటి నాటు పాట"
        }
    )

res1 = r1.json()
print("\n[TEST 1: Upbeat Audio + Telugu 'ఆనందం' Label + Telugu Lyrics]")
print(f"  Audio primary quadrant : {res1['audio']['primary_quadrant']} (V={res1['audio']['valence']:.4f}, A={res1['audio']['arousal']:.4f})")
print(f"  Label quadrant         : {res1['label']['quadrant']} (is_known: {res1['label']['is_known']})")
print(f"  Lyrics status          : {res1['lyrics']['status']} (quadrant: {res1['lyrics']['primary_quadrant']})")
print(f"  Lyrics coordinates     : V={res1['lyrics']['valence']:.4f}, A={res1['lyrics']['arousal']:.4f}")
print(f"  Lyrics note            : {res1['lyrics']['note']}")
print(f"  Mismatch score         : {res1['mismatch']['score']:.4f} (Threshold: {res1['mismatch']['threshold']:.4f})")
print(f"  Mismatch flagged       : {res1['mismatch']['flagged']}")
print(f"  Distance Audio-Label   : {res1['mismatch']['distance_audio_label']:.4f}")
print(f"  Distance Audio-Lyrics  : {res1['mismatch']['distance_audio_lyrics']:.4f}")
print(f"  Verdict / Reason       : {res1['mismatch']['reason']}")

# TEST 2: Mismatched Telugu Label (Upbeat Audio + Telugu 'బాధ' (Sad) Label + Sad Lyrics)
with open(audio_file, "rb") as f:
    r2 = client.post(
        "/api/analyze",
        files={"file": ("power_house.mp3", f, "audio/mpeg")},
        data={
            "label": "బాధ",
            "title": "Power House",
            "artist": "Telugu Singer",
            "lyrics": "కన్నుల దాచిన కన్నీరై పోయావా ఎందుకీ వేదన బ్రతుకు భారమై"
        }
    )

res2 = r2.json()
print("\n[TEST 2: Upbeat Audio + Telugu 'బాధ' (Sad) Label + Sad Telugu Lyrics]")
print(f"  Audio primary quadrant : {res2['audio']['primary_quadrant']}")
print(f"  Label quadrant         : {res2['label']['quadrant']} (is_known: {res2['label']['is_known']})")
print(f"  Lyrics status          : {res2['lyrics']['status']} (quadrant: {res2['lyrics']['primary_quadrant']})")
print(f"  Lyrics Sad prob        : {res2['lyrics']['quadrant_probs']['sad']:.4f}")
print(f"  Mismatch score         : {res2['mismatch']['score']:.4f} (Threshold: {res2['mismatch']['threshold']:.4f})")
print(f"  Mismatch flagged       : {res2['mismatch']['flagged']}")
print(f"  Distance Audio-Label   : {res2['mismatch']['distance_audio_label']:.4f}")
print(f"  Distance Audio-Lyrics  : {res2['mismatch']['distance_audio_lyrics']:.4f}")
print(f"  Verdict / Reason       : {res2['mismatch']['reason']}")
if res2.get("agent"):
    print(f"  Verification Agent     : {res2['agent']['verdict']} (Confidence: {res2['agent']['confidence']:.1%})")
    print(f"  Agent Explanation      : {res2['agent']['explanation']}")

print("\n" + "=" * 65)
print("AUDIT SUCCESS: All missing fields are now populated!")
print("=" * 65)
