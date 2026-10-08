"""
Modular Tools for Label Audit Verification Agent.
Tools:
1. get_audio_features
2. predict_audio_mood
3. fetch_lyrics (LRCLIB primary, Genius fallback, short timeouts, sends only title/artist)
4. analyze_lyrics_mood
5. compare_signals
"""

import os
import re
import time
import urllib.parse
import urllib.request
import json
import numpy as np

TIMEOUT_SECONDS = 3.0

def get_audio_features(audio_path):
    """Tool 1: Inspects audio file properties and duration."""
    t0 = time.perf_counter()
    if not os.path.exists(audio_path):
        return {
            "status": "error",
            "error": f"Audio file not found: {audio_path}",
            "duration_ms": (time.perf_counter() - t0) * 1000.0
        }
    file_size_kb = os.path.getsize(audio_path) / 1024.0
    return {
        "status": "success",
        "file_path": audio_path,
        "file_size_kb": round(file_size_kb, 1),
        "duration_ms": (time.perf_counter() - t0) * 1000.0
    }

def predict_audio_mood(audio_predictor, norm_mel):
    """Tool 2: Predicts continuous VA coordinates and quadrant probabilities from mel array."""
    t0 = time.perf_counter()
    res = audio_predictor.predict_mel_array(norm_mel)
    res["duration_ms"] = (time.perf_counter() - t0) * 1000.0
    return res

def fetch_lyrics(title, artist):
    """
    Tool 3: Fetches song lyrics from public APIs.
    Queries LRCLIB first, then Genius fallback if GENIUS_API_KEY is present.
    Sends only title and artist. Never sends audio. Never invents lyrics.
    """
    t0 = time.perf_counter()
    if not title or not str(title).strip():
        return {
            "status": "not_found",
            "source": None,
            "lyrics": None,
            "note": "No song title provided for lyrics lookup.",
            "duration_ms": (time.perf_counter() - t0) * 1000.0
        }

    clean_title = str(title).strip()
    clean_artist = str(artist).strip() if artist else ""

    # Generate candidate title and artist queries (stripping film suffixes like "(From ...)")
    candidates = [(clean_title, clean_artist)]
    simplified_title = re.sub(r'[\(\[\{].*?[\)\]\}]', '', clean_title).strip()
    if simplified_title and simplified_title != clean_title:
        candidates.append((simplified_title, clean_artist))
        if clean_artist:
            # Try individual artists if multiple are listed (e.g. S.S. Thaman & Sid Sriram -> Sid Sriram)
            for part in re.split(r'[,&/]', clean_artist):
                cand_a = part.strip()
                if cand_a:
                    candidates.append((simplified_title, cand_a))
        candidates.append((simplified_title, ""))

    # 1. Primary Service: LRCLIB API
    for q_title, q_artist in candidates:
        try:
            query_params = {"track_name": q_title}
            if q_artist:
                query_params["artist_name"] = q_artist
            
            url = f"https://lrclib.net/api/get?{urllib.parse.urlencode(query_params)}"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "LabelAudit/1.0 (academic; music-mismatch-audit)"}
            )

            with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    lyrics_text = data.get("plainLyrics") or data.get("lyrics")
                    if lyrics_text and len(lyrics_text.strip()) > 10:
                        return {
                            "status": "success",
                            "source": "LRCLIB",
                            "lyrics": lyrics_text.strip(),
                            "track_name": data.get("trackName"),
                            "artist_name": data.get("artistName"),
                            "duration_ms": (time.perf_counter() - t0) * 1000.0
                        }
        except Exception as exc:
            pass

    # 2. Fallback Service: Genius API (only if key present in environment)
    genius_token = os.environ.get("GENIUS_API_KEY")
    if genius_token and genius_token.strip():
        try:
            q = f"{clean_title} {clean_artist}".strip()
            url = f"https://api.genius.com/search?{urllib.parse.urlencode({'q': q})}"
            req = urllib.request.Request(
                url,
                headers={"Authorization": f"Bearer {genius_token.strip()}"}
            )
            with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    hits = data.get("response", {}).get("hits", [])
                    if hits:
                        hit_title = hits[0].get("result", {}).get("title")
                        # Genius search API returns metadata/snippets; note that lyrics scraping requires web page parsing
                        return {
                            "status": "partial",
                            "source": "Genius",
                            "lyrics": None,
                            "note": f"Genius hit found: {hit_title}. Full lyrics require direct parsing.",
                            "duration_ms": (time.perf_counter() - t0) * 1000.0
                        }
        except Exception:
            pass

    return {
        "status": "not_found",
        "source": None,
        "lyrics": None,
        "note": f"Lyrics could not be found for '{clean_title}' by '{clean_artist}'.",
        "duration_ms": (time.perf_counter() - t0) * 1000.0
    }

def analyze_lyrics_mood(lyrics_predictor, lyrics_text):
    """Tool 4: Analyzes emotional valence and arousal from lyrics text."""
    t0 = time.perf_counter()
    res = lyrics_predictor.predict_lyrics(lyrics_text)
    res["duration_ms"] = (time.perf_counter() - t0) * 1000.0
    return res

def compare_signals(audio_mood, label_mood, lyrics_mood, comparator):
    """Tool 5: Evaluates mismatch score and spatial divergence among modalities."""
    t0 = time.perf_counter()
    res = comparator.compare(audio_mood, label_mood, lyrics_mood)
    res["duration_ms"] = (time.perf_counter() - t0) * 1000.0
    return res

def fetch_song_audio(title, artist=None):
    """
    Tool 6: Fetches 30-second studio audio preview for title and artist using iTunes Search API.
    Converts to 22,050Hz mono WAV using ffmpeg without requiring any API keys.
    """
    import subprocess
    import tempfile

    t0 = time.perf_counter()
    if not title or not str(title).strip():
        return {
            "status": "error",
            "error": "No title provided for song search.",
            "duration_ms": (time.perf_counter() - t0) * 1000.0
        }

    query = f"{title} {artist}".strip() if artist else str(title).strip()
    encoded_q = urllib.parse.quote(query)
    url = f"https://itunes.apple.com/search?term={encoded_q}&entity=song&limit=1"

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    try:
        with urllib.request.urlopen(req, timeout=6.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            results = data.get("results", [])
            if not results:
                return {
                    "status": "not_found",
                    "error": f"No tracks found matching '{query}'.",
                    "duration_ms": (time.perf_counter() - t0) * 1000.0
                }

            track = results[0]
            preview_url = track.get("previewUrl")
            if not preview_url:
                return {
                    "status": "not_found",
                    "error": "Track found, but audio preview is unavailable.",
                    "duration_ms": (time.perf_counter() - t0) * 1000.0
                }

            # Download audio preview
            with urllib.request.urlopen(preview_url, timeout=10.0) as audio_resp:
                audio_data = audio_resp.read()

            with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as f_in, tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
                f_in.write(audio_data)
                in_path, out_path = f_in.name, f_out.name

            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", in_path, "-ar", "22050", "-ac", "1", out_path],
                    capture_output=True,
                    check=True
                )
            finally:
                if os.path.exists(in_path):
                    os.remove(in_path)

            return {
                "status": "success",
                "track_name": track.get("trackName"),
                "artist_name": track.get("artistName"),
                "artwork_url": track.get("artworkUrl100"),
                "preview_url": preview_url,
                "wav_path": out_path,
                "duration_ms": (time.perf_counter() - t0) * 1000.0
            }
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "duration_ms": (time.perf_counter() - t0) * 1000.0
        }

