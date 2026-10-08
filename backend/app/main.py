"""
FastAPI Server for Label Audit.
Endpoints:
- POST /api/analyze (Audio mood classification, lyrics analysis, mismatch comparator, verification agent)
- GET /api/metrics (Serves reports_train/metrics.json unchanged)
- GET /api/health (System and model loading status)
- Static files served at /

Privacy & Security:
- Rate limit per IP
- Uploads saved to temp dir and deleted in finally block
- Logs only request_id, timing, and error codes (never audio, titles, artists, or lyrics)
- Validated with Pydantic
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import uuid
import json
import logging
import tempfile
from typing import Optional, List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.app.inference.audio import AudioPredictor, TARGET_SR, MAX_FRAMES
from backend.app.inference.lyrics import LyricsPredictor
from backend.app.inference.label import LabelParser
from backend.app.inference.comparator import MismatchComparator
from backend.app.agent.engine import VerificationAgent
from backend.app.agent.tools import fetch_lyrics, fetch_song_audio
import urllib.parse
import urllib.request

# Setup privacy-compliant logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("label_audit")

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
MAX_ANALYSIS_DURATION_SECONDS = 180.0    # 3 minutes cap
RATE_LIMIT_PER_MINUTE = 60

# Magic bytes signature checker for audio formats
AUDIO_SIGNATURES = {
    b"RIFF": "wav",
    b"ID3": "mp3",
    b"\xff\xfb": "mp3",
    b"\xff\xf3": "mp3",
    b"\xff\xf2": "mp3",
    b"OggS": "ogg",
    b"fLaC": "flac",
    b"ftyp": "m4a"  # MP4/M4A container usually contains 'ftyp' in first 12 bytes
}

# --- Pydantic Schemas ---
class AudioResultSchema(BaseModel):
    valence: float
    arousal: float
    raw_valence: float
    raw_arousal: float
    quadrant_probs: Dict[str, float]
    primary_quadrant: str
    top_tags: List[str]

class LabelResultSchema(BaseModel):
    raw_label: Optional[str] = None
    quadrant: str
    is_known: bool
    valence: Optional[float] = None
    arousal: Optional[float] = None
    note: Optional[str] = None

class MismatchResultSchema(BaseModel):
    score: Optional[float] = None
    threshold: float
    flagged: bool
    reason: Optional[str] = None
    distance_audio_label: Optional[float] = None
    distance_audio_lyrics: Optional[float] = None
    disclaimer: str

class LyricsResultSchema(BaseModel):
    status: str
    source: Optional[str] = None
    valence: Optional[float] = None
    arousal: Optional[float] = None
    quadrant_probs: Optional[Dict[str, float]] = None
    primary_quadrant: Optional[str] = None
    note: Optional[str] = None

class TraceItemSchema(BaseModel):
    step: int
    tool: str
    input_summary: str
    output_summary: str
    duration_ms: float

class AgentResultSchema(BaseModel):
    verdict: str
    confidence: float
    explanation: str
    trace: List[TraceItemSchema]
    lyrics_source: Optional[str] = None

class MelDisplaySchema(BaseModel):
    n_mels: int
    n_frames: int
    db_min: float
    db_max: float
    downsample_factor: int
    uint8_data: str

class AnalyzeResponse(BaseModel):
    request_id: str
    duration_s: float
    duration_analyzed_s: float
    artwork_url: Optional[str] = None
    track_title: Optional[str] = None
    track_artist: Optional[str] = None
    audio: AudioResultSchema
    label: Optional[LabelResultSchema] = None
    mismatch: Optional[MismatchResultSchema] = None
    lyrics: Optional[LyricsResultSchema] = None
    agent: Optional[AgentResultSchema] = None
    mel: MelDisplaySchema

# In-memory IP rate limiter
class IPRateLimiter:
    def __init__(self, limit=RATE_LIMIT_PER_MINUTE, window_s=60):
        self.limit = limit
        self.window_s = window_s
        self.requests = {}

    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        timestamps = self.requests.get(client_ip, [])
        # Expire old timestamps
        valid_timestamps = [t for t in timestamps if now - t < self.window_s]
        if len(valid_timestamps) >= self.limit:
            self.requests[client_ip] = valid_timestamps
            return False
        valid_timestamps.append(now)
        self.requests[client_ip] = valid_timestamps
        return True

rate_limiter = IPRateLimiter()

def create_app(
    onnx_path="weights/audio_mood.onnx",
    lyrics_model_dir="weights/lyrics_model",
    temperature_path="weights/temperature.json",
    mood_map_path="configs/mood_map.yaml",
    threshold_path="backend/calibration/threshold.json",
    metrics_path="reports_train/metrics.json"
):
    app = FastAPI(
        title="Label Audit API",
        description="Audio & Lyrics Mood Intelligence and Editorial Mismatch Engine",
        version="1.0.0"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize models
    app.state.models_loaded = False
    app.state.metrics_path = metrics_path if os.path.exists(metrics_path) else "reports/metrics.json"

    try:
        app.state.audio_predictor = AudioPredictor(
            onnx_path=onnx_path,
            temperature_path=temperature_path,
            mood_map_path=mood_map_path
        )
        app.state.lyrics_predictor = LyricsPredictor(model_dir=lyrics_model_dir)
        app.state.label_parser = LabelParser(mood_map_path=mood_map_path)
        app.state.comparator = MismatchComparator(threshold_path=threshold_path)
        app.state.agent = VerificationAgent(
            app.state.audio_predictor,
            app.state.lyrics_predictor,
            app.state.comparator
        )
        app.state.models_loaded = True
        logger.info("All machine learning inference models loaded successfully.")
    except Exception as exc:
        logger.warning(f"Model initialization note: {exc}")
        app.state.models_loaded = False

    # Check for placeholder warnings in privacy and terms
    placeholders = ["[ENTITY NAME]", "[CONTACT EMAIL]", "[GOVERNING LAW]", "[EFFECTIVE DATE]"]
    found_placeholders = []
    for doc in ["backend/static/privacy.html", "backend/static/terms.html"]:
        if os.path.exists(doc):
            with open(doc, encoding="utf-8") as f:
                content = f.read()
                for ph in placeholders:
                    if ph in content and ph not in found_placeholders:
                        found_placeholders.append(ph)

    if found_placeholders:
        logger.warning(
            f"STARTUP WARNING: Legal document placeholders remain unpopulated: {', '.join(found_placeholders)}. "
            f"Replace before public production release."
        )

    def is_valid_audio_content(file_bytes: bytes) -> bool:
        """Inspects magic byte headers to verify legitimate audio format."""
        if len(file_bytes) < 12:
            return False
        # Direct prefix matches
        for sig in [b"RIFF", b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2", b"OggS", b"fLaC"]:
            if file_bytes.startswith(sig):
                return True
        # Container search for ftyp (m4a/mp4 audio)
        if b"ftyp" in file_bytes[:16]:
            return True
        return False

    @app.get("/api/health")
    def health():
        if app.state.models_loaded:
            return {"status": "ok", "model_loaded": True}
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "degraded", "model_loaded": False, "error": "model_not_loaded"}
        )

    @app.get("/api/metrics")
    def get_metrics():
        if not os.path.exists(app.state.metrics_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Metrics report not found."
            )
        with open(app.state.metrics_path) as f:
            data = json.load(f)
        return data

    @app.post("/api/analyze", response_model=AnalyzeResponse)
    async def analyze_audio(
        request: Request,
        file: UploadFile = File(...),
        label: Optional[str] = Form(None),
        title: Optional[str] = Form(None),
        artist: Optional[str] = Form(None),
        lyrics: Optional[str] = Form(None)
    ):
        t_start = time.perf_counter()
        req_id = str(uuid.uuid4())
        client_ip = request.client.host if request.client else "unknown"

        # 1. Rate Limiting Check
        if not rate_limiter.is_allowed(client_ip):
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            logger.warning(f"Request {req_id} [IP {client_ip}] rate limited in {elapsed_ms:.1f}ms - 429")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Maximum 60 requests per minute allowed."
            )

        # 2. Model Loading Check
        if not app.state.models_loaded:
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            logger.error(f"Request {req_id} failed: model_not_loaded in {elapsed_ms:.1f}ms - 503")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"error": "model_not_loaded", "message": "The analysis model is not connected to this deployment yet."}
            )

        # 3. Read Header and Size Check
        header_chunk = await file.read(512)
        if len(header_chunk) == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

        # Check content signature (415)
        if not is_valid_audio_content(header_chunk):
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            logger.warning(f"Request {req_id} rejected unsupported media type in {elapsed_ms:.1f}ms - 415")
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Invalid audio content. Supported formats: MP3, WAV, OGG, FLAC, M4A."
            )

        # 4. Save to temporary file with size enforcement
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, f"label_audit_{req_id}.audio")
        total_size = len(header_chunk)

        try:
            with open(temp_file_path, "wb") as f_out:
                f_out.write(header_chunk)
                while True:
                    chunk = await file.read(1024 * 1024) # 1 MB chunks
                    if not chunk:
                        break
                    total_size += len(chunk)
                    if total_size > MAX_FILE_SIZE_BYTES:
                        raise HTTPException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=f"Audio file exceeds size limit of {MAX_FILE_SIZE_BYTES // (1024*1024)} MB."
                        )
                    f_out.write(chunk)

            return execute_analysis_pipeline(
                audio_file_path=temp_file_path,
                label=label,
                title=title,
                artist=artist,
                lyrics=lyrics,
                req_id=req_id,
                t_start=t_start
            )

        finally:
            # Enforce immediate file deletion in finally block
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception as del_err:
                    logger.error(f"Failed to delete temp file {temp_file_path}: {del_err}")

    def execute_analysis_pipeline(
        audio_file_path: str,
        label: Optional[str],
        title: Optional[str],
        artist: Optional[str],
        lyrics: Optional[str],
        req_id: str,
        t_start: float,
        artwork_url: Optional[str] = None
    ) -> AnalyzeResponse:
        # Audio Decoding & Capped Duration Handling
        y, sr = app.state.audio_predictor.load_audio(audio_file_path)
        full_duration_s = float(len(y) / sr)

        # Cap duration for analysis
        if full_duration_s > MAX_ANALYSIS_DURATION_SECONDS:
            analyzed_samples = int(MAX_ANALYSIS_DURATION_SECONDS * sr)
            y_analyzed = y[:analyzed_samples]
            duration_analyzed_s = MAX_ANALYSIS_DURATION_SECONDS
        else:
            y_analyzed = y
            duration_analyzed_s = full_duration_s

        # Audio Inference
        norm_mel, raw_log_mel = app.state.audio_predictor.compute_log_mel(y_analyzed, sr)
        audio_pred = app.state.audio_predictor.predict_mel_array(norm_mel)
        mel_display = app.state.audio_predictor.prepare_display_mel(raw_log_mel, downsample_factor=4)

        # Label Parsing
        label_info = app.state.label_parser.parse_label(label)

        # Lyrics Analysis
        lyrics_text = lyrics
        lyrics_source = "user_provided" if lyrics else None

        if not lyrics_text and title:
            # Query LRCLIB or Genius (title & artist only)
            fetch_res = fetch_lyrics(title, artist)
            if fetch_res.get("status") == "success":
                lyrics_text = fetch_res.get("lyrics")
                lyrics_source = fetch_res.get("source")

        lyrics_result = None
        if lyrics_text and str(lyrics_text).strip():
            lyr_pred = app.state.lyrics_predictor.predict_lyrics(lyrics_text)
            lyrics_result = {
                "status": lyr_pred["status"],
                "source": lyrics_source,
                "valence": lyr_pred.get("valence"),
                "arousal": lyr_pred.get("arousal"),
                "quadrant_probs": lyr_pred.get("quadrant_probs"),
                "primary_quadrant": lyr_pred.get("primary_quadrant"),
                "note": lyr_pred.get("note")
            }
        else:
            lyrics_result = {
                "status": "not_found",
                "source": None,
                "valence": None,
                "arousal": None,
                "quadrant_probs": None,
                "primary_quadrant": None,
                "note": "No lyrics provided or found online."
            }

        # Mismatch Comparator
        mismatch_eval = app.state.comparator.compare(
            audio_result=audio_pred,
            label_result=label_info,
            lyrics_result=lyrics_result
        )

        # Verification Agent (runs only when mismatch is flagged)
        agent_result = None
        if mismatch_eval["flagged"]:
            agent_res = app.state.agent.run(
                audio_path=audio_file_path,
                norm_mel=norm_mel,
                label_info=label_info,
                title=title,
                artist=artist,
                provided_lyrics=lyrics_text
            )
            agent_result = {
                "verdict": agent_res["verdict"],
                "confidence": agent_res["confidence"],
                "explanation": agent_res["explanation"],
                "trace": agent_res["trace"],
                "lyrics_source": agent_res.get("lyrics_source")
            }

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        logger.info(f"Request {req_id} completed successfully in {elapsed_ms:.1f}ms - 200 OK")

        return AnalyzeResponse(
            request_id=req_id,
            duration_s=round(full_duration_s, 2),
            duration_analyzed_s=round(duration_analyzed_s, 2),
            artwork_url=artwork_url,
            track_title=title,
            track_artist=artist,
            audio=AudioResultSchema(
                valence=round(audio_pred["valence"], 4),
                arousal=round(audio_pred["arousal"], 4),
                raw_valence=round(audio_pred["raw_valence"], 4),
                raw_arousal=round(audio_pred["raw_arousal"], 4),
                quadrant_probs={k: round(v, 4) for k, v in audio_pred["quadrant_probs"].items()},
                primary_quadrant=audio_pred["primary_quadrant"],
                top_tags=audio_pred["top_tags"]
            ),
            label=LabelResultSchema(
                raw_label=label_info.get("raw_label"),
                quadrant=label_info.get("quadrant", "unknown"),
                is_known=label_info.get("is_known", False),
                valence=label_info.get("valence"),
                arousal=label_info.get("arousal"),
                note=label_info.get("note")
            ),
            mismatch=MismatchResultSchema(
                score=mismatch_eval.get("score"),
                threshold=mismatch_eval["threshold"],
                flagged=mismatch_eval["flagged"],
                reason=mismatch_eval.get("reason"),
                distance_audio_label=mismatch_eval.get("distance_audio_label"),
                distance_audio_lyrics=mismatch_eval.get("distance_audio_lyrics"),
                disclaimer=mismatch_eval["disclaimer"]
            ),
            lyrics=LyricsResultSchema(**lyrics_result),
            agent=AgentResultSchema(**agent_result) if agent_result else None,
            mel=MelDisplaySchema(**mel_display)
        )

    @app.post("/api/fetch_and_analyze", response_model=AnalyzeResponse)
    async def fetch_and_analyze(
        request: Request,
        title: str = Form(...),
        artist: Optional[str] = Form(None),
        label: Optional[str] = Form(None),
        lyrics: Optional[str] = Form(None)
    ):
        t_start = time.perf_counter()
        req_id = str(uuid.uuid4())
        client_ip = request.client.host if request.client else "unknown"

        if not rate_limiter.is_allowed(client_ip):
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded.")

        if not app.state.models_loaded:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail={"error": "model_not_loaded"})

        # Fetch audio via iTunes API
        song_fetch = fetch_song_audio(title, artist)
        if song_fetch["status"] != "success":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=song_fetch.get("error", f"Could not find or fetch song '{title}'.")
            )

        wav_path = song_fetch["wav_path"]
        resolved_title = song_fetch.get("track_name") or title
        resolved_artist = song_fetch.get("artist_name") or artist
        artwork_url = song_fetch.get("artwork_url")

        try:
            return execute_analysis_pipeline(
                audio_file_path=wav_path,
                label=label,
                title=resolved_title,
                artist=resolved_artist,
                lyrics=lyrics,
                req_id=req_id,
                t_start=t_start,
                artwork_url=artwork_url
            )
        finally:
            if os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except Exception as del_err:
                    logger.error(f"Failed to delete temp wav {wav_path}: {del_err}")

    @app.get("/api/search_track")
    def search_track(title: str, artist: Optional[str] = None):
        """Allows users to quickly search and preview tracks."""
        query = f"{title} {artist}".strip() if artist else str(title).strip()
        encoded_q = urllib.parse.quote(query)
        url = f"https://itunes.apple.com/search?term={encoded_q}&entity=song&limit=5"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                tracks = []
                for t in data.get("results", []):
                    tracks.append({
                        "track_name": t.get("trackName"),
                        "artist_name": t.get("artistName"),
                        "artwork_url": t.get("artworkUrl100"),
                        "preview_url": t.get("previewUrl")
                    })
                return {"results": tracks}
        except Exception as exc:
            return {"results": [], "error": str(exc)}

    # Mount static files if present
    static_dir = Path("backend/static")
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app

# Default instance
app = create_app()
