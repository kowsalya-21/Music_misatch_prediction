# Final Backend and Web Interface Engineering Report: Label Audit

**Date**: October 4, 2026  
**System**: Label Audit (Backend Service and Self-Served Web Interface)  
**Host Architecture**: Linux x86_64, NVIDIA DGX B200 / CPU ONNX Runtime  
**Runtime**: Python 3.12.14, FastAPI 0.142.2, ONNX Runtime 1.20.1, Transformers 4.49.0  
**Test Suite Status**: 36 of 36 tests passing (100% pass rate)  
**Audit Gate**: ALL PHASES 0 TO 7 PASSED (No assumed or skipped gates)

---

## 1. Executive Summary

This report documents the end-to-end architecture, empirical performance, and operational verification of the **Label Audit** backend application (`backend/`). The backend directly hosts and serves a lightweight, accessible web user interface from `backend/static/` using pure Vanilla HTML, CSS, and JavaScript with zero external frameworks, zero Node.js dependencies, and zero build steps.

All components load the trained machine learning models from `weights/` and execute with strict scientific hygiene, reproducible calibration, and privacy-compliant data handling.

---

## 2. System Architecture and Implementation

### 2.1 Inference Core (`backend/app/inference/`)
- **Acoustic Mood Inference (`audio.py`)**:
  - Decodes audio to mono 22,050 Hz.
  - Computes 128-bin log-mel spectrograms ($N_{\text{fft}} = 1024$, hop = 512, Hann window, floor -80 dB, normalized linearly to $[0, 1]$).
  - Handles clips longer than 30 seconds via overlapping sliding windows (window length 1,292 frames with 50% overlap of 646 frames) and averages predicted continuous coordinates and logits across windows.
  - Executes `weights/audio_mood.onnx` on CPU using `onnxruntime.InferenceSession` with `CPUExecutionProvider`.
  - Rescales continuous outputs from raw $[0, 1]$ to the standard Russell Circumplex range $[-1, 1]$:
    $$V_{[-1, 1]} = 2 \cdot V_{\text{raw}} - 1, \quad A_{[-1, 1]} = 2 \cdot A_{\text{raw}} - 1$$
  - Applies post-hoc temperature scaling ($T = 1.345202$ loaded from `weights/temperature.json`) to calibrate quadrant logits into well-calibrated probabilities.
  - Generates lightweight downsampled uint8 base64-encoded spectrogram payloads for client-side HTML5 canvas rendering.
- **Lyrical Mood Inference (`lyrics.py`)**:
  - Loads fine-tuned multilingual `XLM-RoBERTa-base` (278M parameters) on CPU from `weights/lyrics_model/`.
  - Includes automated script and language heuristics. If non-Latin or non-English text is detected, the model skips inference and returns a structured `language_not_validated` response.
  - For English lyrics, predicts 4-quadrant probabilities and maps them to continuous circumplex coordinates $[-1, 1]^2$ via quadrant centroids.
- **Label Parsing (`label.py`)**:
  - Ingests free-text human-assigned mood labels and normalizes them against the canonical taxonomy in `configs/mood_map.yaml`.
  - Maps recognized terms to one of the 4 canonical quadrants (Happy, Angry, Sad, Calm).
  - Ambiguous functional tags (e.g. *film*, *soundtrack*, *background*, *space*, *retro*) and unknown terms are mapped strictly to `"unknown"` with `mismatch: null`. The system never guesses or invents an unmapped label.
- **Mismatch Comparator (`comparator.py`)**:
  - Evaluates spatial Euclidean distance on $[-1, 1]^2$ normalized to $[0, 1]$.
  - Evaluates probability-weighted disagreement: $d_{\text{prob}} = 1.0 - p(Q_{\text{label}})$.
  - Balanced mismatch score: $S_{\text{mismatch}} = 0.5 \cdot d_{\text{norm}} + 0.5 \cdot d_{\text{prob}}$.
  - Compares against the calibrated threshold $\tau = 0.6000$.

### 2.2 Threshold Calibration (`backend/calibration/`)
- Evaluated strictly on the held-out **validation split** (`val_ids` from `data/splits/splits_deam_pmemo.json`, 234 tracks).
- Evaluated against synthetic mismatches created by swapping labels on a random 15% of validation tracks (`seed=42`).
- Threshold swept across $\tau \in [0.10, 0.90]$ with step $0.01$. Optimal threshold maximizing F1-score:
  $$\tau^* = 0.6000 \quad (\text{F1} = 0.6027, \text{Precision} = 0.5789, \text{Recall} = 0.6286)$$
- Complete curve and metadata saved to `backend/calibration/threshold.json`.
- Zero test-split data was used for calibration.

### 2.3 Verification Agent (`backend/app/agent/`)
- Autonomous tool-calling loop triggered exclusively when a mismatch is flagged ($S_{\text{mismatch}} \ge \tau$).
- 5 modular tools: `get_audio_features`, `predict_audio_mood`, `fetch_lyrics`, `analyze_lyrics_mood`, `compare_signals`.
- Lyrics lookup queries LRCLIB first, with fallback to Genius if `GENIUS_API_KEY` is present. Enforces 3.0s timeouts and sends only title and artist strings (never audio).
- Deterministic rule-based reasoning engine generates verdicts (`genuine_mismatch`, `ambiguous`, `false_alarm`) and quotes real empirical numbers in explanations.
- Records a complete audit trace with execution duration in milliseconds.

### 2.4 FastAPI Service & Endpoints (`backend/app/main.py`)
- `POST /api/analyze`: Multipart upload accepting audio file, optional label, title, artist, and lyrics.
- `GET /api/metrics`: Serves `reports_train/metrics.json` unchanged.
- `GET /api/health`: Returns system status and model loading flag.
- Static mount at `/`: Serves static web pages with HTML5 fallbacks.
- Privacy compliance: Uploads are stored in a temporary directory and deleted in a `finally` block immediately. Rate limits enforced per client IP. Server logs record only request ID, timing, and HTTP status codes (never audio, titles, artists, or lyrics).

### 2.5 Web Interface (`backend/static/`)
- Pages: Analyze (`/`), Results (`/results.html`), Privacy (`/privacy.html`), Terms (`/terms.html`), Credits (`/credits.html`).
- Design System: Warm off-white background (`#F3F0E8`), near-black text (`#15140F`), burnt rust accent (`#C8401B`), solid flat colors (0 gradients), rectangular buttons with 2-4px radius (0 pill buttons), 0 emoji, 0 hype words, 0 em dashes.
- Self-hosted fonts: IBM Plex Serif, IBM Plex Sans, IBM Plex Mono under SIL Open Font License 1.1.
- Accessibility: High contrast ratio (> 15:1), visible focus outlines, labels on all inputs, `aria-live` status regions, keyboard navigable.
- Verified by `backend/check_rules.py` with zero violations.

---

## 3. Measured System Latency and Performance

All latency metrics were measured on the local deployment:

| Operation | Measured Latency | Notes |
| :--- | :---: | :--- |
| **Full Audio Analysis (30s file)** | **7,090 ms to 7,479 ms** | End-to-end: librosa decoding, STFT, mel extraction, ONNX CPU inference, XLM-RoBERTa lyrics, mismatch evaluation |
| **Audio Only (ONNX CPU forward pass)** | **16.7 ms** | Pure ONNX Runtime forward pass on 1,292 time frames |
| **Lyrics Analysis (XLM-RoBERTa CPU)** | **4,074 ms** | 128-token sequence classification on CPU |
| **Health Check (`GET /api/health`)** | **< 2 ms** | In-memory status check |
| **Metrics Fetch (`GET /api/metrics`)** | **< 2 ms** | Direct file read |
| **Static HTML / CSS Delivery** | **< 5 ms** | Direct static file streaming |

---

## 4. Known Limitations and Critical Observations

1. **Acoustic Valence Modeling Weakness**:
   Continuous valence estimation remains more challenging than arousal estimation from acoustic features alone. Minor chord progressions in energetic rock or electronic songs can communicate uplifting energy despite lower acoustic harmonic consonance.
2. **MTG-Jamendo Tag Noise**:
   The autotagging mood/theme subset comprises uncurated crowdsourced user tags. While useful for Stage 1 representation learning, tag frequencies vary substantially.
3. **Regional Music (Telugu / Indian Film Music)**:
   The lyrics model was fine-tuned on English emotional benchmarks (GoEmotions and MoodyLyrics). While the base `XLM-RoBERTa` encoder has subword representations for Indian languages, colloquial regional poetry and film lyric idioms have not been fine-tuned. The system detects non-English text and marks it as `language_not_validated` to avoid hallucinated predictions.
4. **Synthetic Mismatch Threshold as a Proxy**:
   The operational threshold ($\tau = 0.6000$) was calibrated using synthetic label swaps on 15% of validation tracks. In real music catalogs, editorial mislabeling errors frequently involve adjacent-quadrant confusion rather than random permutation.

---

## 5. What the User Must Do Manually

Before deploying this service to public production:

1. **Populate Legal Placeholders**:
   Update `[ENTITY NAME]`, `[CONTACT EMAIL]`, `[GOVERNING LAW]`, and `[EFFECTIVE DATE]` in `backend/static/privacy.html` and `backend/static/terms.html`. The server emits a startup log warning as long as these placeholders remain.
2. **Review External Lyrics API Terms**:
   Verify LRCLIB API terms of service and commercial compliance for your use case before launching public operations (referenced in `backend/static/credits.html`).
3. **Optional External Service Keys**:
   If you wish to enable the Genius lyrics fallback or hosted LLM reasoning in the verification agent, populate `GENIUS_API_KEY` and your provider key in `backend/.env`.
4. **Production Hosting and Domain**:
   Configure a reverse proxy (e.g. Nginx or Caddy) with TLS/HTTPS certificates to proxy traffic to the FastAPI backend (e.g. on port 8050) and open the appropriate firewall rules.

---

## 6. Audit Gate Verification Evidence

- **Test Suite Results**:
  `tests/` contains 36 passing tests covering GPU hardware, preprocessing parity, weights SHA256 integrity, ONNX CPU inference, lyrics language detection, threshold calibration without test leakage, verification agent verdicts, and FastAPI endpoints.
  ```
  ================== 36 passed, 10 warnings in 95.09s ==================
  ```
- **Rule Checker**:
  `backend/check_rules.py` executed and confirmed 0 em dashes, 0 gradients, 0 pill shapes, 0 emoji, and 0 banned hype words across `backend/static/`.
- **Live Server Test**:
  `backend/live_test.py` executed live requests on port 8050, verified model-not-loaded 503 fallback, confirmed numerical parity between UI and API, and generated responsive screenshots in `reports/screens/` at 360px, 768px, and 1440px.

---
*Backend engineering report completed and verified.*
