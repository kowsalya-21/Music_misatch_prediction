# Preprocessing Parity Document: Label Audit

## 1. Overview and Verification Status

This document establishes the exact mathematical and algorithmic preprocessing steps used during training and required for runtime inference parity in the backend service. All steps are derived directly from the training codebase in `src/features/audio_features.py`, `src/features/preprocess.py`, `src/models/audio_model.py`, and `weights/norm_stats.json`.

All model weight files in `weights/` have been verified against `weights/SHA256SUMS` with zero discrepancies.

---

## 2. Audio Preprocessing Pipeline

### 2.1 Audio Loading and Decoding
- **Target Sample Rate**: `22,050 Hz` (librosa default resampler `soxr_hq` or standard scipy).
- **Channels**: Mono (`mono=True`). Multi-channel audio is averaged across channels: $y = \frac{1}{C}\sum_{c=1}^C y_c$.
- **Format Support**: Supports `.mp3`, `.wav`, `.ogg`, `.flac`, `.m4a` via soundfile and audioread backends.
- **Duration / Truncation**: Standard training window is 30.0 seconds ($30 \times 22,050 = 661,500$ samples).
- **Numerical Sanitization**: Any NaN or Infinite values in raw waveform samples are mapped to 0.0:
  $$y = \text{np.nan\_to\_num}(y, \text{nan}=0.0, \text{posinf}=0.0, \text{neginf}=0.0)$$

### 2.2 Mel-Spectrogram Extraction (`MelSpectrogramCNN`)
- **Number of Mel Bins ($N_{\text{mels}}$)**: `128`
- **FFT Window Length ($N_{\text{fft}}$)**: `1,024` samples (~46.4 ms)
- **Hop Length**: `512` samples (~23.2 ms)
- **Window Type**: Hann window
- **Frequency Range**: $f_{\min} = 20\text{ Hz}$, $f_{\max} = 11,025\text{ Hz}$ ($f_s / 2$)
- **Power**: Power spectrogram ($|STFT|^2$)
- **Decibel Scaling**: Logarithmic decibel compression using maximum reference power:
  $$\text{log\_mel} = 10 \cdot \log_{10}\left(\frac{\max(\text{mel}, 10^{-10})}{\max(\text{mel})}\right)$$
  Floor set to $-80.0\text{ dB}$.
- **Temporal Alignment (Frame Length)**:
  - Exact frame count: $T = 1,292$ time frames ($\sim 30.0$ seconds).
  - Short audio ($T < 1,292$): Right-padded with constant value $-80.0\text{ dB}$.
  - Long audio ($T > 1,292$): Evaluated via overlapping sliding windows (window size $1,292$ frames, 50% hop of $646$ frames) with predicted continuous valence/arousal coordinates and quadrant logits averaged across windows.
- **Normalization to $[0, 1]$ Range**:
  Neural network inputs were scaled linearly during training using:
  $$\text{norm\_mel} = \text{clip}\left(\frac{\text{log\_mel} + 80.0}{80.0}, 0.0, 1.0\right)$$
  Tensor shape expected by `audio_mood.onnx`: `(batch_size, 1, 128, 1292)` in float32.

### 2.3 Acoustic Librosa Tabular Features (76 Dimensions)
Extracted for LightGBM and SVM baselines:
1. `tempo`: Global BPM via onset envelope autocorrelation (1 feature)
2. `rms_mean`, `rms_std`: Root mean square energy (2 features)
3. `spectral_centroid_mean`, `spectral_centroid_std`: Frequency center of mass (2 features)
4. `spectral_rolloff_mean`, `spectral_rolloff_std`: 85% energy rolloff frequency (2 features)
5. `spectral_bandwidth_mean`, `spectral_bandwidth_std`: Spectral spread (2 features)
6. `zcr_mean`, `zcr_std`: Zero-crossing rate (2 features)
7. `mfcc_1_mean` through `mfcc_20_mean`, `mfcc_1_std` through `mfcc_20_std`: 20 Mel-Frequency Cepstral Coefficients (40 features)
8. `chroma_1_mean` through `chroma_12_mean`, `chroma_1_std` through `chroma_12_std`: 12 Chroma pitch classes (24 features)
9. `key_mode`: Major (1.0) vs Minor (0.0) estimate using Krumhansl-Schmuckler pitch class profiles correlation (1 feature)
- **Feature Standardization**:
  $$x_{\text{norm}} = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
  $\mu_{\text{train}}$ and $\sigma_{\text{train}}$ are loaded strictly from `weights/norm_stats.json`.

---

## 3. Post-Processing and Output Transformations

### 3.1 Valence-Arousal Coordinates
- **Raw Model Output**: Sigmoid activation produces continuous values in $[0, 1]$:
  $$V_{\text{raw}} \in [0, 1], \quad A_{\text{raw}} \in [0, 1]$$
- **Circumplex Standard Rescaling**: Mapped linearly to the standard Russell Circumplex range $[-1, 1]$:
  $$V_{[-1, 1]} = 2 \cdot V_{\text{raw}} - 1, \quad A_{[-1, 1]} = 2 \cdot A_{\text{raw}} - 1$$
- **Quadrant Mapping**:
  - Quadrant 1 (Happy): $V \ge 0, A \ge 0$
  - Quadrant 2 (Angry): $V < 0, A \ge 0$
  - Quadrant 3 (Sad): $V < 0, A < 0$
  - Quadrant 4 (Calm): $V \ge 0, A < 0$

### 3.2 Temperature Calibration for Quadrants
- **Raw Logits**: $z = (z_0, z_1, z_2, z_3)$ for [Happy, Angry, Sad, Calm].
- **Calibrated Softmax**: Temperature $T = 1.345202$ loaded from `weights/temperature.json`:
  $$p_i = \frac{e^{z_i / T}}{\sum_{j=0}^3 e^{z_j / T}}$$

---

## 4. Lyrics Model Preprocessing Pipeline

- **Tokenizer**: Pretrained SentencePiece tokenizer loaded from `weights/lyrics_model/` (`xlm-roberta-base`).
- **Input Cleaning**: Unicode normalization (NFKC), strip surrounding whitespace, lowercase handling preserved by tokenizer.
- **Script and Language Detection**:
  - The model was fine-tuned on English emotional benchmarks (GoEmotions and MoodyLyrics).
  - Any non-Latin script (such as Telugu, Devanagari, Arabic, Cyrillic) or non-English text is detected using langdetect / unicodedata heuristics.
  - If text is not English, the model is not executed and returns:
    `{"status": "language_not_validated", "note": "Lyrics model validated for English only. Regional transfer pending fine-tuning."}`
- **Token Truncation**: Max length 128 subword tokens (`truncation=True`, `padding="max_length"`).
- **Quadrant Mapping**: Logits projected to 4 quadrants, converted to probability distribution via Softmax, and mapped to continuous centroid proxy $(V, A) \in [-1, 1]^2$.

---

## 5. Free-Text Human Label Parsing

- **Canonical Mapping**: Labels are looked up in `configs/mood_map.yaml` (case-insensitive, trimmed).
- **Ambiguous Tags**: Tags defined in `excluded_ambiguous_tags` (e.g., *film*, *soundtrack*, *heavy*, *slow*, *fast*) and tags not in `mood_quadrant_map` map to `"unknown"`.
- **Mismatch Policy**: If a label is "unknown", the system returns `quadrant: "unknown"` and `mismatch: null`. The system never guesses an unmapped label.

---

## 6. Inventory of Project Artifacts

All required files are verified present:
- `weights/audio_mood.onnx`: PRESENT (10.8 MB)
- `weights/audio_mood_best.pt`: PRESENT (10.8 MB)
- `weights/baseline_lgbm_valence.txt`: PRESENT
- `weights/baseline_lgbm_arousal.txt`: PRESENT
- `weights/baseline_svm.pkl`: PRESENT
- `weights/lyrics_model/model.safetensors`: PRESENT (1.11 GB)
- `weights/lyrics_model/config.json`: PRESENT
- `weights/lyrics_model/tokenizer.json`: PRESENT
- `weights/temperature.json`: PRESENT
- `weights/norm_stats.json`: PRESENT
- `weights/mood_map.yaml`: PRESENT
- `weights/MODEL_CARD.md`: PRESENT
- `weights/SHA256SUMS`: PRESENT
- `reports_train/metrics.json`: PRESENT
- `configs/mood_map.yaml`: PRESENT
- `data/splits/splits_deam_pmemo.json`: PRESENT
- `requirements.txt`: PRESENT

No required files are missing.
