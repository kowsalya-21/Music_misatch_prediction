---
title: Music Mismatch Prediction
emoji: 🎧
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8050
pinned: false
---

# 🎧 Label Audit: Music Mood and Editorial Mismatch Engine

[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![ONNX Runtime](https://img.shields.io/badge/Inference-ONNX%20Runtime%20CPU-005CED?style=flat-square)](https://onnxruntime.ai/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch%202.5-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

An academic and industrial Music Information Retrieval (MIR) intelligence platform. **Label Audit** cross-examines human distributor mood labels against continuous acoustic affect spaces (Russell's Circumplex Model) and multi-modal multilingual lyrical sentiment using empirical machine learning.

---

## 🌟 Key Features

* **Multi-Task Acoustic CNN (`MelSpectrogramCNN`)**:
  * Extracts **128-bin Log-Mel Spectrograms** at $22,050\text{ Hz}$ mono.
  * Jointly predicts continuous **$(Valence, Arousal) \in [-1.0, +1.0]^2$** dynamic coordinates and 4-quadrant probabilities (`Happy`, `Angry`, `Sad`, `Calm`).
  * Optimized via **ONNX Runtime on CPU** ($< 60\text{ ms}$ inference latency, no GPU needed).
* **Zero-Shot Multilingual Lyrical Analysis (`XLM-RoBERTa`)**:
  * Evaluates lyrical emotional sentiment across English, regional Indian languages, and Romanized Telugu (e.g. *"Samajavaragamana"*).
* **Calibrated Mismatch Comparator**:
  * Evaluates normalized Euclidean spatial distance and probability disagreement against an empirical calibrated boundary threshold ($\tau = 0.6000$).
* **Autonomous Verification Agent**:
  * 5-step investigative loop triggering on flagged discrepancies.
  * Cross-references audio features, acoustic mood, lyrics sentiment, and metadata to diagnose intro-sampling artifacts vs. genuine catalog mislabeling.
* **Online Catalog Search (Zero Audio File Required)**:
  * Automatically fetches official 30-second studio previews and lyrics online via iTunes Search API and LRCLIB.

---

## 📐 System Architecture

```
                       ┌──────────────────────────────────────────────┐
                       │               AUDIT WORKSPACE                │
                       │   Audio File (.mp3/.wav) OR Online Search    │
                       └──────────────────────┬───────────────────────┘
                                              │
              ┌───────────────────────────────┴───────────────────────────────┐
              ▼                                                               ▼
   ┌───────────────────────┐                                       ┌───────────────────────┐
   │    ACOUSTIC STREAM    │                                       │     LYRICS STREAM     │
   │  22,050 Hz Mono Audio │                                       │ LRCLIB / User Input   │
   │  Log-Mel Spectrogram  │                                       │ XLM-RoBERTa (Zero-Shot│
   │  MelSpectrogramCNN    │                                       │ Multilingual Embeds)  │
   │      (ONNX, CPU)      │                                       │                       │
   └──────────┬────────────┘                                       └───────────┬───────────┘
              │                                                                │
              ▼                                                                ▼
   Continuous (Valence, Arousal)                                    Continuous (Valence, Arousal)
   + Quadrant Probabilities                                         + Primary Quadrant
              │                                                                │
              └───────────────────────────────┬────────────────────────────────┘
                                              │
                                              ▼
                             ┌──────────────────────────────────┐
                             │       MISMATCH COMPARATOR        │
                             │  Calibrated Decision Threshold   │
                             │  (Threshold = 0.6000)            │
                             │  Euclidean Distance Across       │
                             │  Russell's Circumplex Space      │
                             └────────────────┬─────────────────┘
                                              │
                              ┌───────────────┴───────────────┐
                              ▼                               ▼
                      [Score <= 0.60]                 [Score > 0.60]
                      VERIFIED MATCH                  MISMATCH FLAGGED
                                                              │
                                                              ▼
                                              ┌──────────────────────────────────┐
                                              │   AUTONOMOUS VERIFICATION AGENT  │
                                              │   5-Step Tool Execution:         │
                                              │   • File features                │
                                              │   • Acoustic mood                │
                                              │   • Lyrics fetch                 │
                                              │   • Sentiment analysis           │
                                              │   • Multi-modal arbitration      │
                                              │   Verdict: genuine_mismatch /    │
                                              │            ambiguous /           │
                                              │            false_alarm           │
                                              └──────────────────────────────────┘
```

---

## 📊 Empirical Benchmarks (Held-out Test Split)

Evaluated strictly on held-out test splits (DEAM & PMEmo datasets):

| Architecture | Modality | Valence CCC | Arousal CCC | Quadrant Accuracy | Latency (CPU) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **LightGBM Baseline** | 76 Librosa Features | $0.7062$ | $0.6205$ | $61.30\%$ | $12\text{ ms}$ |
| **MelSpectrogramCNN (Selected)** | 128 Log-Mel Spectrogram | **$0.7392$** | **$0.7686$** | **$74.35\%$** | **$58\text{ ms}$** |
| **XLM-RoBERTa-base** | Raw Text / Lyrics | $0.6841$ | $0.7120$ | $72.10\%$ | $110\text{ ms}$ |

---

## 🛠️ Tech Stack

* **Backend**: FastAPI, Uvicorn, Python 3.11
* **Machine Learning**: PyTorch 2.5, ONNX Runtime, Hugging Face Transformers (`xlm-roberta-base`), LightGBM, Scikit-learn
* **Audio Engineering**: Librosa, SoundFile, FFmpeg, NumPy
* **Frontend**: Vanilla JavaScript (ES6+), HTML5 Canvas (Spectrogram rendering), SVG (Circumplex Affect Plane)

---

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have **Python 3.10+** and **FFmpeg** installed on your system.

### 2. Installation
```bash
git clone https://github.com/kowsalya-21/Music_misatch_prediction.git
cd Music_misatch_prediction

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements-deploy.txt
```

### 3. Start the Server
```bash
python run_server.py
```
Open your browser at **`http://127.0.0.1:8050`** to access the dashboard.

---

## 🐳 Docker Deployment

A production-ready `Dockerfile` is provided for containerized deployment (e.g. Hugging Face Spaces, Render, AWS, GCP):

```bash
docker build -t label-audit .
docker run -p 8050:8050 label-audit
```

---

## 📜 Citation & Credits

* **DEAM**: Database for Emotional Analysis of Music (Aljanaki et al.)
* **PMEmo**: A Dataset for Pop Music Emotion Analysis (Zhang et al.)
* **MTG-Jamendo**: Multi-label dataset for music autotagging (Bogdanov et al.)
* **GoEmotions & MoodyLyrics**: Lyrical emotion taxonomy mapping.

---

## ⚖️ License
This project is licensed under the [MIT License](LICENSE).
