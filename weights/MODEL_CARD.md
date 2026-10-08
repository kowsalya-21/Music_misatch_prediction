# Model Card: Label Audit Audio & Lyrics Mood Models

## 1. Overview
Label Audit utilizes machine learning models to analyze musical audio and lyrical text for mood classification and valence-arousal dimension estimation:
1. **Audio Deep Learning Model**: `MelSpectrogramCNN` (5-stage convolutional neural network trained from scratch, exported to PyTorch and ONNX formats).
2. **Audio Baselines**: LightGBM and Support Vector Regressors (SVR) trained on 76 Librosa acoustic features.
3. **Lyrics Deep Learning Model**: Multilingual transformer encoder (`XLM-RoBERTa-base`) fine-tuned for 4-quadrant sentiment classification.

## 2. Architectures & Specifications

### MelSpectrogramCNN (Audio Model)
- **Input Representation**: 128-bin log-mel spectrogram, standardized to 1,292 time frames (~30s duration) at 22,050 Hz sampling rate.
- **Backbone Architecture**:
  - 5 sequential convolutional blocks (Conv2D 3x3, BatchNorm2D, GELU, MaxPool2D, Dropout2D).
  - Feature channels: 32 -> 64 -> 128 -> 256 -> 256.
  - Global Pooling: Concatenated Adaptive Average Pooling and Adaptive Max Pooling yielding a 512-dimensional embedding.
- **Stage 1 (MTG-Jamendo Pretraining)**:
  - Multi-label classification head: Linear(512 -> 256), GELU, Dropout, Linear(256 -> 37 tags) with BCEWithLogitsLoss.
- **Stage 2 (DEAM + PMEmo Fine-tuning)**:
  - Shared projection: Linear(512 -> 256), LayerNorm, GELU, Dropout(0.2).
  - Valence-Arousal Regression Head: Linear(256 -> 128), GELU, Linear(128 -> 2), Sigmoid() predicting continuous $(V, A) \in [0, 1]^2$.
  - 4-Quadrant Mood Classification Head: Linear(256 -> 128), GELU, Linear(128 -> 4) predicting logits for Happy, Angry, Sad, Calm.
- **Probability Calibration**: Temperature scaling parameter $T = 1.3452$ fitted on validation negative log-likelihood.
- **Export**: Exported to ONNX (`weights/audio_mood.onnx`) with verified CPU inference parity ($< 10^{-6}$ max error vs PyTorch).

### XLM-RoBERTa-base (Lyrics Model)
- **Base Architecture**: Multilingual Transformer encoder (`xlm-roberta-base`, 278M parameters, 100 languages supported including Telugu and English).
- **Classification Head**: Dense projection with Dropout and Linear layer mapping pooled token embeddings to the 4 canonical quadrants (Happy, Angry, Sad, Calm).

### Tabular Baselines
- **LightGBM Regressors**: Gradient boosted decision trees (150 estimators, learning rate 0.05, num leaves 31, subsample 0.8) predicting continuous valence and arousal from 76 Librosa features.
- **Support Vector Regressors (SVR)**: RBF kernel, $C=1.0$, $\epsilon=0.05$ on standardized Librosa features.

## 3. Datasets and Licenses

| Dataset | Modality | Samples / Tracks | License | Role in System |
| :--- | :--- | :--- | :--- | :--- |
| **MTG-Jamendo** | Audio + Tags | 992 tracks (autotagging_moodtheme) | CC BY-NC-SA 4.0 | Stage 1 pretraining for acoustic mood representation |
| **DEAM** | Audio + Continuous VA | 1,802 tracks | CC BY-NC-ND 4.0 | Stage 2 valence/arousal regression and quadrant labeling |
| **PMEmo** | Audio + Lyrics + VA | 794 tracks (629 lyrics) | Academic / Research | Stage 2 valence/arousal regression and quadrant labeling |
| **GoEmotions** | Text (Reddit comments) | 54,263 utterances | Apache 2.0 | Lyrics model fine-tuning (mapped to 4 quadrants) |
| **MoodyLyrics4Q** | Lyrical Sentiment | 2,000 song titles & artists | Academic / Research | Lyrics model fine-tuning (4 balanced quadrants) |
| **XLM-RoBERTa-base** | Pretrained Encoder | 278M parameters | MIT License | Multilingual base text encoder |

## 4. Training Configuration & Hyperparameters

- **Hardware**: NVIDIA DGX B200 (Blackwell architecture, Compute Capability 10.0).
- **Precision**: Brain Floating Point 16 (`torch.bfloat16`) autocast.
- **Random Seeds Evaluated**: `42`, `1337`, `2026`.
- **Audio Augmentations**: SpecAugment (time masking, frequency masking), random gain jitter ($\pm 20\%$), random time shift/roll.
- **Optimizer & Schedule**: AdamW with weight decay $10^{-4}$ and Cosine Annealing learning rate schedule.
- **Batch Sizes**: 64 for audio CNN and lyrics transformer.

## 5. Measured Evaluation Metrics (Held-out Test Split Only)

### Audio Models Comparison Table

| Model | Valence CCC | Arousal CCC | Valence RMSE | Arousal RMSE | Quadrant Accuracy | Quadrant Macro-F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LightGBM Baseline** | 0.7062 | 0.6205 | 0.1278 | 0.1364 | 0.6130 | 0.4407 |
| **SVM Baseline** | 0.6721 | 0.6180 | 0.1321 | 0.1383 | 0.6130 | 0.4356 |
| **Mel-CNN (Seed 42)** | 0.7388 | 0.7064 | 0.1221 | 0.1272 | 0.7174 | 0.4154 |
| **Mel-CNN (Seed 1337)** | 0.7389 | 0.7139 | 0.1227 | 0.1245 | 0.6957 | 0.4035 |
| **Mel-CNN (Seed 2026)** | 0.7159 | 0.7060 | 0.1264 | 0.1267 | 0.7043 | 0.4078 |
| **Mel-CNN (3 Seeds Avg)** | **0.7312** | **0.7088** | **0.1238** | **0.1261** | **0.7058** | **0.4089** |

*Selected Model*: `MelSpectrogramCNN` (Mean CCC 0.7200 vs LightGBM 0.6633).

### Lyrics Model Test Metrics (XLM-RoBERTa-base)
- **Held-out Test Accuracy**: 88.43%
- **Held-out Test Macro-F1**: 86.18%
- **Class-Level Performance**:
  - Happy: Precision 0.92, Recall 0.94, F1 0.93 (support: 2,080)
  - Angry: Precision 0.84, Recall 0.82, F1 0.83 (support: 970)
  - Sad: Precision 0.76, Recall 0.70, F1 0.73 (support: 382)
  - Calm: Precision 1.00, Recall 0.91, F1 0.96 (support: 93)

## 6. Known Limitations
1. **Valence Ambiguity**: Acoustic valence (musical positivity vs negativity) is notoriously more difficult to predict from acoustic spectral features alone than arousal (energy/activation), because minor key or distorted instrumentation can still communicate uplifting or triumphant themes.
2. **Tag Noise**: MTG-Jamendo mood tags are crowdsourced user folksonomy tags, which contain subjective label noise.
3. **Domain Shift in Lyrics**: GoEmotions derives from Reddit comments; although mapped to emotions, informal social media text differs in meter and metaphor from song lyric poetry.

## 7. Multilingual & Indic Language Support (Telugu Case Study)

Empirical evaluation on regional Indian music (Telugu audio tracks, Unicode script lyrics, and transliterations) demonstrated the following modality behaviors:

1. **Acoustic Generalization (Language-Agnostic)**:
   - `MelSpectrogramCNN` operates strictly on time-frequency acoustic representations (128-bin log-mel spectrogram, 22.05 kHz).
   - Evaluated on contemporary Telugu tracks with sliding window chunking (~209s), successfully predicting primary mood quadrants (e.g. `happy`, Valence $= +0.357$, Arousal $= +0.485$) and relevant acoustic tags (`['happy', 'party', 'fun', 'positive', 'upbeat']`) without language bias.

2. **Multilingual Zero-Shot Lyrics Transfer**:
   - `XLM-RoBERTa-base` natively supports Telugu tokens.
   - Tested on high-polarity lyrical sentiments:
     - **Sad Telugu Lyrics** (*"కన్నుల దాచిన కన్నీరై పోయావా ఎందుకీ వేదన బ్రతుకు భారమై"*): Correctly predicted `sad` with **86.01%** probability.
     - **Aggressive/Angry Telugu Lyrics** (*"రగిలే గుండెల్లో రౌద్రం ఆగదు శత్రువుని చంపేస్తా"*): Correctly predicted `angry` with **75.90%** probability.
     - **Energetic/Folk Beats ("Naatu Naatu")**: Skews toward `angry` (high arousal) in zero-shot transfer due to fast-paced phonetic density and lack of regional dance-folk sentiment tokens in English pretraining corpora.

3. **Taxonomy & Regional Vocabulary**:
   - `configs/mood_map.yaml` includes native Telugu sentiment mappings (`ఆనందం`, `బాధ`, `కోపం`, `ప్రశాంతం`, `మాస్`) mapped to the canonical Russell Circumplex quadrants, enabling continuous distance calculation and automated mismatch audits across regional and global music catalogues.

