"""
Audio Feature Extraction Module for Label Audit.
Decodes audio to mono 22.05 kHz, computes 128-mel log-spectrograms,
and extracts tabular Librosa features with complete major/minor key estimation.
"""

import warnings
import numpy as np
import librosa
import soundfile as sf

TARGET_SR = 22050
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
MAX_FRAMES = 1292  # ~30 seconds at sr=22050, hop_length=512

# Krumhansl-Schmuckler key profiles for major/minor key estimation
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

def load_audio_mono(audio_path, target_sr=TARGET_SR, duration=30.0):
    """
    Decodes audio to mono 22.05 kHz, trimmed or padded to standard duration.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, sr = librosa.load(audio_path, sr=target_sr, mono=True, duration=duration)
    
    if len(y) == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")
    
    # Clean any accidental NaNs or Infs in raw audio
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y, sr

def compute_mel_spectrogram(y, sr=TARGET_SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, max_frames=MAX_FRAMES):
    """
    Computes 128-mel log-mel spectrogram normalized to [0, 1] range.
    Returns array of shape (n_mels, max_frames).
    """
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=20, fmax=sr // 2
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = np.nan_to_num(log_mel, nan=-80.0, posinf=0.0, neginf=-80.0)

    # Pad or truncate to max_frames
    if log_mel.shape[1] < max_frames:
        pad_width = max_frames - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode="constant", constant_values=-80.0)
    else:
        log_mel = log_mel[:, :max_frames]

    # Normalize to [0, 1] for stable neural network inputs
    log_mel = (log_mel + 80.0) / 80.0
    log_mel = np.clip(log_mel, 0.0, 1.0).astype(np.float32)
    return log_mel

def estimate_key_mode(chroma):
    """
    Estimates key mode: 1 for Major, 0 for Minor using correlation with Krumhansl-Schmuckler profiles.
    """
    chroma_mean = np.mean(chroma, axis=1)
    if np.sum(chroma_mean) > 0:
        chroma_mean = chroma_mean / np.sum(chroma_mean)
    else:
        return 1.0  # default to major

    best_major_corr = -1.0
    best_minor_corr = -1.0

    for shift in range(12):
        shifted_chroma = np.roll(chroma_mean, -shift)
        maj_corr = np.corrcoef(shifted_chroma, MAJOR_PROFILE)[0, 1]
        min_corr = np.corrcoef(shifted_chroma, MINOR_PROFILE)[0, 1]
        if not np.isnan(maj_corr) and maj_corr > best_major_corr:
            best_major_corr = maj_corr
        if not np.isnan(min_corr) and min_corr > best_minor_corr:
            best_minor_corr = min_corr

    return 1.0 if best_major_corr >= best_minor_corr else 0.0

def extract_librosa_features(y, sr=TARGET_SR):
    """
    Extracts comprehensive tabular acoustic features:
    - tempo (BPM)
    - RMS energy (mean, std)
    - Spectral Centroid (mean, std)
    - Spectral Rolloff (mean, std)
    - Spectral Bandwidth (mean, std)
    - Zero Crossing Rate (mean, std)
    - MFCCs 1-20 (means and stds = 40 features)
    - Chroma 12 pitch classes (means and stds = 24 features)
    - Major/minor key mode estimate (1.0 or 0.0)
    Total dimension: 74 features.
    """
    features = {}

    # 1. Tempo
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo_val = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
    features["tempo"] = float(tempo_val[0]) if len(tempo_val) > 0 else 120.0

    # 2. RMS Energy
    rms = librosa.feature.rms(y=y)[0]
    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))

    # 3. Spectral Centroid
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features["spectral_centroid_mean"] = float(np.mean(sc))
    features["spectral_centroid_std"] = float(np.std(sc))

    # 4. Spectral Rolloff
    sr_feat = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features["spectral_rolloff_mean"] = float(np.mean(sr_feat))
    features["spectral_rolloff_std"] = float(np.std(sr_feat))

    # 5. Spectral Bandwidth
    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features["spectral_bandwidth_mean"] = float(np.mean(sb))
    features["spectral_bandwidth_std"] = float(np.std(sb))

    # 6. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"] = float(np.std(zcr))

    # 7. MFCCs (20 coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(20):
        features[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))

    # 8. Chroma & Key Mode
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    for i in range(12):
        features[f"chroma_{i+1}_mean"] = float(np.mean(chroma[i]))
        features[f"chroma_{i+1}_std"] = float(np.std(chroma[i]))

    features["key_mode"] = float(estimate_key_mode(chroma))

    # Clean any NaNs or Infs
    feature_vector = np.array(list(features.values()), dtype=np.float32)
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
    return feature_vector, list(features.keys())
