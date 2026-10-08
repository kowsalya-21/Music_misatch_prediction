"""
Audio Inference Engine for Label Audit.
Executes exact training preprocessing, sliding-window chunking for long clips,
ONNX Runtime CPU inference with temperature scaling calibration,
Russell Circumplex [-1, 1] rescaling, and base64 mel visualization output.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import base64
import yaml
import numpy as np
import librosa
import onnxruntime as ort

TARGET_SR = 22050
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
MAX_FRAMES = 1292  # ~30 seconds standard window

QUADRANT_NAMES = ["happy", "angry", "sad", "calm"]

class AudioPredictor:
    def __init__(
        self,
        onnx_path="weights/audio_mood.onnx",
        temperature_path="weights/temperature.json",
        mood_map_path="configs/mood_map.yaml",
        norm_stats_path="weights/norm_stats.json"
    ):
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model file not found: {onnx_path}")
        
        # CPU Execution Provider
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        
        # Load Temperature
        if os.path.exists(temperature_path):
            with open(temperature_path) as f:
                t_data = json.load(f)
                self.temperature = float(t_data.get("temperature", 1.0))
        else:
            self.temperature = 1.0

        # Load Mood Map for Top Tags
        self.tag_map = {}
        if os.path.exists(mood_map_path):
            with open(mood_map_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                self.tag_map = cfg.get("jamendo_tag_map", {})
        
        # Organize tags by quadrant
        self.quadrant_tags = {"happy": [], "angry": [], "sad": [], "calm": []}
        for full_tag, quad in self.tag_map.items():
            clean_tag = full_tag.replace("mood/theme---", "")
            if quad in self.quadrant_tags:
                self.quadrant_tags[quad].append(clean_tag)

    @staticmethod
    def load_audio(file_path_or_buffer, target_sr=TARGET_SR):
        """
        Loads and decodes audio to mono float32 at target sampling rate.
        Multi-channel audio is averaged to mono.
        """
        y, sr = librosa.load(file_path_or_buffer, sr=target_sr, mono=True)
        if len(y) == 0:
            raise ValueError("Decoded audio signal is empty.")
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        return y, sr

    @staticmethod
    def compute_log_mel(y, sr=TARGET_SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
        """
        Computes 128-bin log-mel spectrogram normalized to [0, 1].
        Matches src/features/audio_features.py exactly.
        """
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=20, fmax=sr // 2
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        log_mel = np.nan_to_num(log_mel, nan=-80.0, posinf=0.0, neginf=-80.0)
        
        # Normalize dB scale [-80, 0] to [0, 1]
        norm_mel = (log_mel + 80.0) / 80.0
        norm_mel = np.clip(norm_mel, 0.0, 1.0).astype(np.float32)
        return norm_mel, log_mel

    @staticmethod
    def prepare_display_mel(log_mel, downsample_factor=4):
        """
        Generates quantized uint8 mel spectrogram for web UI visualization.
        Downsampled in time to keep payload lightweight.
        """
        # Downsample along time dimension
        downsampled = log_mel[:, ::downsample_factor]
        n_mels, n_frames = downsampled.shape
        
        # Map [-80 dB, 0 dB] -> [0, 255]
        clipped = np.clip(downsampled, -80.0, 0.0)
        normalized = ((clipped + 80.0) / 80.0 * 255.0).astype(np.uint8)
        
        b64_data = base64.b64encode(normalized.tobytes()).decode("ascii")
        return {
            "n_mels": int(n_mels),
            "n_frames": int(n_frames),
            "db_min": -80.0,
            "db_max": 0.0,
            "downsample_factor": downsample_factor,
            "uint8_data": b64_data
        }

    def predict_mel_array(self, norm_mel):
        """
        Runs ONNX model on a normalized mel array of shape (128, T).
        Handles window chunking for long clips and averaging.
        """
        total_frames = norm_mel.shape[1]
        
        if total_frames <= MAX_FRAMES:
            # Pad to 1,292 if shorter
            if total_frames < MAX_FRAMES:
                pad_width = MAX_FRAMES - total_frames
                window = np.pad(norm_mel, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)
            else:
                window = norm_mel[:, :MAX_FRAMES]
            windows = [window]
        else:
            # Sliding window with 50% overlap (hop = 646 frames ~15s)
            windows = []
            hop = MAX_FRAMES // 2
            start = 0
            while start + MAX_FRAMES <= total_frames:
                windows.append(norm_mel[:, start:start + MAX_FRAMES])
                start += hop
            # Ensure tail is captured
            if start < total_frames and total_frames - start >= hop // 2:
                tail = norm_mel[:, -MAX_FRAMES:]
                windows.append(tail)

        # Batch inputs: (B, 1, 128, 1292)
        batch_inputs = np.stack(windows, axis=0)[:, np.newaxis, :, :].astype(np.float32)
        
        ort_outputs = self.session.run(None, {"mel_spectrogram": batch_inputs})
        va_preds = ort_outputs[0]        # (B, 2) in [0, 1]
        quad_logits = ort_outputs[1]     # (B, 4)

        # Average across sliding windows for long clips
        avg_va_raw = np.mean(va_preds, axis=0)       # (2,) in [0, 1]
        avg_quad_logits = np.mean(quad_logits, axis=0) # (4,)

        # 1. Rescale Valence and Arousal from [0, 1] to Circumplex [-1, 1]
        # Rescaling Formula: V_circ = 2 * V_raw - 1, A_circ = 2 * A_raw - 1
        v_rescaled = float(2.0 * avg_va_raw[0] - 1.0)
        a_rescaled = float(2.0 * avg_va_raw[1] - 1.0)
        v_rescaled = max(-1.0, min(1.0, v_rescaled))
        a_rescaled = max(-1.0, min(1.0, a_rescaled))

        # 2. Calibrate Quadrant Logits with Temperature Scaling
        calibrated_logits = avg_quad_logits / self.temperature
        exp_logits = np.exp(calibrated_logits - np.max(calibrated_logits))
        quad_probs = exp_logits / np.sum(exp_logits)

        quad_dict = {
            "happy": float(quad_probs[0]),
            "angry": float(quad_probs[1]),
            "sad": float(quad_probs[2]),
            "calm": float(quad_probs[3])
        }

        # 3. Determine Primary Quadrant
        best_quad_idx = int(np.argmax(quad_probs))
        primary_quadrant = QUADRANT_NAMES[best_quad_idx]

        # 4. Derive Top Mood Tags from Weighted Quadrants
        top_tags = []
        for q_name in sorted(quad_dict, key=quad_dict.get, reverse=True):
            prob = quad_dict[q_name]
            for tag in self.quadrant_tags.get(q_name, []):
                if tag not in top_tags:
                    top_tags.append(tag)
                if len(top_tags) >= 5:
                    break
            if len(top_tags) >= 5:
                break

        return {
            "valence": v_rescaled,
            "arousal": a_rescaled,
            "raw_valence": float(avg_va_raw[0]),
            "raw_arousal": float(avg_va_raw[1]),
            "quadrant_probs": quad_dict,
            "primary_quadrant": primary_quadrant,
            "top_tags": top_tags[:5],
            "num_windows": len(windows)
        }

    def predict_audio_file(self, audio_path):
        """
        End-to-end audio pipeline:
        Loads audio, computes log-mel, runs inference, and returns predictions + display mel.
        """
        y, sr = self.load_audio(audio_path)
        duration_s = float(len(y) / sr)
        norm_mel, raw_log_mel = self.compute_log_mel(y, sr)
        
        preds = self.predict_mel_array(norm_mel)
        display_mel = self.prepare_display_mel(raw_log_mel, downsample_factor=4)
        
        preds["duration_s"] = duration_s
        preds["mel_display"] = display_mel
        return preds
