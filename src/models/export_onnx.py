"""
ONNX Export and CPU Parity Verification Module for Label Audit.
Exports trained MelSpectrogramCNN to weights/audio_mood.onnx,
runs inference using ONNX Runtime on CPU, and asserts output parity with PyTorch within 1e-3.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
from src.models.audio_model import MelSpectrogramCNN

class AudioMoodONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x, mode="va")
        return out["va"], out["quadrant_logits"]

def export_audio_model_to_onnx(pt_path="weights/audio_mood_best.pt", onnx_path="weights/audio_mood.onnx"):
    print(f"Loading PyTorch model weights from {pt_path}...")
    checkpoint = torch.load(pt_path, map_location="cpu")
    model = MelSpectrogramCNN()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    wrapper = AudioMoodONNXWrapper(model)
    wrapper.eval()

    # Dummy input: (1, 1, 128, 1292)
    dummy_input = torch.randn(1, 1, 128, 1292, dtype=torch.float32)

    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    print(f"Exporting model to ONNX: {onnx_path}...")

    torch.onnx.export(
        wrapper,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["mel_spectrogram"],
        output_names=["va_pred", "quadrant_logits"],
        dynamic_axes={
            "mel_spectrogram": {0: "batch_size"},
            "va_pred": {0: "batch_size"},
            "quadrant_logits": {0: "batch_size"}
        },
        dynamo=False
    )

    # 1. Verify ONNX structure
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model structure checked and valid!")

    # 2. Check Parity with ONNX Runtime on CPU
    print("\n--- Running ONNX Runtime CPU Parity Check ---")
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    test_input = torch.randn(2, 1, 128, 1292, dtype=torch.float32)

    with torch.no_grad():
        pt_va, pt_quad = wrapper(test_input)

    ort_inputs = {"mel_spectrogram": test_input.numpy()}
    ort_outputs = session.run(None, ort_inputs)
    ort_va, ort_quad = ort_outputs[0], ort_outputs[1]

    va_diff = np.max(np.abs(pt_va.numpy() - ort_va))
    quad_diff = np.max(np.abs(pt_quad.numpy() - ort_quad))

    print(f"Valence/Arousal Max Absolute Difference: {va_diff:.6f}")
    print(f"Quadrant Logits Max Absolute Difference: {quad_diff:.6f}")

    assert va_diff < 1e-3, f"VA parity error too high: {va_diff}"
    assert quad_diff < 1e-3, f"Quadrant parity error too high: {quad_diff}"
    print("[PASS] ONNX CPU parity check passed within 1e-3 tolerance!")

    return onnx_path

if __name__ == "__main__":
    export_audio_model_to_onnx()
