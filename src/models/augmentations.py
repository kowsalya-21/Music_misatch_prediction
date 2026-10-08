"""
Spectrogram Augmentations Module for Label Audit.
Implements SpecAugment (time and frequency masking), random gain, random crop,
and time shift directly on PyTorch tensors for zero CPU bottleneck.
"""

import torch
import torch.nn as nn

class SpecAugment(nn.Module):
    def __init__(self, freq_mask_param=16, time_mask_param=64, n_freq_masks=2, n_time_masks=2):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def forward(self, x):
        """
        x: Tensor of shape (B, 1, n_mels, time_steps)
        """
        if not self.training:
            return x

        B, C, F, T = x.shape
        cloned = x.clone()

        # Frequency masking
        for _ in range(self.n_freq_masks):
            f_len = torch.randint(0, self.freq_mask_param, (B,), device=x.device)
            f_start = torch.randint(0, F - self.freq_mask_param + 1, (B,), device=x.device)
            for b in range(B):
                cloned[b, :, f_start[b]:f_start[b] + f_len[b], :] = 0.0

        # Time masking
        for _ in range(self.n_time_masks):
            t_len = torch.randint(0, self.time_mask_param, (B,), device=x.device)
            t_start = torch.randint(0, T - self.time_mask_param + 1, (B,), device=x.device)
            for b in range(B):
                cloned[b, :, :, t_start[b]:t_start[b] + t_len[b]] = 0.0

        return cloned

class AudioSpectrogramAugment(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.spec_aug = SpecAugment()
        self.p = p

    def forward(self, x):
        """
        x: (B, 1, 128, T)
        """
        if not self.training:
            return x

        B, C, F, T = x.shape

        # Random gain jittering (+- 20%)
        if torch.rand(1).item() < self.p:
            gain = 0.8 + 0.4 * torch.rand((B, 1, 1, 1), device=x.device)
            x = torch.clamp(x * gain, 0.0, 1.0)

        # Random time shift/roll
        if torch.rand(1).item() < self.p:
            shift = torch.randint(-T // 10, T // 10, (1,)).item()
            x = torch.roll(x, shifts=shift, dims=-1)

        # SpecAugment (time & freq mask)
        if torch.rand(1).item() < self.p:
            x = self.spec_aug(x)

        return x
