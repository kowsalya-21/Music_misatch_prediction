"""
Mel-Spectrogram CNN Architecture for Label Audit.
Trained from scratch (no pretrained weights).
Features a shared convolutional backbone with Stage 1 multi-label tag head
and Stage 2 multi-task valence/arousal regression + 4-quadrant classification heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        return x

class MelSpectrogramCNN(nn.Module):
    def __init__(self, num_tags=37, num_quadrants=4, embedding_dim=512):
        super().__init__()
        self.num_tags = num_tags
        self.num_quadrants = num_quadrants
        self.embedding_dim = embedding_dim

        # 5-stage convolutional backbone (from scratch)
        self.block1 = ConvBlock(1, 32, dropout=0.05)    # -> (32, 64, 646)
        self.block2 = ConvBlock(32, 64, dropout=0.1)    # -> (64, 32, 323)
        self.block3 = ConvBlock(64, 128, dropout=0.15)  # -> (128, 16, 161)
        self.block4 = ConvBlock(128, 256, dropout=0.2)  # -> (256, 8, 80)
        self.block5 = ConvBlock(256, 256, dropout=0.25) # -> (256, 4, 40)

        # Global Pooling (Mean + Max pooling concatenation = 512 dims)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        # Stage 1: Multi-label Jamendo tag head
        self.tag_head = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_tags)
        )

        # Stage 2: Fine-tuning Heads (DEAM + PMEmo)
        self.shared_proj = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        # Continuous Valence & Arousal Regression (outputs in [0, 1])
        self.va_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )
        # 4-Quadrant Mood Classification Head
        self.quadrant_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, num_quadrants)
        )

    def extract_features(self, x):
        """
        Extracts 512-dimensional pooled embedding from mel spectrogram.
        x: (B, 1, 128, T)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        avg_p = self.avg_pool(x).flatten(1)
        max_p = self.max_pool(x).flatten(1)
        emb = torch.cat([avg_p, max_p], dim=1)  # (B, 512)
        return emb

    def forward(self, x, mode="va"):
        """
        mode:
          - 'jamendo': returns tag logits
          - 'va': returns dict with 'va', 'quadrant_logits', and 'embedding'
        """
        emb = self.extract_features(x)

        if mode == "jamendo":
            tag_logits = self.tag_head(emb)
            return tag_logits
        else:
            shared = self.shared_proj(emb)
            va_pred = self.va_head(shared)
            quadrant_logits = self.quadrant_head(shared)
            return {
                "va": va_pred,
                "quadrant_logits": quadrant_logits,
                "embedding": emb
            }
