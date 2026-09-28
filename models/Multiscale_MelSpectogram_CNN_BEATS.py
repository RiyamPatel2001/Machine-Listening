import torch
import torch.nn as nn
from transformers import AutoModel


class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv3d(32, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.proj = nn.Linear(128, 768)  # Project to beats model's input dimension if needed
        self.beats = AutoModel.from_pretrained('facebook/wav2vec2-base-960h')

    def forward(self, mel_specs):
        mel_specs = mel_specs.permute(0, 2, 1, 3, 4)
        x = self.cnn(mel_specs).squeeze()
        x = self.proj(x)
        x = self.beats(x).last_hidden_state[:, 0, :]  # Use beats model and extract embedding
        return x
