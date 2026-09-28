import torch
import torch.nn as nn
from transformers import AutoModel


class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.proj = nn.Linear(128, 768)  # Project CNN output to 768 dimensions
        self.beats = AutoModel.from_pretrained('facebook/wav2vec2-base-960h')
        self.beats_proj = nn.Linear(768, 768)  # Project BEATs output to 768 dimensions

    def forward(self, mel_specs):
        # CNN processing
        x = self.cnn(mel_specs).squeeze()
        x = self.proj(x)  # Project CNN output

        # BEATs processing
        beats_output = self.beats(mel_specs.squeeze(1)).last_hidden_state
        beats_output = self.beats_proj(beats_output[:, 0, :])  # Project and take first token's embedding

        # Feature concatenation
        x = torch.cat([x, beats_output], dim=1)  

        return x  # Return combined features
