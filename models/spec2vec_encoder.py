
import torch
import torch.nn as nn

class Spec2VecEncoder(nn.Module):
    def __init__(self, input_size=(64, 128)):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size[0] * input_size[1], 4096),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(4096, 2048),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
        )

    def forward(self, x):  # (B, 1, H, W)
        return self.encoder(x)
